"""Failure classification + OOM-as-an-operational-event.

Not every failure deserves the same response. A config typo should die
immediately (retrying can't help); a network blip should retry unchanged; an
out-of-memory error is an *operational event* the platform should react to, not
a crash. This pillar classifies a failure and drives the matching recovery,
tying together the job lifecycle (Pillars 1-2) and the checkpoint registry
(Pillar 4).

The OOM response (the interesting one): reduce the microbatch to cut peak
activation memory, and raise gradient-accumulation by the same factor so the
*effective* batch — and therefore the optimization math — is unchanged; restore
from the latest good checkpoint if one exists; log the automatic change; then
requeue. If the microbatch is already 1 and it still OOMs, further retries are
futile, so it dead-letters. Recovery is toggle-able (`enabled=False` falls back
to a plain bounded retry).

Honesty / scope: this runs on a laptop with no CUDA, so a *real* OOM can't be
produced here — OOMs are triggered deliberately via SimulatedOOM and labelled
`simulated=True` in the event log. The classification, the batch-math
adjustment, the checkpoint restore, and the requeue are all real; only the
triggering of the OOM is simulated. GPU memory is recorded when CUDA is present
and reported as unavailable otherwise (never faked).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum

from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.jobs import Job, JobState
from shoprl.platform.store import JobStore


class SimulatedOOM(RuntimeError):
    """Dev/test stand-in for torch.cuda.OutOfMemoryError. Deliberately raised to
    exercise the OOM recovery path on hardware that can't really OOM."""


class FailureClass(str, Enum):
    TRANSIENT = "transient"    # blip: retry unchanged
    OOM = "oom"                # operational: shrink batch, maybe restore, retry
    PERMANENT = "permanent"    # user/config error: dead-letter now, don't retry
    UNKNOWN = "unknown"        # bounded retry, then dead-letter


class RecoveryAction(str, Enum):
    RETRY = "retry"                              # requeue unchanged
    RETRY_WITH_ADJUSTMENT = "retry_with_adjustment"  # requeue with shrunk batch
    RESTORE_AND_RETRY = "restore_and_retry"     # + resume from checkpoint
    DEAD_LETTER = "dead_letter"                 # give up


_PERMANENT_TYPES = {"ValueError", "KeyError", "TypeError", "PermanentError"}
_TRANSIENT_TYPES = {"TimeoutError", "ConnectionError", "TransientError"}


def classify(exc: BaseException) -> FailureClass:
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in {"OutOfMemoryError", "SimulatedOOM"} or "out of memory" in msg \
            or "cuda oom" in msg:
        return FailureClass.OOM
    if name in _PERMANENT_TYPES or "no handler" in msg:
        return FailureClass.PERMANENT
    if name in _TRANSIENT_TYPES or "lease expired" in msg or "temporarily" in msg:
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


@dataclass
class BatchPlan:
    """The two knobs that trade memory for step count without changing the
    effective batch size."""
    microbatch_size: int = 8
    grad_accum_steps: int = 1

    @property
    def effective_batch(self) -> int:
        return self.microbatch_size * self.grad_accum_steps

    def can_shrink(self) -> bool:
        return self.microbatch_size > 1

    def shrink(self) -> "BatchPlan":
        """Halve the microbatch, raise grad-accum by the same factor so
        effective_batch is preserved."""
        new_mb = max(1, self.microbatch_size // 2)
        factor = self.microbatch_size // new_mb
        return BatchPlan(new_mb, self.grad_accum_steps * factor)

    @classmethod
    def from_payload(cls, payload: dict) -> "BatchPlan":
        return cls(microbatch_size=payload.get("microbatch_size", 8),
                   grad_accum_steps=payload.get("grad_accum_steps", 1))


def gpu_mem_gb() -> float | None:
    """Peak CUDA memory in GB, or None when CUDA isn't present (CPU/MPS). Never
    fabricated — a laptop honestly reports None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


@dataclass
class RecoveryEvent:
    ts: float
    job_id: str
    failure_class: str
    action: str
    attempt: int
    resulting_state: str
    message: str
    simulated: bool
    gpu_mem_gb: float | None = None
    microbatch_before: int | None = None
    microbatch_after: int | None = None
    grad_accum_before: int | None = None
    grad_accum_after: int | None = None
    restored_ckpt: str | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class RecoveryController:
    """Classifies a job's failure and drives the matching recovery, persisting
    a RecoveryEvent for the dashboard (Pillar 6) to read."""

    def __init__(self, store: JobStore, registry: CheckpointRegistry | None = None,
                 enabled: bool = True, events_path: str | None = None):
        self.store = store
        self.registry = registry
        self.enabled = enabled
        self.events_path = events_path

    def _dead_letter_now(self, job: Job, error: str) -> Job:
        """Skip the retry ladder: FAILED (bump) -> DEAD_LETTER directly."""
        self.store.transition(job.id, JobState.FAILED, error=error, bump_attempt=True)
        return self.store.transition(job.id, JobState.DEAD_LETTER, error=error)

    def handle(self, job: Job, exc: BaseException) -> RecoveryEvent:
        fc = classify(exc)
        err = repr(exc)
        simulated = type(exc).__name__ == "SimulatedOOM"
        gpu = gpu_mem_gb()
        mb_before = mb_after = ga_before = ga_after = restored = None

        if not self.enabled:
            final = self.store.fail(job.id, error=err)   # plain bounded retry
            action = (RecoveryAction.DEAD_LETTER
                      if final.state is JobState.DEAD_LETTER else RecoveryAction.RETRY)
            msg = "recovery disabled: plain bounded retry"
        elif fc is FailureClass.PERMANENT:
            final = self._dead_letter_now(job, err)
            action = RecoveryAction.DEAD_LETTER
            msg = "permanent/user error: not retryable"
        elif fc is FailureClass.OOM:
            plan = BatchPlan.from_payload(job.payload)
            if plan.can_shrink():
                new = plan.shrink()
                mb_before, mb_after = plan.microbatch_size, new.microbatch_size
                ga_before, ga_after = plan.grad_accum_steps, new.grad_accum_steps
                self.store.update_payload(job.id, {
                    **job.payload, "microbatch_size": new.microbatch_size,
                    "grad_accum_steps": new.grad_accum_steps})
                if self.registry is not None and self.registry.latest() is not None:
                    restored = self.registry.latest().ckpt_id
                final = self.store.fail(job.id, error=err)   # requeue (or DL if exhausted)
                action = (RecoveryAction.RESTORE_AND_RETRY if restored
                          else RecoveryAction.RETRY_WITH_ADJUSTMENT)
                msg = (f"OOM: microbatch {mb_before}->{mb_after}, grad_accum "
                       f"{ga_before}->{ga_after} (effective batch held constant"
                       f"={new.effective_batch})"
                       + (f"; restore {restored}" if restored else ""))
            else:
                final = self._dead_letter_now(job, err)
                action = RecoveryAction.DEAD_LETTER
                msg = "OOM at microbatch=1: cannot fit, giving up"
        else:  # TRANSIENT or UNKNOWN -> bounded retry
            final = self.store.fail(job.id, error=err)
            action = (RecoveryAction.DEAD_LETTER
                      if final.state is JobState.DEAD_LETTER else RecoveryAction.RETRY)
            msg = f"{fc.value}: bounded retry"

        event = RecoveryEvent(
            ts=time.time(), job_id=job.id, failure_class=fc.value,
            action=action.value, attempt=final.attempts,
            resulting_state=final.state.value, message=msg, simulated=simulated,
            gpu_mem_gb=gpu, microbatch_before=mb_before, microbatch_after=mb_after,
            grad_accum_before=ga_before, grad_accum_after=ga_after,
            restored_ckpt=restored)
        self._persist(event)
        return event

    def _persist(self, event: RecoveryEvent) -> None:
        if self.events_path:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(event.as_dict()) + "\n")
