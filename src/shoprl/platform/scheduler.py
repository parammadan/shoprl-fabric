"""Resource scheduler + admission control.

One machine has scarce, non-shareable resources — here, a single GPU. If every
submitted job were allowed to run immediately you'd oversubscribe the GPU and
OOM (or thrash). The scheduler is the gate: jobs are *submitted* (enqueued), and
only *admitted* to RUNNING when a slot of the right resource class is free,
highest priority first. When a job leaves RUNNING (succeeds, fails, or is
cancelled) its slot frees and the next queued job is admitted.

It answers three questions explicitly:
  - Which job gets the GPU?  -> the highest-priority PENDING gpu job (ties: oldest).
  - What if two arrive?      -> higher priority wins the slot; the other stays queued.
  - What when capacity full?  -> nothing is admitted; jobs wait as PENDING.

Design choice: capacity is DERIVED from the live count of RUNNING jobs per
resource (the store is the source of truth), not a separate counter that could
leak if a worker died. So "resource release" is automatic — a job leaving
RUNNING frees its slot with no bookkeeping. CPU jobs are never blocked behind a
GPU-starved job (per-class admission = backfill).

Scope: single machine. `gpu_slots` is normally 1 here; the same logic scales to
a multi-GPU box by raising it. Multi-node is out of scope (documented, not built).
"""
from __future__ import annotations

from dataclasses import dataclass

from shoprl.platform.jobs import Job, JobState
from shoprl.platform.store import JobStore

# Resource classes a job can request. Kept small and explicit on purpose.
GPU = "gpu"
CPU = "cpu"


@dataclass
class ResourceConfig:
    gpu_slots: int = 1
    cpu_worker_slots: int = 4
    max_concurrent_jobs: int = 8      # global cap across all resource classes


class Scheduler:
    def __init__(self, store: JobStore, config: ResourceConfig | None = None):
        self.store = store
        self.cfg = config or ResourceConfig()

    # --- submit ----------------------------------------------------------
    def submit(self, kind: str, payload: dict | None = None, *,
               resource: str = CPU, priority: int = 0, max_attempts: int = 3) -> Job:
        """Enqueue a job (PENDING). It is not admitted until schedule() gives it
        a slot."""
        return self.store.create(kind, payload, max_attempts=max_attempts,
                                 resource=resource, priority=priority)

    # --- admission -------------------------------------------------------
    def _slots(self) -> dict[str, int]:
        return {GPU: self.cfg.gpu_slots, CPU: self.cfg.cpu_worker_slots}

    def schedule(self, now: float | None = None) -> list[Job]:
        """Admit as many queued jobs as capacity allows (per-resource slots and
        the global max_concurrent cap), highest priority first. Returns the jobs
        admitted (now RUNNING) this pass. Idempotent: call it whenever state
        changes (submit / completion / cancel)."""
        admitted: list[Job] = []
        running = self.store.running_counts_by_resource()
        total = sum(running.values())
        slots = self._slots()
        # GPU first — it's the scarce resource; don't let CPU admissions race it.
        for resource in (GPU, CPU):
            free = slots.get(resource, 0) - running.get(resource, 0)
            while free > 0 and total < self.cfg.max_concurrent_jobs:
                job = self.store.claim_priority(resource, now=now)
                if job is None:
                    break                          # no queued job of this class
                admitted.append(job)
                free -= 1
                total += 1
        return admitted

    # --- lifecycle helpers ----------------------------------------------
    def complete(self, job_id: str) -> Job:
        """Mark a job done; its slot frees for the next schedule() pass."""
        return self.store.complete(job_id)

    def fail(self, job_id: str, error: str) -> Job:
        return self.store.fail(job_id, error)

    def cancel(self, job_id: str) -> Job:
        """Cancel a job. If PENDING/PAUSED it is dequeued; if RUNNING its slot is
        released. Either way it ends CANCELLED."""
        job = self.store.get(job_id)
        return self.store.transition(job_id, JobState.CANCELLED)

    # --- observability ---------------------------------------------------
    def status(self) -> dict:
        running = self.store.running_counts_by_resource()
        pending = self.store.pending_counts_by_resource()
        slots = self._slots()
        return {
            "gpu": {"used": running.get(GPU, 0), "slots": slots[GPU],
                    "queued": pending.get(GPU, 0)},
            "cpu": {"used": running.get(CPU, 0), "slots": slots[CPU],
                    "queued": pending.get(CPU, 0)},
            "running_total": sum(running.values()),
            "max_concurrent": self.cfg.max_concurrent_jobs,
            "at_capacity": sum(running.values()) >= self.cfg.max_concurrent_jobs,
        }
