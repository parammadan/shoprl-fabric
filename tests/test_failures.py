"""Pillar 5 tests: failure classification + OOM-as-operational-event recovery."""
import json

import pytest

from shoprl.platform import (BatchPlan, CheckpointRegistry, FailureClass,
                             JobStore, RecoveryAction, RecoveryController,
                             SimulatedOOM, classify)
from shoprl.platform.jobs import JobState

S = JobState


# --- classification --------------------------------------------------------
def test_classify_oom():
    assert classify(SimulatedOOM("cuda oom")) is FailureClass.OOM
    assert classify(RuntimeError("CUDA out of memory. Tried to allocate")) is FailureClass.OOM


def test_classify_permanent_and_transient_and_unknown():
    assert classify(ValueError("bad config")) is FailureClass.PERMANENT
    assert classify(TimeoutError("slow")) is FailureClass.TRANSIENT
    assert classify(RuntimeError("something odd")) is FailureClass.UNKNOWN


# --- batch math (effective batch held constant) ----------------------------
def test_shrink_halves_microbatch_and_preserves_effective_batch():
    p = BatchPlan(microbatch_size=8, grad_accum_steps=1)   # eff 8
    q = p.shrink()
    assert q.microbatch_size == 4 and q.grad_accum_steps == 2
    assert q.effective_batch == p.effective_batch == 8
    r = q.shrink()
    assert r.microbatch_size == 2 and r.grad_accum_steps == 4 and r.effective_batch == 8


def test_cannot_shrink_below_one():
    assert BatchPlan(microbatch_size=1, grad_accum_steps=8).can_shrink() is False


# --- OOM recovery flow -----------------------------------------------------
def test_oom_adjusts_batch_and_requeues(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("optimize", {"microbatch_size": 8, "grad_accum_steps": 1},
                 max_attempts=5)
    s.claim(now=1.0)
    ctl = RecoveryController(s, events_path=str(tmp_path / "events.jsonl"))
    ev = ctl.handle(s.get(j.id), SimulatedOOM("cuda oom"))
    assert ev.failure_class == "oom"
    assert ev.action == RecoveryAction.RETRY_WITH_ADJUSTMENT.value
    assert ev.simulated is True
    assert ev.resulting_state == "pending"                 # requeued
    back = s.get(j.id)
    assert back.payload["microbatch_size"] == 4            # shrunk for the retry
    assert back.payload["grad_accum_steps"] == 2


def test_oom_restores_checkpoint_when_available(tmp_path):
    s = JobStore(tmp_path / "j.db")
    reg = CheckpointRegistry(tmp_path / "ckpts")
    src = tmp_path / "src"; src.mkdir(); (src / "w").write_bytes(b"weights")
    m = reg.save(src, step=3)
    j = s.create("optimize", {"microbatch_size": 8}, max_attempts=5)
    s.claim(now=1.0)
    ev = RecoveryController(s, registry=reg).handle(s.get(j.id), SimulatedOOM("oom"))
    assert ev.action == RecoveryAction.RESTORE_AND_RETRY.value
    assert ev.restored_ckpt == m.ckpt_id


def test_oom_at_microbatch_one_dead_letters(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("optimize", {"microbatch_size": 1, "grad_accum_steps": 8})
    s.claim(now=1.0)
    ev = RecoveryController(s).handle(s.get(j.id), SimulatedOOM("oom"))
    assert ev.action == RecoveryAction.DEAD_LETTER.value
    assert s.get(j.id).state is S.DEAD_LETTER


# --- other classes ---------------------------------------------------------
def test_permanent_dead_letters_immediately_without_retrying(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("optimize", max_attempts=5)               # plenty of attempts left
    s.claim(now=1.0)
    ev = RecoveryController(s).handle(s.get(j.id), ValueError("bad config"))
    assert ev.action == RecoveryAction.DEAD_LETTER.value
    assert s.get(j.id).state is S.DEAD_LETTER              # not retried despite budget


def test_transient_bounded_retry(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("rollout", max_attempts=3)
    s.claim(now=1.0)
    ev = RecoveryController(s).handle(s.get(j.id), TimeoutError("slow"))
    assert ev.action == RecoveryAction.RETRY.value
    assert s.get(j.id).state is S.PENDING


def test_recovery_disabled_is_plain_retry_no_adjustment(tmp_path):
    s = JobStore(tmp_path / "j.db")
    j = s.create("optimize", {"microbatch_size": 8}, max_attempts=3)
    s.claim(now=1.0)
    ev = RecoveryController(s, enabled=False).handle(s.get(j.id), SimulatedOOM("oom"))
    assert "disabled" in ev.message
    assert s.get(j.id).payload["microbatch_size"] == 8     # NOT adjusted
    assert s.get(j.id).state is S.PENDING                  # just a plain requeue


# --- event persistence (for the dashboard) ---------------------------------
def test_events_are_persisted_as_jsonl(tmp_path):
    s = JobStore(tmp_path / "j.db")
    path = tmp_path / "events.jsonl"
    ctl = RecoveryController(s, events_path=str(path))
    j = s.create("optimize", {"microbatch_size": 8}); s.claim(now=1.0)
    ctl.handle(s.get(j.id), SimulatedOOM("oom"))
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["failure_class"] == "oom"
    assert rows[0]["microbatch_after"] == 4 and rows[0]["simulated"] is True
