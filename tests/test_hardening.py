"""Reliability hardening tests: the fault-tolerance mechanisms are now wired
into the real worker loop, and state recovery is crash-safe."""
import os
import pathlib
import tempfile
import time

import pytest

from shoprl.platform import JobStore, ResourceConfig, Scheduler
from shoprl.platform.control import (TRAIN_KIND, _Heartbeat, serve_pending,
                                     submit_training)
from shoprl.platform.jobs import InvalidTransition, JobState
from shoprl.platform.registry import ExperimentRegistry, RunStatus

CFG = "configs/smoke_cpu.yaml"
S = JobState


def _fake_runner(config, n_prompts, num_samples) -> dict:
    d = tempfile.mkdtemp(prefix="fakeckpt-")
    (pathlib.Path(d) / "adapter.safetensors").write_bytes(b"\x00w")
    return {"checkpoint_dir": d,
            "result": {"algorithm": config.algorithm, "reward_gain": 0.0,
                       "final_kl": 0.01, "train_time_s": 1.0},
            "samples": [{"prompt": "p", "response": "ADD X", "reward": 0.5,
                         "components": {"total": 0.5}, "prompt_id": "P-1"}]}


def _sched(root, gpu_slots=1):
    store = JobStore(str(root / "jobs.db"))
    return store, Scheduler(store, ResourceConfig(gpu_slots=gpu_slots))


# --- reaper is wired into the real worker loop (R1) ------------------------
def test_serve_pending_reaps_dead_worker_then_reruns(tmp_path):
    store, sch = _sched(tmp_path)
    job = submit_training(store, CFG, platform_root=str(tmp_path))
    # simulate a worker that CLAIMED the job then died (RUNNING, short lease)
    store.claim_priority("gpu", lease_seconds=1.0, now=0.0)
    assert store.get(job.id).state is S.RUNNING

    res = serve_pending(sch, runner=_fake_runner, now=10_000.0)   # lease long expired
    statuses = [r["status"] for r in res]
    assert "reaped" in statuses                       # reaper reclaimed it (WIRED)
    assert store.get(job.id).state is S.SUCCEEDED     # requeued + re-run in one pass
    assert store.get(job.id).attempts == 1            # the death counted as one failure


def test_reaper_dead_letters_after_exhaustion(tmp_path):
    store, sch = _sched(tmp_path)
    job = submit_training(store, CFG, platform_root=str(tmp_path))
    store.update_payload(job.id, job.payload)         # no-op; keep max_attempts default
    # exhaust: repeatedly claim+die (reap) until dead-letter
    for _ in range(job.max_attempts):
        store.claim_priority("gpu", lease_seconds=1.0, now=0.0)
        store.reap_expired(now=10_000.0)
    assert store.get(job.id).state is S.DEAD_LETTER   # bounded; no infinite reap loop


# --- heartbeat renews the lease so a healthy long job isn't reaped (R2) ----
def test_heartbeat_renews_lease(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    job = store.create("train", resource="gpu")
    store.claim_priority("gpu", lease_seconds=0.3)    # RUNNING, ~now+0.3
    before = store.get(job.id).lease_expires_at
    with _Heartbeat(str(store.db_path), job.id, lease_seconds=0.3):
        time.sleep(0.5)                               # heartbeat renews at ~0.1s intervals
    after = store.get(job.id).lease_expires_at
    assert after > before                             # lease was extended by the heartbeat


# --- cancel vs complete race handled, not mislabeled (R7) ------------------
def test_cancel_during_run_is_not_reported_as_failure(tmp_path):
    store, sch = _sched(tmp_path)
    job = submit_training(store, CFG, platform_root=str(tmp_path))
    side = JobStore(str(store.db_path))

    def cancelling_runner(config, n, ns):
        side.transition(job.id, S.CANCELLED)          # concurrent cancel mid-run
        return _fake_runner(config, n, ns)

    res = serve_pending(sch, runner=cancelling_runner)
    assert res[0]["status"] == "cancelled"            # not "failed"
    assert store.get(job.id).state is S.CANCELLED


def test_scheduler_complete_on_cancelled_raises_invalid_transition(tmp_path):
    store, sch = _sched(tmp_path)
    j = sch.submit("optimize", resource="gpu")
    sch.schedule(now=0.0)                             # RUNNING
    sch.cancel(j.id)                                  # RUNNING -> CANCELLED
    with pytest.raises(InvalidTransition):
        sch.complete(j.id)                            # terminal -> cannot complete


# --- fail() is a single atomic write (R3) ----------------------------------
def test_fail_is_atomic_single_state(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    j = store.create("train", max_attempts=2, resource="gpu")
    store.claim_priority("gpu", now=0.0)
    back = store.fail(j.id, "boom")                   # RUNNING -> PENDING (one write)
    assert back.state is S.PENDING and back.attempts == 1   # never observably FAILED/RETRYING
    store.claim_priority("gpu", now=0.0)
    dead = store.fail(j.id, "boom again")             # RUNNING -> DEAD_LETTER (one write)
    assert dead.state is S.DEAD_LETTER and dead.attempts == 2


# --- gated REAL trainer smoke (slow; downloads/uses the model) -------------
@pytest.mark.skipif(not os.environ.get("SHOPRL_REAL_SMOKE"),
                    reason="slow real-trainer smoke; set SHOPRL_REAL_SMOKE=1 to run")
def test_real_training_through_control_plane(tmp_path):
    store, sch = _sched(tmp_path)
    submit_training(store, CFG, n_prompts=4, num_samples=2, platform_root=str(tmp_path))
    res = serve_pending(sch)                          # REAL runner (run_experiment)
    assert res[-1]["status"] == "succeeded"
    reg = ExperimentRegistry(str(tmp_path / "jobs.db").replace("jobs.db", "registry.db"))
    run = reg.list()[0]
    assert run.status is RunStatus.SUCCEEDED and run.policy_version == 1 and run.best_checkpoint
