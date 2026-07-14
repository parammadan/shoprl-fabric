"""Control plane — route REAL training through job → scheduler → worker → trainer.

This is the missing link the audit flagged: real GRPO/RLOO/PPO training used to
run as a bare in-process loop, bypassing the job store, scheduler, and worker
layer. Here a training run is a JOB: submitted to the store, admitted by the
scheduler (GPU-slot accounting), executed by a worker, and driven through the
platform via the single `run_through_platform` path (registries + checkpoint +
policy + trajectories). No new trainer — it calls the existing one.

    API  ->  submit_training()  ->  JobStore (PENDING, resource=gpu)
                                       |
                        Scheduler.schedule()  (admits within gpu_slots)
                                       |
                        serve_pending()  == the worker loop
                                       |
                        run_through_platform()  ->  real RLTrainer + registries
                                       |
                        scheduler.complete() / fail()

The worker runs the admitted job in-process (a single-GPU box runs one training
job at a time — gpu_slots=1 — so a spawned pool would only add model-reload
cost). `runner` is injectable so the control path is testable without a model.
"""
from __future__ import annotations

import os
import threading

from shoprl.platform.jobs import InvalidTransition, Job
from shoprl.platform.scheduler import GPU, Scheduler
from shoprl.platform.store import JobStore

TRAIN_KIND = "train"
# A training job is long (minutes+) and singular (gpu_slots=1). We hold a
# generous lease and keep it alive with a heartbeat, so a HEALTHY long job is
# never reaped, while a DEAD worker stops heartbeating and its lease expires
# within ~LEASE seconds -> the reaper reclaims it. lease + heartbeat + reaper is
# the complete loop; all three are now wired into serve_pending.
TRAIN_LEASE_SECONDS = 120.0


class _Heartbeat:
    """Renews a job's lease from a BACKGROUND thread (its own DB connection —
    sqlite connections aren't shareable across threads) while the job runs. If
    the worker process dies, the thread dies with it, the lease expires, and the
    reaper reclaims the job."""

    def __init__(self, db_path: str, job_id: str, lease_seconds: float = TRAIN_LEASE_SECONDS):
        self.db_path, self.job_id, self.lease = db_path, job_id, lease_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        store = JobStore(self.db_path)
        try:
            while not self._stop.wait(self.lease / 3.0):
                try:
                    store.renew_lease(self.job_id, lease_seconds=self.lease)
                except Exception:
                    return                    # job left RUNNING (done/cancelled) -> stop
        finally:
            store.close()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)


def submit_training(store: JobStore, config_path: str, *, n_prompts: int = 64,
                    num_samples: int = 2, gpu_mem_gb: float | None = None,
                    platform_root: str | None = None, priority: int = 0,
                    resource: str = GPU) -> Job:
    """Enqueue a training run as a job (PENDING). Not executed until the
    scheduler admits it."""
    return store.create(TRAIN_KIND, {
        "config_path": config_path, "n_prompts": n_prompts,
        "num_samples": num_samples, "gpu_mem_gb": gpu_mem_gb,
        "platform_root": platform_root}, resource=resource, priority=priority)


def _shrink_batch(config):
    """Halve the training batch for OOM recovery: reduce prompts_per_step first
    (keeps the GRPO group size), then num_samples. Returns a shrunk copy, or None
    if already at the minimum (batch can't get smaller -> genuine OOM)."""
    tr, ro = config.training, config.rollout
    new = config.model_copy(deep=True)
    if tr.prompts_per_step > 1:
        new.training.prompts_per_step = tr.prompts_per_step // 2
    elif ro.num_samples > 2:
        new.rollout.num_samples = ro.num_samples // 2
    else:
        return None
    return new


def _empty_cuda() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_with_oom_recovery(config, n_prompts, num_samples, root, *,
                          runner=None, max_oom_retries: int = 4, gpu_mem_gb=None) -> dict:
    """Run training through the platform; on a REAL CUDA OutOfMemoryError (or the
    labelled SimulatedOOM laptop fallback) recover by: shrink the batch, restore
    the latest good checkpoint, and resume — logging a RecoveryEvent each time so
    it shows live in the dashboard. Bounded: when the batch can't shrink further
    it re-raises (a genuine can't-fit OOM)."""
    import json
    import time
    from pathlib import Path

    from shoprl.platform.checkpoints import CheckpointRegistry
    from shoprl.platform.failures import FailureClass, classify
    from shoprl.rl.run import run_experiment, run_through_platform

    runner = runner or run_experiment
    root = Path(root)
    ckpts = CheckpointRegistry(root / "checkpoints")
    events = str(root / "recovery_events.jsonl")
    cfg, resume_from = config, None

    for attempt in range(max_oom_retries + 1):
        try:
            return run_through_platform(cfg, n_prompts, num_samples, str(root),
                                        runner=runner, gpu_mem_gb=gpu_mem_gb,
                                        resume_from=resume_from)
        except BaseException as e:
            if classify(e) is not FailureClass.OOM:
                raise
            simulated = type(e).__name__ == "SimulatedOOM"
            batch_before = cfg.training.prompts_per_step * cfg.rollout.num_samples
            new = _shrink_batch(cfg)
            latest = ckpts.latest()
            resume_from = str(root / "checkpoints" / latest.ckpt_id) if latest else None
            _empty_cuda()
            batch_after = None if new is None else new.training.prompts_per_step * new.rollout.num_samples
            with open(events, "a") as f:                  # -> Recovery tab (live)
                f.write(json.dumps({
                    "ts": time.time(), "failure_class": "oom",
                    "action": "restore_and_retry" if resume_from else "retry_with_adjustment",
                    "microbatch_before": batch_before, "microbatch_after": batch_after,
                    "restored_ckpt": latest.ckpt_id if latest else None,
                    "resulting_state": "pending" if new else "dead_letter",
                    "simulated": simulated,
                    "message": (f"REAL CUDA OOM" if not simulated else "SIMULATED OOM")
                               + f": batch {batch_before}->{batch_after}"
                               + (f", restore {latest.ckpt_id}" if latest else ", no checkpoint yet")}) + "\n")
            if new is None:
                raise                                     # can't fit even at min batch
            cfg = new


def execute_training_job(job: Job, *, runner=None) -> dict:
    """Run one admitted training job through the platform WITH OOM recovery.
    `job` is already RUNNING (claimed by the scheduler). Returns the run refs."""
    from shoprl.config import load_config

    p = job.payload
    config = load_config(p["config_path"])
    root = p.get("platform_root") or os.path.join(
        "runs", config.experiment.name, "platform")
    return run_with_oom_recovery(config, p["n_prompts"], p["num_samples"], root,
                                 gpu_mem_gb=p.get("gpu_mem_gb"), runner=runner)


def serve_pending(scheduler: Scheduler, *, runner=None, now: float | None = None) -> list[dict]:
    """The worker loop (one pass):
      1. REAP dead-worker jobs (expired leases) -> requeued for another worker.
      2. Admit queued jobs via the scheduler (gpu-slot accounting).
      3. Execute each training job under a lease HEARTBEAT, then complete/fail
         it (releasing its slot).
    Returns a summary per admitted job. Non-train jobs admitted here are failed
    with a clear reason rather than stranded RUNNING."""
    db_path = scheduler.store.db_path
    reaped = scheduler.store.reap_expired(now=now)        # worker-death recovery, wired
    results = [{"job_id": j.id, "status": "reaped", "state": j.state.value}
               for j in reaped]

    for job in scheduler.schedule(now=now):
        if job.kind != TRAIN_KIND:
            scheduler.fail(job.id, f"no control-plane handler for kind={job.kind!r}")
            results.append({"job_id": job.id, "status": "failed", "error": "unhandled kind"})
            continue
        scheduler.store.renew_lease(job.id, lease_seconds=TRAIN_LEASE_SECONDS)  # size lease to the job
        try:
            with _Heartbeat(db_path, job.id):             # keep a healthy long job alive
                ref = execute_training_job(job, runner=runner)
        except Exception as e:                            # training/preflight failure
            try:
                scheduler.fail(job.id, repr(e))
            except InvalidTransition:
                pass                                      # already cancelled/reaped: benign
            results.append({"job_id": job.id, "status": "failed", "error": repr(e)})
            continue
        try:
            scheduler.complete(job.id)
            results.append({"job_id": job.id, "status": "succeeded", **ref})
        except InvalidTransition:                         # cancelled mid-run: benign
            results.append({"job_id": job.id, "status": "cancelled",
                            "note": "cancelled during execution", **ref})
    return results


def serve_forever(scheduler: Scheduler, *, poll_interval: float = 1.0,
                  drain_and_exit: bool = False, runner=None) -> None:
    """Worker daemon: keep admitting + running training jobs. With
    drain_and_exit, return once the queue is empty (used by the CLI smoke)."""
    import time
    while True:
        results = serve_pending(scheduler, runner=runner)
        if not results:
            if drain_and_exit:
                return
            time.sleep(poll_interval)


def main() -> None:
    import argparse

    from shoprl.platform.scheduler import ResourceConfig
    ap = argparse.ArgumentParser(prog="shoprl.platform.control",
                                 description="Training worker: drain the job queue")
    ap.add_argument("--root", required=True, help="control root holding jobs.db")
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    args = ap.parse_args()
    store = JobStore(os.path.join(args.root, "jobs.db"))
    sch = Scheduler(store, ResourceConfig(gpu_slots=args.gpu_slots))
    print(f"[worker] serving from {args.root} (gpu_slots={args.gpu_slots})")
    serve_forever(sch, drain_and_exit=args.drain)


if __name__ == "__main__":
    main()
