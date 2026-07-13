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

from shoprl.platform.jobs import Job
from shoprl.platform.scheduler import GPU, Scheduler
from shoprl.platform.store import JobStore

TRAIN_KIND = "train"


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


def execute_training_job(job: Job, *, runner=None) -> dict:
    """Run one admitted training job through the platform. `job` is already
    RUNNING (claimed by the scheduler). Returns the run refs."""
    from shoprl.config import load_config
    from shoprl.rl.run import run_experiment, run_through_platform

    p = job.payload
    config = load_config(p["config_path"])
    root = p.get("platform_root") or os.path.join(
        "runs", config.experiment.name, "platform")
    return run_through_platform(
        config, p["n_prompts"], p["num_samples"], root,
        gpu_mem_gb=p.get("gpu_mem_gb"), runner=runner or run_experiment)


def serve_pending(scheduler: Scheduler, *, runner=None, now: float | None = None) -> list[dict]:
    """The worker loop (one pass): admit queued jobs via the scheduler, execute
    each training job, and complete/fail it — releasing its slot. Returns a
    summary per admitted job. Non-train jobs admitted here are failed with a
    clear reason rather than left stranded RUNNING."""
    results = []
    for job in scheduler.schedule(now=now):
        if job.kind != TRAIN_KIND:
            scheduler.fail(job.id, f"no control-plane handler for kind={job.kind!r}")
            results.append({"job_id": job.id, "status": "failed",
                            "error": "unhandled kind"})
            continue
        try:
            ref = execute_training_job(job, runner=runner)
            scheduler.complete(job.id)
            results.append({"job_id": job.id, "status": "succeeded", **ref})
        except Exception as e:                        # training/preflight failure
            scheduler.fail(job.id, repr(e))
            results.append({"job_id": job.id, "status": "failed", "error": repr(e)})
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
