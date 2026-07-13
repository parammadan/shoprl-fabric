"""Platform HTTP API (FastAPI) — a thin, validated boundary over the stores.

This is an *interface*, not new logic: every endpoint calls the existing
JobStore / TrajectoryStore / metrics files. It exists so clients (a UI, a CLI, a
scheduler, another service) talk to the platform over a stable, versioned
contract with input validation and clear error codes — instead of importing the
stores directly and coupling to their internals.

    POST   /jobs                    create a job
    GET    /jobs/{id}               fetch a job
    POST   /jobs/{id}/pause         PENDING|RUNNING -> PAUSED
    POST   /jobs/{id}/resume        PAUSED -> PENDING (requeue)
    POST   /jobs/{id}/cancel        -> CANCELLED (if not terminal)
    GET    /runs/{id}/metrics       a training run's metrics.jsonl
    GET    /trajectories/{id}       a trajectory + its lineage

Validation is Pydantic; not-found -> 404; an illegal lifecycle move -> 409 with a
message naming the offending state. Nothing is fabricated: a missing run/metrics
returns 404, not an empty-but-pretend payload.

Scope: single-machine, one SQLite file per store, opened per-request (SQLite
connections are per-thread). Real and fully functional.

Run:  .venv/bin/uvicorn shoprl.platform.api:app   (set SHOPRL_ROOT / SHOPRL_RUNS)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shoprl.platform import dash_data
from shoprl.platform.artifacts import (Artifact, ArtifactNotFound,
                                       ArtifactRegistry, ArtifactType)
from shoprl.platform.jobs import InvalidTransition, Job, JobState
from shoprl.platform.policy import (PolicyNotFound, PolicyRegistry,
                                    PolicyVersion, staleness_report)
from shoprl.platform.registry import (ExperimentRegistry, RunNotFound,
                                      RunRecord, RunStatus)
from shoprl.platform.scheduler import ResourceConfig, Scheduler
from shoprl.platform.store import ConcurrentModification, JobNotFound, JobStore
from shoprl.platform.traj_store import TrajectoryNotFound, TrajectoryStore
from shoprl.observability import alerts as alerts_mod


# --- request / response schemas -------------------------------------------
class JobCreate(BaseModel):
    kind: str = Field(min_length=1, max_length=64,
                      description="e.g. rollout | reward | optimize")
    payload: dict = Field(default_factory=dict)
    max_attempts: int = Field(default=3, ge=1, le=100)


class JobOut(BaseModel):
    id: str
    kind: str
    state: str
    payload: dict
    attempts: int
    max_attempts: int
    error: str | None
    resource: str
    priority: int
    created_at: float
    updated_at: float

    @classmethod
    def of(cls, j: Job) -> "JobOut":
        return cls(id=j.id, kind=j.kind, state=j.state.value, payload=j.payload,
                   attempts=j.attempts, max_attempts=j.max_attempts, error=j.error,
                   resource=j.resource, priority=j.priority,
                   created_at=j.created_at, updated_at=j.updated_at)


class RunMetricsOut(BaseModel):
    run_id: str
    n_steps: int
    metrics: list[dict]


class RunCreate(BaseModel):
    algorithm: str = Field(min_length=1)
    model: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    reward_version: str = Field(min_length=1)
    git_commit: str | None = None
    cost_estimate: dict | None = None


class RunFinish(BaseModel):
    status: Literal["succeeded", "failed", "cancelled"]
    eval_result: dict | None = None
    best_checkpoint: str | None = None
    cost_estimate: dict | None = None


class ArtifactCreate(BaseModel):
    type: ArtifactType
    ref: str = Field(min_length=1)
    run_id: str | None = None
    parents: list[str] = Field(default_factory=list)
    uri: str | None = None
    hash: str | None = None
    metadata: dict = Field(default_factory=dict)


class TrainingJobCreate(BaseModel):
    config_path: str = Field(min_length=1)
    n_prompts: int = Field(default=64, ge=1)
    num_samples: int = Field(default=2, ge=1)
    gpu_mem_gb: float | None = None
    priority: int = 0
    resource: str = "gpu"


def create_app(root: str | Path, runs_dir: str | Path = "runs",
               comparisons_dir: str | Path = "comparisons") -> FastAPI:
    root = Path(root)
    runs_dir = Path(runs_dir)
    comparisons_dir = Path(comparisons_dir)
    jobs_db = str(root / "jobs.db")
    registry_db = str(root / "registry.db")
    app = FastAPI(title="ShopRL Fabric Platform API", version="1.0")

    def _job_store() -> JobStore:
        return JobStore(jobs_db)

    def _registry() -> ExperimentRegistry:
        return ExperimentRegistry(registry_db)

    def _policies() -> PolicyRegistry:
        return PolicyRegistry(root / "policies")

    def _artifacts() -> ArtifactRegistry:
        return ArtifactRegistry(str(root / "artifacts.db"))

    # --- jobs -------------------------------------------------------------
    @app.post("/jobs", response_model=JobOut, status_code=201)
    def create_job(body: JobCreate) -> JobOut:
        s = _job_store()
        try:
            return JobOut.of(s.create(body.kind, body.payload, body.max_attempts))
        finally:
            s.close()

    @app.get("/jobs", response_model=list[JobOut])
    def list_jobs(state: JobState | None = None, resource: str | None = None,
                  limit: int = 200):
        s = _job_store()
        try:
            jobs = s.list(state=state)
            if resource:
                jobs = [j for j in jobs if j.resource == resource]
            return [JobOut.of(j) for j in jobs[:limit]]
        finally:
            s.close()

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str) -> JobOut:
        s = _job_store()
        try:
            return JobOut.of(s.get(job_id))
        except JobNotFound:
            raise HTTPException(404, f"job {job_id} not found")
        finally:
            s.close()

    def _transition(job_id: str, to: JobState, allowed_from: set[JobState] | None,
                    verb: str) -> JobOut:
        s = _job_store()
        try:
            job = s.get(job_id)
            if allowed_from is not None and job.state not in allowed_from:
                raise HTTPException(409, f"cannot {verb} a {job.state.value} job")
            try:
                return JobOut.of(s.transition(job_id, to))
            except InvalidTransition:
                raise HTTPException(409, f"cannot {verb} a {job.state.value} job")
            except ConcurrentModification:
                raise HTTPException(409, "job changed concurrently; retry")
        except JobNotFound:
            raise HTTPException(404, f"job {job_id} not found")
        finally:
            s.close()

    @app.post("/jobs/{job_id}/pause", response_model=JobOut)
    def pause_job(job_id: str) -> JobOut:
        return _transition(job_id, JobState.PAUSED,
                           {JobState.PENDING, JobState.RUNNING}, "pause")

    @app.post("/jobs/{job_id}/resume", response_model=JobOut)
    def resume_job(job_id: str) -> JobOut:
        return _transition(job_id, JobState.PENDING, {JobState.PAUSED}, "resume")

    @app.post("/jobs/{job_id}/cancel", response_model=JobOut)
    def cancel_job(job_id: str) -> JobOut:
        return _transition(job_id, JobState.CANCELLED, None, "cancel")

    # --- experiment registry (runs) --------------------------------------
    @app.post("/runs", response_model=RunRecord, status_code=201)
    def create_run(body: RunCreate) -> RunRecord:
        reg = _registry()
        try:
            rec = RunRecord(algorithm=body.algorithm, model=body.model,
                            config_hash=body.config_hash,
                            dataset_version=body.dataset_version,
                            reward_version=body.reward_version,
                            git_commit=body.git_commit,
                            cost_estimate=body.cost_estimate)
            return reg.save(rec)
        finally:
            reg.close()

    @app.get("/runs", response_model=list[RunRecord])
    def list_runs(algorithm: str | None = None, status: RunStatus | None = None):
        reg = _registry()
        try:
            return reg.list(algorithm=algorithm, status=status)
        finally:
            reg.close()

    @app.get("/runs/compare")
    def compare_runs(ids: str) -> dict:
        run_ids = [i for i in ids.split(",") if i]
        if not run_ids:
            raise HTTPException(400, "provide ?ids=a,b,c")
        reg = _registry()
        try:
            return reg.compare(run_ids)
        except RunNotFound as e:
            raise HTTPException(404, f"run {e.args[0]} not found")
        finally:
            reg.close()

    @app.get("/runs/{run_id}", response_model=RunRecord)
    def get_run(run_id: str) -> RunRecord:
        reg = _registry()
        try:
            return reg.get(run_id)
        except RunNotFound:
            raise HTTPException(404, f"run {run_id} not found")
        finally:
            reg.close()

    @app.post("/runs/{run_id}/start", response_model=RunRecord)
    def start_run(run_id: str) -> RunRecord:
        reg = _registry()
        try:
            return reg.start(run_id)
        except RunNotFound:
            raise HTTPException(404, f"run {run_id} not found")
        finally:
            reg.close()

    @app.post("/runs/{run_id}/finish", response_model=RunRecord)
    def finish_run(run_id: str, body: RunFinish) -> RunRecord:
        reg = _registry()
        try:
            return reg.finish(run_id, RunStatus(body.status),
                              eval_result=body.eval_result,
                              best_checkpoint=body.best_checkpoint,
                              cost_estimate=body.cost_estimate)
        except RunNotFound:
            raise HTTPException(404, f"run {run_id} not found")
        finally:
            reg.close()

    # --- runs: training-metrics file -------------------------------------
    @app.get("/runs/{run_id}/metrics", response_model=RunMetricsOut)
    def run_metrics(run_id: str) -> RunMetricsOut:
        # run_id maps to runs/<run_id>/metrics.jsonl (the RL trainer's output).
        # Path-traversal guard: run_id must be a bare name.
        if "/" in run_id or run_id in ("", ".", ".."):
            raise HTTPException(400, "invalid run_id")
        path = runs_dir / run_id / "metrics.jsonl"
        if not path.exists():
            raise HTTPException(404, f"no metrics for run {run_id}")
        rows = [json.loads(l) for l in path.open() if l.strip()]
        return RunMetricsOut(run_id=run_id, n_steps=len(rows), metrics=rows)

    # --- policy registry + weight sync -----------------------------------
    @app.get("/policies", response_model=list[PolicyVersion])
    def list_policies():
        return _policies().list()

    @app.get("/policies/latest", response_model=PolicyVersion)
    def latest_policy() -> PolicyVersion:
        pv = _policies().latest()
        if pv is None:
            raise HTTPException(404, "no policy published yet")
        return pv

    @app.get("/policies/staleness")
    def policy_staleness() -> dict:
        pv = _policies().latest()
        if pv is None:
            raise HTTPException(404, "no policy published yet")
        ts = TrajectoryStore(str(root / "trajectories.db"))
        try:
            return {"current_version": pv.version,
                    **staleness_report(ts, pv.version)}
        finally:
            ts.close()

    @app.get("/policies/{version}", response_model=PolicyVersion)
    def get_policy(version: int) -> PolicyVersion:
        try:
            return _policies().get(version)
        except PolicyNotFound:
            raise HTTPException(404, f"policy v{version} not found")

    # --- artifact registry -----------------------------------------------
    @app.post("/artifacts", response_model=Artifact, status_code=201)
    def create_artifact(body: ArtifactCreate) -> Artifact:
        reg = _artifacts()
        try:
            return reg.register(body.type, body.ref, run_id=body.run_id,
                                parents=body.parents, uri=body.uri, hash=body.hash,
                                metadata=body.metadata)
        finally:
            reg.close()

    @app.get("/artifacts", response_model=list[Artifact])
    def list_artifacts(type: ArtifactType | None = None, run_id: str | None = None):
        reg = _artifacts()
        try:
            return reg.list(type=type, run_id=run_id)
        finally:
            reg.close()

    @app.get("/artifacts/{artifact_id}", response_model=Artifact)
    def get_artifact(artifact_id: str) -> Artifact:
        reg = _artifacts()
        try:
            return reg.get(artifact_id)
        except ArtifactNotFound:
            raise HTTPException(404, f"artifact {artifact_id} not found")
        finally:
            reg.close()

    @app.get("/artifacts/{artifact_id}/lineage")
    def artifact_lineage(artifact_id: str) -> dict:
        reg = _artifacts()
        try:
            return reg.lineage(artifact_id)
        except ArtifactNotFound:
            raise HTTPException(404, f"artifact {artifact_id} not found")
        finally:
            reg.close()

    # --- trajectories -----------------------------------------------------
    @app.get("/trajectories/{traj_id}")
    def get_trajectory(traj_id: str) -> dict:
        try:
            return dash_data.trajectory_detail(root, traj_id)
        except TrajectoryNotFound:
            raise HTTPException(404, f"trajectory {traj_id} not found")

    # --- training jobs (control plane) -----------------------------------
    @app.post("/training-jobs", response_model=JobOut, status_code=201)
    def submit_training_job(body: TrainingJobCreate) -> JobOut:
        from shoprl.platform.control import submit_training
        s = _job_store()
        try:
            job = submit_training(s, body.config_path, n_prompts=body.n_prompts,
                                  num_samples=body.num_samples, gpu_mem_gb=body.gpu_mem_gb,
                                  platform_root=str(root), priority=body.priority,
                                  resource=body.resource)
            return JobOut.of(job)
        finally:
            s.close()

    @app.get("/training-jobs", response_model=list[JobOut])
    def list_training_jobs():
        from shoprl.platform.control import TRAIN_KIND
        s = _job_store()
        try:
            return [JobOut.of(j) for j in s.list() if j.kind == TRAIN_KIND]
        finally:
            s.close()

    # --- operational reads (for the ops console) -------------------------
    @app.get("/scheduler")
    def scheduler_status() -> dict:
        s = _job_store()
        try:
            return Scheduler(s, ResourceConfig()).status()
        finally:
            s.close()

    @app.get("/overview")
    def overview() -> dict:
        """The operational snapshot: job counts, reward-by-policy, checkpoints
        (+ integrity), recovery events, pipeline metrics. Read-only."""
        return dash_data.snapshot(root)

    @app.get("/checkpoints")
    def checkpoints() -> list[dict]:
        return dash_data.snapshot(root)["checkpoints"]

    @app.get("/trajectories")
    def list_trajectories(limit: int = 200) -> list[dict]:
        out = []
        for t in dash_data.trajectories(root, limit):
            out.append({"id": t.id, "policy_id": t.lineage.policy_id,
                        "reward": t.reward, "kind": t.kind})
        return out

    @app.get("/metrics-runs")
    def metrics_runs() -> list[str]:
        """Run dirs under runs_dir that have a metrics.jsonl (single-run health
        source). Read-only."""
        if not runs_dir.exists():
            return []
        return sorted(d.name for d in runs_dir.iterdir()
                      if (d / "metrics.jsonl").exists())

    @app.get("/comparisons")
    def comparisons() -> list[dict]:
        """Committed PPO/GRPO/RLOO comparison artifacts (historical, measured),
        each enriched with server-side alert counts (via observability.alerts —
        the SAME rules the trainer dashboard used). This is where the RLOO 0.015
        / GRPO 0.58 / PPO 6.78 KL numbers and the PPO critical-KL alert count
        come from. Nothing fabricated; empty list if no artifacts."""
        out = []
        for c in dash_data.comparisons(comparisons_dir):
            crit = warn = 0
            by_rule: dict[str, int] = {}
            # check_run() already folds in per-step check_step() over step_metrics
            # plus run-level checks — use it alone (don't double-count).
            for a in alerts_mod.check_run(c):
                by_rule[a.rule] = by_rule.get(a.rule, 0) + 1
                crit += a.level == alerts_mod.Level.CRITICAL
                warn += a.level == alerts_mod.Level.WARNING
            out.append({
                "algorithm": c["algorithm"], "final_kl": c.get("final_kl"),
                "max_kl": c.get("max_kl"), "reward_gain": c.get("reward_gain"),
                "stability_failures": c.get("stability_failures"),
                "step_metrics": c.get("step_metrics", []),
                "alerts": {"critical": crit, "warning": warn, "by_rule": by_rule},
                "source": c.get("_source")})
        return out

    @app.get("/runs/{run_id}/alerts")
    def run_alerts(run_id: str) -> dict:
        if "/" in run_id or run_id in ("", ".", ".."):
            raise HTTPException(400, "invalid run_id")
        path = runs_dir / run_id / "metrics.jsonl"
        if not path.exists():
            raise HTTPException(404, f"no metrics for run {run_id}")
        rows = [json.loads(l) for l in path.open() if l.strip()]
        active = []
        for m in rows:
            for a in alerts_mod.check_step(m):
                active.append({"level": a.level, "rule": a.rule,
                               "message": a.message, "step": a.step, "value": a.value})
        return {"run_id": run_id, "n_alerts": len(active),
                "max_level": alerts_mod.max_level(
                    [alerts_mod.Alert(a["level"], a["rule"], a["message"]) for a in active]),
                "alerts": active}

    # --- DEV-mode fault injection (SIMULATION) ---------------------------
    @app.post("/dev/kill-worker")
    def dev_kill_worker() -> dict:
        return dash_data.sim_kill_worker(root)

    @app.post("/dev/replay/{traj_id}")
    def dev_replay(traj_id: str) -> dict:
        r = dash_data.sim_duplicate_trajectory(root, traj_id)
        if not r.get("ok"):
            raise HTTPException(404, r.get("error", "cannot replay"))
        return r

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "root": str(root)}

    return app


# Module-level app for `uvicorn shoprl.platform.api:app`.
app = create_app(os.environ.get("SHOPRL_ROOT", "runs/pipeline"),
                 os.environ.get("SHOPRL_RUNS", "runs"),
                 os.environ.get("SHOPRL_COMPARISONS", "comparisons"))
