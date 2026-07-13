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
from shoprl.platform.jobs import InvalidTransition, Job, JobState
from shoprl.platform.registry import (ExperimentRegistry, RunNotFound,
                                      RunRecord, RunStatus)
from shoprl.platform.store import ConcurrentModification, JobNotFound, JobStore
from shoprl.platform.traj_store import TrajectoryNotFound


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
    created_at: float
    updated_at: float

    @classmethod
    def of(cls, j: Job) -> "JobOut":
        return cls(id=j.id, kind=j.kind, state=j.state.value, payload=j.payload,
                   attempts=j.attempts, max_attempts=j.max_attempts,
                   error=j.error, created_at=j.created_at, updated_at=j.updated_at)


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


def create_app(root: str | Path, runs_dir: str | Path = "runs") -> FastAPI:
    root = Path(root)
    runs_dir = Path(runs_dir)
    jobs_db = str(root / "jobs.db")
    registry_db = str(root / "registry.db")
    app = FastAPI(title="ShopRL Fabric Platform API", version="1.0")

    def _job_store() -> JobStore:
        return JobStore(jobs_db)

    def _registry() -> ExperimentRegistry:
        return ExperimentRegistry(registry_db)

    # --- jobs -------------------------------------------------------------
    @app.post("/jobs", response_model=JobOut, status_code=201)
    def create_job(body: JobCreate) -> JobOut:
        s = _job_store()
        try:
            return JobOut.of(s.create(body.kind, body.payload, body.max_attempts))
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

    # --- trajectories -----------------------------------------------------
    @app.get("/trajectories/{traj_id}")
    def get_trajectory(traj_id: str) -> dict:
        try:
            return dash_data.trajectory_detail(root, traj_id)
        except TrajectoryNotFound:
            raise HTTPException(404, f"trajectory {traj_id} not found")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "root": str(root)}

    return app


# Module-level app for `uvicorn shoprl.platform.api:app`.
app = create_app(os.environ.get("SHOPRL_ROOT", "runs/pipeline"),
                 os.environ.get("SHOPRL_RUNS", "runs"))
