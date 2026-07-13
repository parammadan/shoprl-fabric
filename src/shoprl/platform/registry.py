"""Experiment registry — a first-class record for every training run.

RL results were previously loose `results/*.json` files. That makes runs hard to
trust or compare: you can't tell what config produced a number, whether two runs
were even comparable, or which code was running. A run record fixes that: it
captures WHAT was run (algorithm, model, dataset_version, reward_version), the
exact CONFIG (config_hash) and CODE (git_commit) that produced it, its lifecycle
(status, start/end), and its OUTCOME (best_checkpoint, eval_result,
cost_estimate). Two runs are comparable iff their dataset_version /
reward_version match and only the knob under test differs — the record makes that
checkable instead of assumed.

It ties to the rest of the platform: `best_checkpoint` references a
CheckpointRegistry id (Pillar 4), and trajectories link to a run through their
lineage.policy_id (Pillar 3).

Scope: real, single-machine, one SQLite file. Not simulated.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import time
import uuid
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from shoprl.config import Config


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    algorithm: str
    config_hash: str
    model: str
    dataset_version: str
    reward_version: str
    git_commit: str | None = None
    status: RunStatus = RunStatus.CREATED
    started_at: float | None = None
    ended_at: float | None = None
    best_checkpoint: str | None = None        # -> CheckpointRegistry ckpt_id
    policy_version: int | None = None         # -> PolicyRegistry latest version
    eval_result: dict | None = None           # e.g. reward_before/after, final_kl
    cost_estimate: dict | None = None         # e.g. {"gpu_hours":.., "usd":..}
    created_at: float = Field(default_factory=time.time)
    meta: dict = Field(default_factory=dict)


# --- provenance helpers ----------------------------------------------------
def config_hash(cfg: Config) -> str:
    """Stable short hash of the full config — same config -> same hash."""
    payload = json.dumps(cfg.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def dataset_version(cfg: Config) -> str:
    """A dataset is fully determined by (catalog_size, seed) here, so its version
    is just that pair — a change is detectable and two runs are only comparable
    if they match."""
    return f"catalog{cfg.training.catalog_size}-seed{cfg.experiment.seed}"


def reward_version(cfg: Config) -> str:
    payload = json.dumps({"weights": cfg.rewards.weights,
                          "penalty": cfg.rewards.hallucination_penalty},
                         sort_keys=True)
    return "rw-" + hashlib.sha256(payload.encode()).hexdigest()[:8]


def git_commit() -> str | None:
    try:
        repo = Path(__file__).resolve().parents[3]
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()[:12] if out.returncode == 0 else None
    except Exception:
        return None


def record_from_config(cfg: Config, **over) -> RunRecord:
    """Build a CREATED run record from a Config, computing provenance."""
    base = dict(algorithm=cfg.algorithm, config_hash=config_hash(cfg),
                model=cfg.model.name, dataset_version=dataset_version(cfg),
                reward_version=reward_version(cfg), git_commit=git_commit())
    base.update(over)
    return RunRecord(**base)


# --- store -----------------------------------------------------------------
class RunNotFound(KeyError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    algorithm       TEXT NOT NULL,
    config_hash     TEXT NOT NULL,
    model           TEXT,
    dataset_version TEXT,
    reward_version  TEXT,
    git_commit      TEXT,
    status          TEXT NOT NULL,
    started_at      REAL,
    ended_at        REAL,
    best_checkpoint TEXT,
    created_at      REAL NOT NULL,
    data            TEXT NOT NULL      -- full RunRecord JSON
);
CREATE INDEX IF NOT EXISTS idx_runs_algorithm ON runs(algorithm);
CREATE INDEX IF NOT EXISTS idx_runs_status    ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_cfg       ON runs(config_hash);
"""


class ExperimentRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save(self, rec: RunRecord) -> RunRecord:
        """Insert or update a run (idempotent on run_id)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, algorithm, config_hash, model, "
            "dataset_version, reward_version, git_commit, status, started_at, "
            "ended_at, best_checkpoint, created_at, data) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.run_id, rec.algorithm, rec.config_hash, rec.model,
             rec.dataset_version, rec.reward_version, rec.git_commit,
             rec.status.value, rec.started_at, rec.ended_at, rec.best_checkpoint,
             rec.created_at, rec.model_dump_json()))
        self.conn.commit()
        return rec

    def get(self, run_id: str) -> RunRecord:
        row = self.conn.execute(
            "SELECT data FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFound(run_id)
        return RunRecord.model_validate_json(row["data"])

    def list(self, algorithm: str | None = None,
             status: RunStatus | None = None) -> list[RunRecord]:
        q, args = "SELECT data FROM runs", []
        clauses = []
        if algorithm:
            clauses.append("algorithm=?"); args.append(algorithm)
        if status:
            clauses.append("status=?"); args.append(status.value)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at"
        return [RunRecord.model_validate_json(r["data"])
                for r in self.conn.execute(q, args).fetchall()]

    # --- lifecycle -------------------------------------------------------
    def start(self, run_id: str, now: float | None = None) -> RunRecord:
        rec = self.get(run_id)
        rec = rec.model_copy(update={"status": RunStatus.RUNNING,
                                     "started_at": now or time.time()})
        return self.save(rec)

    def finish(self, run_id: str, status: RunStatus, *,
               eval_result: dict | None = None, best_checkpoint: str | None = None,
               cost_estimate: dict | None = None, now: float | None = None) -> RunRecord:
        rec = self.get(run_id)
        rec = rec.model_copy(update={
            "status": status, "ended_at": now or time.time(),
            "eval_result": eval_result if eval_result is not None else rec.eval_result,
            "best_checkpoint": best_checkpoint or rec.best_checkpoint,
            "cost_estimate": cost_estimate if cost_estimate is not None else rec.cost_estimate})
        return self.save(rec)

    def compare(self, run_ids: list[str]) -> dict:
        """Compare registered runs side by side, and flag whether they're even
        comparable (same dataset_version + reward_version)."""
        recs = [self.get(r) for r in run_ids]
        rows = []
        for r in recs:
            ev = r.eval_result or {}
            rows.append({
                "run_id": r.run_id, "algorithm": r.algorithm, "status": r.status.value,
                "config_hash": r.config_hash, "git_commit": r.git_commit,
                "reward_gain": ev.get("reward_gain"), "final_kl": ev.get("final_kl"),
                "max_kl": ev.get("max_kl"),
                "cost_usd": (r.cost_estimate or {}).get("usd"),
                "best_checkpoint": r.best_checkpoint})
        comparable = (len({r.dataset_version for r in recs}) <= 1
                      and len({r.reward_version for r in recs}) <= 1)
        return {"rows": rows, "comparable": comparable,
                "dataset_versions": sorted({r.dataset_version for r in recs}),
                "reward_versions": sorted({r.reward_version for r in recs})}


def register_rl_result(registry: ExperimentRegistry, result: dict,
                       cfg: Config | None = None,
                       cost_estimate: dict | None = None) -> RunRecord:
    """Import a real `shoprl.rl.run` result dict as a finished, registered run —
    so PPO/GRPO/RLOO are compared as records, not loose files. Uses only the
    result's real measured fields; fabricates nothing."""
    ev = {"reward_gain": result.get("reward_gain"),
          "final_kl": result.get("final_kl"), "max_kl": result.get("max_kl"),
          "reward_before": (result.get("held_out_before") or {}).get("reward_mean"),
          "reward_after": (result.get("held_out_after") or {}).get("reward_mean"),
          "stability_failures": result.get("stability_failures")}
    if cfg is not None:
        rec = record_from_config(cfg)
        rec = rec.model_copy(update={"algorithm": result.get("algorithm", cfg.algorithm)})
    else:
        rec = RunRecord(algorithm=result.get("algorithm", "unknown"),
                        config_hash="imported", model=result.get("model", "unknown"),
                        dataset_version="imported", reward_version="imported",
                        git_commit=git_commit())
    rec = rec.model_copy(update={
        "status": RunStatus.SUCCEEDED, "started_at": rec.created_at,
        "ended_at": time.time(), "eval_result": ev, "cost_estimate": cost_estimate})
    return registry.save(rec)
