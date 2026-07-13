"""SQLite persistence + lineage queries for trajectories.

The full trajectory is stored as validated JSON (one row = one episode); the
provenance fields that you actually query by (job, policy version, parent) are
lifted into indexed columns. That split keeps the schema stable (the JSON can
evolve) while making the lineage questions cheap:

  - "all trajectories from job J"        -> by_job()
  - "all trajectories from policy step-7"-> by_policy()
  - "what was derived from trajectory T" -> children()
  - "full derivation chain of T"         -> ancestry()

Scope: real, single-machine, one SQLite file. Not simulated.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from shoprl.platform.trajectory import Trajectory

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trajectories (
    id         TEXT PRIMARY KEY,
    job_id     TEXT,
    policy_id  TEXT NOT NULL,
    parent_id  TEXT,
    prompt_id  TEXT,
    reward     REAL,
    created_at REAL NOT NULL,
    data       TEXT NOT NULL          -- full Trajectory as validated JSON
);
CREATE INDEX IF NOT EXISTS idx_traj_job    ON trajectories(job_id);
CREATE INDEX IF NOT EXISTS idx_traj_policy ON trajectories(policy_id);
CREATE INDEX IF NOT EXISTS idx_traj_parent ON trajectories(parent_id);
"""


class TrajectoryNotFound(KeyError):
    pass


class TrajectoryStore:
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

    def put(self, traj: Trajectory) -> Trajectory:
        """Persist a trajectory (idempotent on id via REPLACE). Validation
        already happened in the Pydantic model, so a stored row is well-formed."""
        lin = traj.lineage
        self.conn.execute(
            "INSERT OR REPLACE INTO trajectories "
            "(id, job_id, policy_id, parent_id, prompt_id, reward, created_at, data) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (traj.id, lin.job_id, lin.policy_id, lin.parent_id, lin.prompt_id,
             traj.reward, traj.created_at, traj.model_dump_json()),
        )
        self.conn.commit()
        return traj

    def get(self, traj_id: str) -> Trajectory:
        row = self.conn.execute(
            "SELECT data FROM trajectories WHERE id=?", (traj_id,)).fetchone()
        if row is None:
            raise TrajectoryNotFound(traj_id)
        return Trajectory.model_validate_json(row["data"])

    def _query(self, where: str, arg) -> list[Trajectory]:
        rows = self.conn.execute(
            f"SELECT data FROM trajectories WHERE {where} ORDER BY created_at",
            (arg,)).fetchall()
        return [Trajectory.model_validate_json(r["data"]) for r in rows]

    def by_job(self, job_id: str) -> list[Trajectory]:
        return self._query("job_id=?", job_id)

    def by_policy(self, policy_id: str) -> list[Trajectory]:
        return self._query("policy_id=?", policy_id)

    def children(self, parent_id: str) -> list[Trajectory]:
        return self._query("parent_id=?", parent_id)

    def ancestry(self, traj_id: str) -> list[Trajectory]:
        """Full derivation chain, root-first, ending at `traj_id`. Follows
        lineage.parent_id upward. Cycle-guarded (ids are immutable, but bound
        the walk defensively)."""
        chain: list[Trajectory] = []
        seen: set[str] = set()
        cur: str | None = traj_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            t = self.get(cur)
            chain.append(t)
            cur = t.lineage.parent_id
        chain.reverse()
        return chain

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM trajectories").fetchone()["c"]

    def prune(self, keep_last_n: int) -> int:
        """Retention: keep the most recent `keep_last_n` trajectories, delete the
        rest. Trajectories accumulate fast (num_samples per prompt per step);
        without this the store grows unbounded. Returns rows deleted."""
        if keep_last_n < 0:
            raise ValueError("keep_last_n must be >= 0")
        cur = self.conn.execute(
            "DELETE FROM trajectories WHERE id NOT IN "
            "(SELECT id FROM trajectories ORDER BY created_at DESC LIMIT ?)",
            (keep_last_n,))
        self.conn.commit()
        return cur.rowcount

    def reward_stats(self) -> dict:
        """Aggregate reward distribution over all persisted trajectories (for
        the dashboard). Reads the indexed reward column directly."""
        vals = [r["reward"] for r in self.conn.execute(
            "SELECT reward FROM trajectories WHERE reward IS NOT NULL").fetchall()]
        if not vals:
            return {"count": 0}
        n = len(vals)
        mean = sum(vals) / n
        return {"count": n, "min": min(vals), "mean": mean, "max": max(vals)}

    def recent(self, limit: int = 500) -> list[Trajectory]:
        """Most-recent trajectories first (for the explorer's picker)."""
        rows = self.conn.execute(
            "SELECT data FROM trajectories ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        return [Trajectory.model_validate_json(r["data"]) for r in rows]

    def reward_by_policy(self) -> list[tuple[str, float, int]]:
        """(policy_id, mean_reward, n) per policy version, oldest first."""
        rows = self.conn.execute(
            "SELECT policy_id, AVG(reward) m, COUNT(*) n FROM trajectories "
            "WHERE reward IS NOT NULL GROUP BY policy_id ORDER BY MIN(created_at)"
        ).fetchall()
        return [(r["policy_id"], r["m"], r["n"]) for r in rows]
