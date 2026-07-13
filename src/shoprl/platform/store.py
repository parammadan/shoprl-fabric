"""SQLite-backed job store: persistence + validated, atomic transitions.

Why SQLite: single file, ACID, no server — the right fit for a single-machine
platform. Jobs survive process restart (crash → reopen the DB → the queue and
in-flight state are still there). The transition uses an optimistic
`WHERE id=? AND state=<expected>` guard, so it is atomic: if two processes race
to move the same job, exactly one UPDATE affects a row; the loser gets a
ConcurrentModification (the primitive Pillar 2's workers use to claim jobs
safely).

Scope: single-machine, one SQLite file, one connection per process (workers in
Pillar 2 are separate PROCESSES, each opening its own connection — SQLite's file
locking + WAL handle the concurrency). Real and fully functional, not simulated.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from shoprl.platform.jobs import Job, JobState, assert_transition

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    state        TEXT NOT NULL,
    payload      TEXT NOT NULL,
    attempts     INTEGER NOT NULL,
    max_attempts INTEGER NOT NULL,
    error        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
"""


class JobNotFound(KeyError):
    pass


class ConcurrentModification(Exception):
    """The job was not in the expected state at UPDATE time (lost a race)."""


class JobStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")   # better multi-process concurrency
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- writes ----------------------------------------------------------
    def create(self, kind: str, payload: dict | None = None, max_attempts: int = 3) -> Job:
        job = Job(kind=kind, payload=payload or {}, max_attempts=max_attempts)
        self.conn.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)",
            (job.id, job.kind, job.state.value, json.dumps(job.payload),
             job.attempts, job.max_attempts, job.error, job.created_at, job.updated_at),
        )
        self.conn.commit()
        return job

    def transition(self, job_id: str, to_state: JobState,
                   error: str | None = None, bump_attempt: bool = False) -> Job:
        """Validated + atomic. Raises InvalidTransition (illegal edge) or
        ConcurrentModification (state changed underneath us)."""
        job = self.get(job_id)
        assert_transition(job.state, to_state)          # illegal edge -> raise
        attempts = job.attempts + (1 if bump_attempt else 0)
        now = time.time()
        cur = self.conn.execute(
            "UPDATE jobs SET state=?, attempts=?, error=?, updated_at=? "
            "WHERE id=? AND state=?",
            (to_state.value, attempts, error, now, job_id, job.state.value),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise ConcurrentModification(
                f"{job_id} was not in {job.state.value} at update time")
        return self.get(job_id)

    # --- reads -----------------------------------------------------------
    def get(self, job_id: str) -> Job:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return self._row_to_job(row)

    def list(self, state: JobState | None = None) -> list[Job]:
        if state is None:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE state=? ORDER BY created_at",
                (state.value,)).fetchall()
        return [self._row_to_job(r) for r in rows]

    def counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) c FROM jobs GROUP BY state").fetchall()
        return {r["state"]: r["c"] for r in rows}

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"], kind=row["kind"], state=JobState(row["state"]),
            payload=json.loads(row["payload"]), attempts=row["attempts"],
            max_attempts=row["max_attempts"], error=row["error"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )
