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

S = JobState  # local alias for the transition targets used below

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    state            TEXT NOT NULL,
    payload          TEXT NOT NULL,
    attempts         INTEGER NOT NULL,
    max_attempts     INTEGER NOT NULL,
    error            TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    lease_expires_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);

-- Pillar 2 idempotency ledger: one row per job that has produced a result.
-- INSERT OR IGNORE on the PRIMARY KEY makes recording a result idempotent, so
-- a redelivered job cannot double-record. (See workers.py for how a worker
-- checks this before executing.)
CREATE TABLE IF NOT EXISTS job_results (
    job_id     TEXT PRIMARY KEY,
    result     TEXT NOT NULL,
    created_at REAL NOT NULL
);
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
        self.conn.execute("PRAGMA busy_timeout=5000")  # wait, don't fail, on writer contention
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- writes ----------------------------------------------------------
    def create(self, kind: str, payload: dict | None = None, max_attempts: int = 3) -> Job:
        job = Job(kind=kind, payload=payload or {}, max_attempts=max_attempts)
        self.conn.execute(
            "INSERT INTO jobs (id, kind, state, payload, attempts, max_attempts, "
            "error, created_at, updated_at, lease_expires_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (job.id, job.kind, job.state.value, json.dumps(job.payload),
             job.attempts, job.max_attempts, job.error, job.created_at,
             job.updated_at, job.lease_expires_at),
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
        # Leaving RUNNING always releases the lease (set NULL); only claim() /
        # renew_lease() ever set it. Keeps lease bookkeeping in one place.
        cur = self.conn.execute(
            "UPDATE jobs SET state=?, attempts=?, error=?, updated_at=?, "
            "lease_expires_at=NULL WHERE id=? AND state=?",
            (to_state.value, attempts, error, now, job_id, job.state.value),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise ConcurrentModification(
                f"{job_id} was not in {job.state.value} at update time")
        return self.get(job_id)

    def update_payload(self, job_id: str, payload: dict) -> Job:
        """Replace a job's payload. Used by Pillar 5's OOM recovery to hand the
        retry a shrunk microbatch / raised grad-accum before requeueing."""
        cur = self.conn.execute(
            "UPDATE jobs SET payload=?, updated_at=? WHERE id=?",
            (json.dumps(payload), time.time(), job_id))
        self.conn.commit()
        if cur.rowcount == 0:
            raise JobNotFound(job_id)
        return self.get(job_id)

    # --- Pillar 2: claim / lease / lifecycle helpers ---------------------
    def claim(self, kinds: list[str] | None = None,
              lease_seconds: float = 30.0, now: float | None = None) -> Job | None:
        """Atomically claim the oldest PENDING job (optionally of a given kind)
        and move it to RUNNING with a lease. Returns None if no work is
        available. Concurrency-safe across processes: the claim is a single
        `UPDATE ... WHERE id=? AND state='pending'`; if another worker won the
        race (rowcount 0) we try the next candidate. This is how decoupled
        local-process workers pull work without double-claiming."""
        now = time.time() if now is None else now
        lease = now + lease_seconds
        while True:
            if kinds:
                q = ("SELECT id FROM jobs WHERE state='pending' AND kind IN (%s) "
                     "ORDER BY created_at LIMIT 8" % ",".join("?" * len(kinds)))
                rows = self.conn.execute(q, tuple(kinds)).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT id FROM jobs WHERE state='pending' "
                    "ORDER BY created_at LIMIT 8").fetchall()
            if not rows:
                return None
            for row in rows:
                assert_transition(S.PENDING, S.RUNNING)   # documents the edge
                cur = self.conn.execute(
                    "UPDATE jobs SET state='running', lease_expires_at=?, "
                    "updated_at=? WHERE id=? AND state='pending'",
                    (lease, now, row["id"]),
                )
                self.conn.commit()
                if cur.rowcount == 1:
                    return self.get(row["id"])
            # every candidate was contested; loop and re-query

    def renew_lease(self, job_id: str, lease_seconds: float = 30.0,
                    now: float | None = None) -> Job:
        """Heartbeat: a still-alive worker extends its lease so the reaper does
        not reclaim a long-but-healthy job."""
        now = time.time() if now is None else now
        cur = self.conn.execute(
            "UPDATE jobs SET lease_expires_at=?, updated_at=? "
            "WHERE id=? AND state='running'",
            (now + lease_seconds, now, job_id),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            raise ConcurrentModification(f"{job_id} not RUNNING at renew time")
        return self.get(job_id)

    def complete(self, job_id: str) -> Job:
        """RUNNING -> SUCCEEDED."""
        return self.transition(job_id, S.SUCCEEDED)

    def fail(self, job_id: str, error: str) -> Job:
        """RUNNING -> FAILED, then apply the bounded-retry policy: requeue
        (FAILED->RETRYING->PENDING) while attempts remain, else dead-letter
        (FAILED->DEAD_LETTER). Returns the job in its resulting state."""
        j = self.transition(job_id, S.FAILED, error=error, bump_attempt=True)
        if j.attempts >= j.max_attempts:
            return self.transition(job_id, S.DEAD_LETTER, error=error)
        self.transition(job_id, S.RETRYING)
        return self.transition(job_id, S.PENDING)

    def reap_expired(self, now: float | None = None) -> list[Job]:
        """Worker-kill recovery: find RUNNING jobs whose lease has expired (the
        worker died/hung without completing or renewing) and route each through
        the same failure policy as any other failure. Returns the reaped jobs.

        Real: a genuine lease-expiry reaper. In tests the *worker death* is
        simulated (we kill the process / stop renewing); the detection and
        recovery logic here is real."""
        now = time.time() if now is None else now
        rows = self.conn.execute(
            "SELECT id FROM jobs WHERE state='running' "
            "AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
            (now,)).fetchall()
        return [self.fail(r["id"], error="lease expired: worker presumed dead")
                for r in rows]

    # --- Pillar 2: idempotency ledger ------------------------------------
    def record_result(self, job_id: str, result: dict) -> bool:
        """Idempotently record a job's result. Returns True if newly recorded,
        False if a result already existed (redelivery -> no double-write)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO job_results (job_id, result, created_at) "
            "VALUES (?,?,?)", (job_id, json.dumps(result), time.time()))
        self.conn.commit()
        return cur.rowcount == 1

    def has_result(self, job_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM job_results WHERE job_id=?", (job_id,)).fetchone() is not None

    def get_result(self, job_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT result FROM job_results WHERE job_id=?", (job_id,)).fetchone()
        return None if row is None else json.loads(row["result"])

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
            lease_expires_at=row["lease_expires_at"],
        )
