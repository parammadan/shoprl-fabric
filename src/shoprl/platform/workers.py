"""Queue-decoupled workers.

A Worker pulls jobs from the JobStore, executes a handler for the job's `kind`,
and drives the job through its lifecycle (complete / fail-with-retry). Workers
are decoupled from producers by the store (the queue): a producer just calls
`store.create(...)`; workers claim work independently. Delivery is *at-least-
once* (a claimed job that a crashed worker never completed is reclaimed by the
reaper and re-run), so handlers must be idempotent — the store's result ledger
(`has_result` / `record_result`) provides the dedup primitive.

Scope / honesty: "workers" here are LOCAL PROCESSES on one machine, not a
distributed fleet. `run_local_pool` spawns N OS processes that share the single
SQLite file; SQLite's file locking + WAL make concurrent claiming safe. This
validates the queue/worker *architecture and concurrency semantics* locally; it
is explicitly not a multi-node claim. Scaling to real remote workers (swap
SQLite for a real queue/DB, add auth/networking) is a documented design, not
something measured here.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import time
from typing import Callable

from shoprl.platform.jobs import Job
from shoprl.platform.store import JobStore

# A handler maps one job to a JSON-serialisable result dict. Raising signals
# failure -> the store applies the bounded-retry / dead-letter policy.
Handler = Callable[[Job], dict]


class Worker:
    def __init__(self, store: JobStore, handlers: dict[str, Handler],
                 lease_seconds: float = 30.0, name: str = "worker"):
        self.store = store
        self.handlers = handlers
        self.lease_seconds = lease_seconds
        self.name = name

    def run_once(self, kinds: list[str] | None = None) -> Job | None:
        """Claim one job and process it. Returns the resulting Job, or None if
        the queue was empty (nothing to do)."""
        job = self.store.claim(kinds, lease_seconds=self.lease_seconds)
        if job is None:
            return None
        # Idempotency: if this job already produced a result (e.g. it was
        # redelivered after a crash that happened *after* the result was
        # recorded but *before* the state flip), don't re-run the side effect —
        # just finish the lifecycle.
        if self.store.has_result(job.id):
            return self.store.complete(job.id)
        handler = self.handlers.get(job.kind)
        if handler is None:
            return self.store.fail(job.id, error=f"no handler for kind={job.kind!r}")
        try:
            result = handler(job) or {}
            self.store.record_result(job.id, result)
            return self.store.complete(job.id)
        except Exception as exc:                       # handler raised -> failure
            return self.store.fail(job.id, error=repr(exc))

    def run_forever(self, kinds: list[str] | None = None,
                    stop: "mp.Event | None" = None,
                    idle_sleep: float = 0.02,
                    drain_and_exit: bool = False) -> None:
        """Loop claiming+processing jobs. Stops when `stop` is set, or (if
        `drain_and_exit`) when the queue is observed empty."""
        while stop is None or not stop.is_set():
            if self.run_once(kinds) is None:           # queue empty right now
                if drain_and_exit:
                    return
                time.sleep(idle_sleep)


# --- Local-process pool demo (NOT a distributed fleet) -------------------
# Handlers must be importable by name so child processes (spawn start method on
# macOS) can reconstruct them. This registry holds demo handlers.
def _echo_handler(job: Job) -> dict:
    """Trivial demo handler: echoes the payload back as the result."""
    return {"echoed": job.payload, "kind": job.kind}


_HANDLER_REGISTRY: dict[str, Handler] = {"echo": _echo_handler}


def _resolve_handler(spec: str) -> Handler:
    """A handler spec is either a registry name ("echo") or a dotted import
    path "module:function" — the latter lets any module (e.g. the pipeline)
    supply handlers to a spawned worker without a shared registry."""
    if ":" in spec:
        mod, fn = spec.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    return _HANDLER_REGISTRY[spec]


def _pool_worker_entry(db_path: str, handler_specs: dict[str, str],
                       lease_seconds: float, name: str) -> None:
    store = JobStore(db_path)
    handlers = {kind: _resolve_handler(spec) for kind, spec in handler_specs.items()}
    try:
        Worker(store, handlers, lease_seconds=lease_seconds, name=name).run_forever(
            drain_and_exit=True)
    finally:
        store.close()


def run_local_pool(db_path: str, n_workers: int = 3,
                   handlers: dict[str, str] | None = None,
                   lease_seconds: float = 30.0) -> None:
    """Spawn `n_workers` LOCAL processes that drain the queue in `db_path` and
    exit. `handlers` maps a job kind -> a handler spec (registry name or
    "module:func"). Producers must have already created the jobs. Demonstrates
    concurrency-safe claiming across real OS processes on one machine."""
    handlers = handlers or {"echo": "echo"}
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_pool_worker_entry,
                         args=(db_path, handlers, lease_seconds, f"w{i}"))
             for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
