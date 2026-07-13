"""Platform layer: the production-style substrate under the RL core.

Pillar 1: a job state machine + SQLite persistence — a single-machine,
fully-real job lifecycle that survives process restart.
Pillar 2: queue-decoupled workers (Worker / run_local_pool) with atomic
claiming + leases, at-least-once delivery, bounded retry -> dead-letter,
worker-death recovery via a lease reaper, and an idempotency ledger.
Pillar 3: a Pydantic trajectory schema (validated episodes) + lineage/provenance
persisted in a TrajectoryStore (query by job / policy version, walk derivation
chains). Later pillars (checkpoint registry, failure handling, dashboard) build
on this.

Honest scope: this is a single-machine platform for an individual project. Where
distributed behavior is demonstrated later it's via local processes and clearly
labeled as such — never a multi-node claim.
"""
from shoprl.platform.jobs import (TERMINAL, InvalidTransition, Job, JobState,
                                   can_transition)
from shoprl.platform.store import JobStore
from shoprl.platform.traj_store import TrajectoryStore
from shoprl.platform.trajectory import Lineage, Trajectory, TrajectoryStep
from shoprl.platform.workers import Worker, run_local_pool

__all__ = ["JobState", "Job", "can_transition", "TERMINAL",
           "InvalidTransition", "JobStore", "Worker", "run_local_pool",
           "Trajectory", "TrajectoryStep", "Lineage", "TrajectoryStore"]
