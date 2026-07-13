"""Job lifecycle as an explicit state machine.

A "job" is one unit of platform work (e.g. a rollout, reward-scoring, or optimize
task — the RLTrainer method boundaries). Rather than a free-form `status` string
that any code can set to anything, the lifecycle is an explicit graph of allowed
transitions. Illegal moves (resurrecting a finished job, skipping execution,
double-completing) are rejected by construction — which makes the system's
behavior auditable, testable, and safe under retries/concurrency.

States:
  PENDING     - created, waiting for a worker
  RUNNING     - a worker claimed it and is executing
  SUCCEEDED   - finished OK (terminal)
  FAILED      - errored; may retry or dead-letter
  RETRYING    - scheduled for another attempt
  DEAD_LETTER - retries exhausted / permanently failed (terminal)
  CANCELLED   - cancelled before completion (terminal)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"              # operator-held; excluded from claiming
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"
    CANCELLED = "cancelled"


# The allowed transition graph. Anything not listed is illegal.
_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.PENDING: {JobState.RUNNING, JobState.PAUSED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.SUCCEEDED, JobState.FAILED, JobState.PAUSED,
                       JobState.CANCELLED},
    JobState.PAUSED: {JobState.PENDING, JobState.CANCELLED},  # resume / cancel
    JobState.FAILED: {JobState.RETRYING, JobState.DEAD_LETTER},
    JobState.RETRYING: {JobState.PENDING, JobState.DEAD_LETTER},
    JobState.SUCCEEDED: set(),     # terminal
    JobState.DEAD_LETTER: set(),   # terminal
    JobState.CANCELLED: set(),     # terminal
}

TERMINAL: frozenset[JobState] = frozenset(
    {JobState.SUCCEEDED, JobState.DEAD_LETTER, JobState.CANCELLED}
)


class InvalidTransition(Exception):
    """Raised when a job is asked to move along an edge the graph forbids."""


def can_transition(src: JobState, dst: JobState) -> bool:
    return dst in _TRANSITIONS[src]


def assert_transition(src: JobState, dst: JobState) -> None:
    if not can_transition(src, dst):
        raise InvalidTransition(f"illegal transition {src.value} -> {dst.value}")


@dataclass
class Job:
    kind: str                                   # rollout | reward | optimize | ...
    payload: dict = field(default_factory=dict)  # job-specific args
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: JobState = JobState.PENDING
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # Pillar 2: a worker holds a time-bounded LEASE while RUNNING. If the worker
    # dies without completing/renewing, the lease expires and a reaper requeues
    # the job. NULL whenever the job is not actively claimed.
    lease_expires_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL
