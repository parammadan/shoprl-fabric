"""Platform layer: the production-style substrate under the RL core.

Pillar 1 (this): a job state machine + SQLite persistence — a single-machine,
fully-real job lifecycle that survives process restart. Later pillars (workers,
lineage, checkpoint registry, failure handling, dashboard) build on this.

Honest scope: this is a single-machine platform for an individual project. Where
distributed behavior is demonstrated later it's via local processes and clearly
labeled as such — never a multi-node claim.
"""
from shoprl.platform.jobs import (TERMINAL, InvalidTransition, Job, JobState,
                                   can_transition)
from shoprl.platform.store import JobStore

__all__ = ["JobState", "Job", "can_transition", "TERMINAL",
           "InvalidTransition", "JobStore"]
