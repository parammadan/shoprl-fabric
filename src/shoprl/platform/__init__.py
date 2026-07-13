"""Platform layer: the production-style substrate under the RL core.

Pillar 1: a job state machine + SQLite persistence — a single-machine,
fully-real job lifecycle that survives process restart.
Pillar 2: queue-decoupled workers (Worker / run_local_pool) with atomic
claiming + leases, at-least-once delivery, bounded retry -> dead-letter,
worker-death recovery via a lease reaper, and an idempotency ledger.
Pillar 3: a Pydantic trajectory schema (validated episodes) + lineage/provenance
persisted in a TrajectoryStore (query by job / policy version, walk derivation
chains).
Pillar 4: a CheckpointRegistry with atomic write (stage -> hash -> manifest ->
fsync -> os.replace) + per-file sha256 verification (corruption detection) so a
resume never reads a half-written or bit-rotted checkpoint.
Pillar 5: failure classification (classify -> transient/oom/permanent/unknown)
+ a RecoveryController that treats OOM as an operational event (shrink
microbatch / raise grad-accum holding the effective batch constant, restore the
latest checkpoint, requeue) and logs each RecoveryEvent. The dashboard (Pillar
6) builds on these persisted events.

Honest scope: this is a single-machine platform for an individual project. Where
distributed behavior is demonstrated later it's via local processes and clearly
labeled as such — never a multi-node claim.
"""
from shoprl.platform.jobs import (TERMINAL, InvalidTransition, Job, JobState,
                                   can_transition)
from shoprl.platform.checkpoints import (CheckpointCorrupt, CheckpointRegistry,
                                         Manifest)
from shoprl.platform.failures import (BatchPlan, FailureClass, RecoveryAction,
                                      RecoveryController, RecoveryEvent,
                                      SimulatedOOM, classify)
from shoprl.platform.policy import (PolicyClient, PolicyRegistry,
                                    PolicyVersion, staleness, staleness_report)
from shoprl.platform.scheduler import ResourceConfig, Scheduler
from shoprl.platform.store import JobStore
from shoprl.platform.traj_store import TrajectoryStore
from shoprl.platform.trajectory import Lineage, Trajectory, TrajectoryStep
from shoprl.platform.workers import Worker, run_local_pool

__all__ = ["JobState", "Job", "can_transition", "TERMINAL",
           "InvalidTransition", "JobStore", "Worker", "run_local_pool",
           "Trajectory", "TrajectoryStep", "Lineage", "TrajectoryStore",
           "CheckpointRegistry", "CheckpointCorrupt", "Manifest",
           "FailureClass", "RecoveryAction", "RecoveryController",
           "RecoveryEvent", "BatchPlan", "SimulatedOOM", "classify",
           "Scheduler", "ResourceConfig", "PolicyRegistry", "PolicyVersion",
           "PolicyClient", "staleness", "staleness_report"]
