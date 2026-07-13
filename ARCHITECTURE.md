# ShopRL Fabric — Architecture

A single-machine, production-style RL post-training platform. Real training flows
through the platform; the platform is not a side-car. Scope is deliberately one
laptop + one occasional cloud GPU — every "distributed" behaviour is local
processes, labelled as such, with a documented scale-out path.

## 1. Component map

```
                         ┌────────────┐
                         │ Ops console │  (Streamlit, HTTP-only)
                         └─────┬───────┘
                               │ REST
                         ┌─────▼───────┐
                         │  FastAPI    │  api.py  (validated boundary)
                         └─────┬───────┘
              submit_training  │  reads (overview/scheduler/registries)
                         ┌─────▼───────┐
                         │  JobStore   │  jobs.db  (state machine + queue)
                         └─────┬───────┘
                    admit      │  claim_priority (gpu slot)
                         ┌─────▼───────┐
                         │  Scheduler  │  gpu_slots / priority / backfill
                         └─────┬───────┘
             serve_pending     │  reap → admit → heartbeat → run → complete/fail
                         ┌─────▼───────────────┐
                         │  Worker (control.py) │  in-process, gpu_slots=1
                         └─────┬───────────────┘
                    run_through_platform (the ONE path)
        ┌──────────────────────┼───────────────────────────────┐
        ▼            ▼          ▼            ▼          ▼         ▼
   Preflight   RLTrainer   Checkpoint    Policy    Trajectory  Experiment
   (fail-fast) (GRPO/RLOO  Registry      Registry  Store       Registry
               /PPO)       (atomic+sha)  (v{n})    (tagged)    (RunRecord)
                               │            │          │          │
                               └──────── Artifact Registry (lineage DAG) ┘
                                            │
                                      Metrics (metrics.jsonl) → RL-metrics dashboard
```

There is **one** execution path: API → JobStore → Scheduler → Worker →
`run_through_platform` → RLTrainer + registries. The `shoprl.rl.run` CLI is the
direct single-run entry and calls the *same* `run_through_platform`; it is not an
alternate path (the previous `--no-platform` branch was removed).

## 2. Job lifecycle (state machine)

```
            submit
              │
              ▼
          ┌────────┐  claim (scheduler)   ┌─────────┐  complete   ┌───────────┐
          │PENDING │ ───────────────────▶ │ RUNNING │ ──────────▶ │ SUCCEEDED │ (terminal)
          └───┬────┘                      └────┬────┘             └───────────┘
              │ pause                          │ fail (atomic)
              ▼                                 │
          ┌────────┐  resume                    ├─▶ PENDING  (attempts < max, requeue)
          │ PAUSED │ ──▶ PENDING                └─▶ DEAD_LETTER (attempts ≥ max) (terminal)
          └───┬────┘
              │ cancel
              ▼
          CANCELLED (terminal)          [RUNNING/PENDING/PAUSED → CANCELLED]
```

- **Transitions are validated** against an explicit graph (`jobs.py::_TRANSITIONS`)
  and applied **atomically** with an optimistic guard
  (`UPDATE … WHERE id=? AND state=?`) — a lost race yields `ConcurrentModification`.
- **`fail()` is a single atomic write.** The logical path
  `RUNNING → FAILED → (RETRYING → PENDING | DEAD_LETTER)` is validated edge-by-edge
  but only the *final* state is persisted in one guarded `UPDATE`. A crash cannot
  strand a job in an intermediate state (this was the pre-hardening bug).
- The only non-terminal state a crashed worker can leave is `RUNNING`; the reaper
  covers it (below).

## 3. Worker & recovery flow (leases + heartbeat + reaper — all wired)

```
serve_pending(scheduler):                         # control.py — one pass
  1. reap_expired()          RUNNING jobs whose lease expired → fail() → requeue
  2. schedule()              admit PENDING jobs within gpu_slots (priority, backfill)
  3. renew_lease(TRAIN_LEASE=120s)                # size the lease to the job
  4. with _Heartbeat(...):   background thread renews the lease every LEASE/3
         run_through_platform(...)                # real training
  5. complete() | fail()                          # release the slot
```

- **Lease + heartbeat + reaper is the complete loop.** A healthy long job (minutes)
  is kept alive by the heartbeat; a dead worker stops heartbeating, its lease
  expires within ~`TRAIN_LEASE` seconds, and the next `serve_pending` pass reaps
  and requeues it. All three are invoked on the real path (verified in
  `tests/test_hardening.py`), not decorative.
- **Bounded retry → dead-letter**: reaps/failures bump `attempts`; at
  `max_attempts` the job dead-letters (no infinite reap loop).
- **Cancel vs complete**: a cancel during execution makes `complete()` raise
  `InvalidTransition`, which the worker treats as a benign "cancelled", not a
  failure.
- **Graceful shutdown**: `serve_forever(drain_and_exit=True)` returns when the
  queue is empty; a hard kill leaves at most one `RUNNING` job, reclaimed by the
  reaper on restart.

## 4. Checkpoint lifecycle (single authoritative writer)

```
trainer.model.save_pretrained(TEMP)  →  CheckpointRegistry.save(TEMP)
                                          │  _atomic.atomic_ingest:
                                          │    stage → sha256 each → manifest → fsync
                                          │    → os.replace(staging → final)   [commit]
                                          ▼
                                      READY checkpoint  (verify() re-hashes on resume)
```

- The **CheckpointRegistry is the only authoritative checkpoint writer.** The
  trainer serialises the adapter to a temp dir; the registry atomically ingests
  it; the temp dir is deleted. There is no persistent `save_pretrained` outside
  the registry, and the mid-training periodic save (which bypassed the registry)
  was removed.
- `atomic_ingest` (`_atomic.py`) is shared by the checkpoint **and** policy
  registries — the crash-safe write is implemented once.
- Corruption/truncation is detected by re-hashing against the manifest on
  `verify()` / `resolve()` before any resume.

## 5. Policy lifecycle & on-policy correctness

```
train step → PolicyRegistry.publish(adapter)  → v{n}  (atomic, fingerprinted)
           → RunRecord.policy_version = n
           → trajectories tagged lineage.policy_id = "v{n}"
           → staleness = current_version − trajectory_version   (staleness_gate)
```

- Single-process training is **on-policy** — rollout and optimize share the live
  weights, so the generating policy is the current one; the staleness gate
  confirms this (measured `staleness=0` on the real run).
- `publish` is **fail-closed** on a version collision (`atomic_ingest` raises
  `FileExistsError` rather than clobbering `v{n}`), and `gpu_slots=1` enforces a
  single publisher in practice.
- The `staleness_gate` (warn/reject) exists to catch decoupled/lagging rollout;
  `PolicyClient.refresh/pin` demonstrate the mechanism (and a simulated lagging
  worker).

## 6. Registries & lineage (referential integrity)

| Registry | Key | Crash-safe write | Lineage edge |
|---|---|---|---|
| Experiment (`registry.db`) | `run_id` | SQLite txn | ← trajectory.job_id, → best_checkpoint, policy_version |
| Checkpoint | `ckpt_id` | atomic_ingest | ← run, ← prompt_dataset (artifact) |
| Policy | `v{n}` | atomic_ingest | ← checkpoint (artifact) |
| Prompt | `dataset_version` | atomic rename + hash | → run.dataset_version |
| Trajectory (`trajectories.db`) | `id` | SQLite txn | policy_id="v{n}", job_id=run_id, parent_id |
| Artifact (`artifacts.db`) | `artifact_id` | SQLite txn | parent-edge DAG across all types |

The artifact registry ties them into one DAG: `prompt_dataset → checkpoint →
policy`, `checkpoint → eval_report`. A run's `eval_result`, `best_checkpoint`, and
`policy_version` make every number traceable to the exact config, code
(`git_commit`), dataset (`dataset_version`), and reward (`reward_version`).

---

## 7. Architecture Decision Records

**ADR-1 — SQLite as the queue + state store.**
*Problem:* need a durable job queue + state machine on one machine.
*Decision:* one SQLite file (WAL) as both queue and persistence; atomic optimistic
transitions.
*Alternatives:* Redis/RabbitMQ (broker), Postgres, in-memory.
*Trade-offs:* SQLite gives ACID + zero-ops + single-file durability, at the cost of
single-writer throughput and no network access. Right-sized for one machine.
*Scale-out:* swap the `JobStore` interface for a real broker + Postgres; the
transition-graph + claim semantics port directly.

**ADR-2 — Lease + heartbeat + reaper for worker-death recovery.**
*Problem:* a worker can die mid-job, stranding it `RUNNING`.
*Decision:* claim sets a lease; a heartbeat renews it; a reaper requeues expired
leases.
*Alternatives:* no recovery (manual), a supervisor process, OS-level job control.
*Trade-offs:* lease length trades detection speed vs heartbeat overhead; chose a
generous lease (120s) + heartbeat since training jobs are long and singular.
*Scale-out:* the same lease/reaper model is how distributed queues (SQS visibility
timeout, K8s leases) work — documented, not built.

**ADR-3 — Single authoritative checkpoint writer.**
*Problem:* a non-atomic `save_pretrained` + a post-hoc registry copy defeated the
integrity guarantee.
*Decision:* the trainer writes to a temp dir; the CheckpointRegistry atomically
ingests it; no other persistent writer.
*Alternatives:* trainer writes final + registry indexes (dual write); registry
wraps the trainer's save.
*Trade-offs:* one extra copy of a small LoRA adapter, for a real
atomic+checksummed guarantee.
*Scale-out:* the registry root becomes object storage (S3) with the same
manifest+checksum contract.

**ADR-4 — Own RLTrainer, not TRL.**
*Problem:* need GRPO/RLOO/PPO with full control + a teaching goal.
*Decision:* implement the algorithms behind one `RLTrainer` interface.
*Alternatives:* TRL/OpenRLHF/verl.
*Trade-offs:* more code to own, but full correctness control and no heavy
dependency; the `RLTrainer` interface is the integration seam if a framework
backend is ever wanted.
*Scale-out:* add a `TRLTrainer(RLTrainer)` adapter behind the same interface.

**ADR-5 — Reward is a verifiable rule function, not a model.**
*Problem:* need a fast, reproducible, ungameable reward on one machine.
*Decision:* score responses against a synthetic catalog (ground truth).
*Trade-offs:* less "realistic" than a learned RM, but fully verifiable and cheap
(reward = ~0% of loop time, measured).
*Scale-out:* swap in a reward-worker pool when reward becomes expensive (documented).

---

## 8. SLOs (single-machine, honest)

These are **targets for the platform's own correctness**, not model quality:
- **Job durability:** 100% — no acknowledged job is lost across process restart
  (SQLite WAL; verified by restart tests).
- **Worker-death recovery:** a job orphaned by a dead worker is requeued within
  ≤ `TRAIN_LEASE` (120s) of the next `serve_pending` pass.
- **Checkpoint integrity:** 100% of resolvable checkpoints pass sha256
  verification; a corrupt/truncated checkpoint is never silently resumed.
- **State-transition safety:** no observable intermediate/orphan state after a
  crash mid-recovery (atomic `fail()`).
- **No fabricated metrics:** every reported metric is measured or reported absent.
Not an SLO: model reward improvement (task is near-saturated) or GPU throughput
(hardware-gated, unmeasured).

## 9. Operational runbook

- **Start the API:** `SHOPRL_ROOT=runs/pipeline uvicorn shoprl.platform.api:app`
- **Start a worker:** `python -m shoprl.platform.control --root runs/<name>/platform [--drain]`
- **Submit a run:** `POST /training-jobs {config_path, n_prompts, num_samples}`
  (or `python -m shoprl.rl.run --config <cfg> --out results/<x>.json`).
- **Observe:** `streamlit run src/shoprl/platform/ops_console.py` (embedded or HTTP).
- **Recover a stuck job:** none needed — the reaper requeues on the next worker
  pass. Inspect via `GET /jobs` / `GET /jobs/{id}`.
- **Verify a checkpoint:** `CheckpointRegistry(root).verify(ckpt_id)`.
- **Cost/GPU:** every cloud run is terminated + verified (0 instances/volumes).

## 10. Security & data-governance notes

- **No auth / multi-tenancy** (out of scope): the API is a single-operator
  localhost control plane; do **not** expose it to a network without adding auth.
- **Secrets:** HF / cloud tokens are passed via env / SSM Parameter Store on the
  GPU box, never committed. (Any token pasted into a chat is compromised and must
  be rotated.)
- **Data:** the catalog + prompts are synthetic and deterministic (`seed`), so
  there is no PII and datasets are reproducible + hash-versioned. Trajectories
  store model text only; no user data.
- **Provenance:** every run records `git_commit`, `config_hash`, `dataset_version`,
  `reward_version` for reproducibility/audit.

## 11. Real vs simulated (authoritative)

- **Real:** RLTrainer (GRPO/RLOO/PPO), rule reward, SQLite persistence + atomic
  transitions, local-process workers, lease/heartbeat/reaper recovery, atomic
  checkpoint + sha256 verify, policy versioning + staleness, all registries,
  end-to-end control-plane routing.
- **Simulated (labelled):** OOM trigger (`SimulatedOOM`), worker death (lease
  expiry), lagging worker (`PolicyClient.pin`), trajectory replay (duplicate).
- **Stub (labelled):** `StubRolloutEngine` for the dependency-free pipeline demo.
- **Unmeasured (honest):** GPU throughput/memory/utilization and the vLLM-vs-HF
  benchmark — hardware-gated; tooling is ready and degrades to `{available:False}`
  on CPU.
