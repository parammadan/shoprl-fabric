# ShopRL Fabric — Project Brief

> A self-contained briefing document. Hand this to an LLM (or a reader) to give
> full context on what this project is, how it's built, what's real vs.
> simulated, what it measured, and how to run it. Written to be honest: every
> metric here was measured, and everything simulated is labelled.

---

## 1. One-paragraph positioning

ShopRL Fabric is a **production-style miniature RL post-training platform** for a
shopping-recommendation LLM. It validates the **architecture, correctness,
observability, and failure semantics** of `rollout → reward → optimize` systems
under **individual, single-laptop / single-GPU constraints**. It has two layers:
(1) a **real RL core** — GRPO / RLOO / PPO fine-tuning of a small LLM with a
verifiable rule-based reward, run for real on a GPU; and (2) a **platform layer**
of six pillars (job state machine, queue-decoupled workers, trajectory lineage,
atomic checkpoint registry, failure/OOM recovery, and a live dashboard) that make
the training loop operable and fault-tolerant. Distributed behaviour is validated
**locally via multiple OS processes** and documented as a scale-out design — it is
**not** a measured multi-node claim.

**Honest scope statement (use verbatim):** *"A production-style miniature RL
post-training platform that validates the architecture, correctness,
observability, and failure semantics of rollout→reward→optimize systems under
individual, single-GPU constraints. Distributed behavior is validated locally via
multiple processes and documented as a scale-out design, not a measured
multi-node claim."*

---

## 2. Constraints & environment (why the design looks the way it does)

- **Hardware:** an **8 GB M1 MacBook** for development (no CUDA; MPS thrashes swap
  on training backward passes, so training smoke runs on CPU), plus **short bursts
  on a single cloud GPU** (AWS spot `g5.xlarge` = A10G 24 GB; also Kaggle T4 /
  Modal). ~\$200 of cloud credits total.
- **Model:** `Qwen/Qwen3-0.6B` (small enough to LoRA-finetune on one GPU).
- **One person.** Every feature is something one engineer can build, understand,
  and defend. The project deliberately prefers **fewer components built deeply**
  over breadth.
- **Language/stack:** Python 3.12, `pydantic` for config + schemas, `pytest`
  (166 tests), SQLite for persistence, PyTorch + `transformers` + `peft` (LoRA)
  for the RL core, optional vLLM for fast rollout on GPU.
- **Cost discipline:** every cloud run is terminated and verified torn down.

---

## 3. The task and the reward (why it's verifiable)

**Task:** given a shopping request with constraints (e.g. "a laptop under \$1200
with ≥16 GB RAM"), the model recommends products from a synthetic catalog.

**Reward is programmatic and verifiable — there is no reward model.** A synthetic
catalog is generated deterministically and is the **ground truth**. The reward
function parses the model's response and scores it against that truth, so the
signal is fast, cheap, reproducible, and impossible to game by flattering a
learned critic. Components (`src/shoprl/reward/`):

- **budget_compliance** — do recommended items actually meet the budget/constraints (checked against catalog truth, not the model's stated specs)?
- **catalog_groundedness** + **is_hallucinated** — are the cited SKUs real? Are stated specs consistent with the catalog? Invented SKUs / spec lies are penalised.
- **attribute_coverage** — does it address the requested attributes?
- **response_quality** — format + comparative reasoning.
- Composite weights ≈ 0.25 / 0.25 / 0.25 / 0.15 / 0.10, hallucination penalty −0.50, range ≈ [−0.5, 1.0].

There is also a **multi-turn Shopping Environment** (`src/shoprl/env/`): a
`ShopEnv` with `reset/step`, cart/budget/filter state, and an episode-level reward
(`episode_reward`) plus credit assignment (`assign_credit`, uniform or discounted)
— the substrate for trajectory-level RL.

---

## 4. The RL core (real, taught + built with WHY comments)

Fine-tuning uses **LoRA**: the adapter *is* the policy; disabling the adapter
gives the frozen reference policy for KL control (no second model in memory). Key
memory trick: **logsumexp log-probs** instead of a full-vocab softmax over Qwen's
~152k vocabulary, and **gradient checkpointing** to fit T² attention activations.

Three algorithms share one `RLTrainer` interface (`src/shoprl/rl/`), differing
only in how they compute the advantage baseline:

| Algo | Baseline | Notes |
|---|---|---|
| **GRPO** | group mean / std | standard group-relative advantage |
| **RLOO** | leave-one-out mean | unbiased; no std normalisation |
| **PPO** | learned value head (critic) | advantage = reward − V, + value loss |

All three use a **clipped surrogate** and a **k3 KL estimator** vs. the frozen
reference.

**The one big bug (worth knowing):** early runs didn't learn — entropy sat at
~4.4, `grad_norm` ~0, output ≈ random. Cause: rollouts were generated in
`model.train()` mode with gradient checkpointing **on**, which nulls the KV cache
→ degenerate generation → flat reward → no gradient. **Fix:** generate rollouts in
`model.eval()` + `gradient_checkpointing_disable()`; switch to `train()` +
checkpointing **only** for the loss forward pass. After the fix: entropy
4.4 → 0.017, `grad_norm` 0 → 2.8. (Bisected on CPU.)

---

## 5. Real measured results (nothing fabricated)

- **Reward quality (real Qwen3-0.6B, n=40):** mean +0.82; groundedness 0.93,
  format 0.95, coverage 0.83, comparison 0.48; hallucination 12.5% (down from
  ~33% after a parser fix; the residual is real model garbling, not a parser
  artifact). Base model is already strong on the easy components → the task is
  **near-saturated**, so held-out reward gains from RL are small.
- **Algorithm comparison (real, AWS A10G, 30 steps, identical bf16 / batch 8,
  held-out n=64):** held-out reward gain ≈ 0 for all three (task saturated). The
  **substantive result is KL-stability control:**
  - **RLOO — final KL 0.015 / max 0.22** (quietest, simplest, equal efficiency)
  - **GRPO — final KL 0.58 / max 0.99**
  - **PPO — KL 6.78 (blew up)** — the fresh critic destabilised training with no
    reward benefit.
  The declarative alerting fired **CRITICAL kl_blowup ×13 on PPO** and stayed
  **quiet on RLOO** — an honest, legible signal of the difference.
- **Over-optimisation finding:** a 100-step run went **flat** (0.836 → 0.840, KL
  climbing) — more steps ≠ better; best regime ≈ 30 steps.
- **Efficiency profiling (real, A10G):** **rollout is 82–84% of wall-clock**,
  optimize ~17%, **reward ~0%**. So reward-worker parallelism is moot, and the
  real levers are vLLM + decoupled rollout. Async rollout and reward-workers gave
  **no gain on a single GPU** (measured negative result, reported as such).
- **Known gap:** the vLLM throughput leg didn't run — `pip install vllm` pulled a
  CUDA-13 build conflicting with the AMI's CUDA-12.8 (`libnvrtc.so.13`). Fix is
  documented (pin vLLM to the AMI's CUDA / separate image); the headline rollout
  lever is therefore **unmeasured**, and that's stated openly.

---

## 6. The platform layer — six pillars

New additive package **`src/shoprl/platform/`**. It changes **nothing** in the RL
core; all prior tests still pass. Guiding principle throughout: single machine,
label everything simulated, give each pillar a one-line "real vs simulated"
truth. Built and taught one pillar at a time.

### Pillar 1 — Job state machine + SQLite persistence
`jobs.py`, `store.py`
- Explicit lifecycle: `PENDING → RUNNING → {SUCCEEDED | FAILED} `, `FAILED →
  {RETRYING → PENDING | DEAD_LETTER}`, plus `CANCELLED`. Terminal states have no
  exits. Illegal transitions raise `InvalidTransition` — the lifecycle is a
  contract, not a free-form status string.
- SQLite (WAL) persistence: jobs survive process restart. `transition()` is
  validated **and atomic** via an optimistic `UPDATE … WHERE id=? AND
  state=<expected>`; a lost race yields `ConcurrentModification`. This is the
  concurrency-safe claim primitive everything else builds on.
- **Real vs simulated:** *real* single-machine job state machine that survives
  restart; nothing simulated.

### Pillar 2 — Queue-decoupled workers (local processes)
`workers.py` (+ store additions)
- The store **is** the queue. Producers `create()`; workers `claim()`
  independently. `run_local_pool` spawns **N local OS processes** that drain the
  queue safely (SQLite file locking + WAL). **Explicitly local processes, not a
  distributed fleet.**
- **Atomic claim** (single ownership via the `WHERE state='pending'` guard),
  **leases + `renew_lease` heartbeat**, and a **`reap_expired` reaper** that
  requeues jobs whose worker died (the answer to "stuck RUNNING forever").
- **Bounded retry → dead-letter** (`max_attempts`), and **at-least-once
  delivery** with an **idempotency ledger** (`record_result` = `INSERT OR
  IGNORE`; workers skip already-resulted jobs).
- **Real vs simulated:** claiming/leases/retry/idempotency are *real*, validated
  across real OS processes; **worker death is simulated** (kill/stall a process),
  but detection + recovery are real; delivery is honestly **at-least-once**, not
  exactly-once.

### Pillar 3 — Trajectory schema + lineage
`trajectory.py`, `traj_store.py`
- **Pydantic** models (deliberately, vs. the dataclass `Job`) because trajectories
  cross process/disk boundaries and get replayed → validate at the boundary:
  rewards/logprobs **finite**, steps **non-empty and contiguous 0..n-1** (a
  dropped/duplicated turn would misalign credit assignment).
- **Lineage / provenance:** each trajectory carries `policy_id`, `job_id`,
  `prompt_id`, `seed`, and `parent_id`. `derive()` mints a child pointing at its
  parent → a derivation graph. `TrajectoryStore` persists validated JSON with
  provenance lifted into indexed columns; queries `by_job / by_policy / children /
  ancestry` (root-first chain walk).
- **Real vs simulated:** fully *real* validated schema + persisted, queryable
  provenance.

### Pillar 4 — Checkpoint registry with atomic write
`checkpoints.py`
- Fixes the legacy `save_checkpoint`, which writes straight into the final path
  (a crash mid-write leaves a corrupt dir that looks valid).
- **Atomic recipe:** `stage → copy files (sha256 + size each) → write manifest →
  fsync → os.replace(staging → final)`. `os.replace` is the **single commit
  point**; a crash earlier leaves only a `.staging/` orphan (swept later), never a
  resolvable half-written checkpoint.
- **Corruption detection:** `verify()` re-hashes every file vs. the manifest;
  `resolve()` / `load_state()` verify **before** returning, so a resume never
  reads corruption. The on-disk manifest **is** the registry (no separate DB
  index to drift). **Resume-equivalence** proven byte-for-byte across a fresh
  instance.
- **Real vs simulated:** *real* atomic writes + sha256 verification; **corruption
  is injected deliberately** in tests (flip a byte / delete a file), detection is
  real. `os.replace` atomicity holds within one filesystem (true here).

### Pillar 5 — Failure classification + OOM-as-operational-event
`failures.py`
- `classify(exc)` → **transient / oom / permanent / unknown**, each with a
  matching response: PERMANENT dead-letters **immediately** (no wasted retries),
  TRANSIENT/UNKNOWN take the bounded retry, OOM is handled as an **operational
  event**.
- **OOM response:** `BatchPlan.shrink()` **halves the microbatch and raises
  grad-accumulation by the same factor**, so peak activation memory drops while
  the **effective batch — and thus the optimization math — is unchanged**;
  restore the latest good checkpoint (Pillar 4); requeue with the shrunk config
  (written into the job payload). At microbatch = 1 it dead-letters (won't fit).
  This is the *automated* version of the real T4/A10G OOM fix from §5.
- **Toggle-able** (`enabled=False` → plain bounded retry). Every action writes a
  **`RecoveryEvent`** (jsonl) — the real feed the dashboard reads.
- **Real vs simulated:** classification, batch math, restore-selection, requeue,
  logging are *real*; **the OOM trigger is `SimulatedOOM`, labelled
  `simulated=True`**, because a laptop can't produce a real CUDA OOM. `gpu_mem_gb`
  honestly returns `None` on CPU/MPS rather than faking a number.

### Pillar 6 — Live dashboard (+ end-to-end pipeline)
`dashboard.py`, `pipeline.py`
- **`pipeline.py`** wires all six pillars into one runnable flow on **real data**:
  enqueue rollout jobs → local worker processes run them → produce
  reward-scored, lineage-tagged trajectories (using the **real** `compute_reward`)
  → an optimize job checkpoints the step atomically → a deliberately injected
  `SimulatedOOM` is recovered automatically → all state persists. It uses
  `StubRolloutEngine` for text so it runs anywhere; **reward numbers are genuine,
  only token generation is a stub, and there is no gradient update so it produces
  no KL** (and never pretends to).
- **`dashboard.py`** renders a **self-contained HTML** view from persisted state
  only — job states, reward per policy version, the checkpoint registry with a
  **live integrity re-check** (`OK` / `CORRUPT`), and recovery events labelled
  `SIMULATED`. It imports **zero trainer internals**. If a **real** RL-trainer
  `metrics.jsonl` is supplied it shows that KL/entropy verbatim; otherwise it says
  so instead of inventing a curve.
- **Real vs simulated:** the orchestration, reward, persistence, atomicity,
  recovery, and dashboard are *real*; workers are local processes, the OOM is a
  labelled simulation, and token generation is a stub.

---

## 7. Explicitly deferred (designed / documented, NOT built)

Multi-node GPU training, FSDP / tensor parallelism, Kubernetes, NCCL,
thousands of workers, Prometheus, distributed storage. These are described as a
scale-out design, not implemented, and not claimed as measured.

---

## 8. Repository map

```
src/shoprl/
  config.py            pydantic config (one YAML drives a run)
  cli.py  train.py     entry points
  task.py              retrieve → shortlist → LLM prompt builder
  data/                deterministic synthetic catalog + prompts (ground truth)
  reward/              verifiable rule-based reward (parse, functions, composite)
  env/                 multi-turn ShopEnv + episode reward + credit assignment
  rollout/             RolloutEngine ABC + Stub / HF(MPS) / vLLM(GPU) engines
  grpo/                RL math: advantages, kl, logprobs, loss, trainer
  rl/                  shared RLTrainer + GRPO/RLOO/PPO adapters + run/report
  eval/                baseline + reward_report (before/after, distributions)
  bench/               phase profiler + A/B harness (rollout/reward/packing)
  observability/       metrics.jsonl dashboard (6 panels) + declarative alerts
  platform/            THE 6 PILLARS:
    jobs.py store.py           P1 state machine + SQLite
    workers.py                 P2 queue-decoupled local-process workers
    trajectory.py traj_store.py P3 schema + lineage
    checkpoints.py             P4 atomic checkpoint registry
    failures.py                P5 classification + OOM recovery
    pipeline.py dashboard.py   P6 end-to-end + live dashboard
tests/                166 tests (pytest)
configs/              YAMLs: dev/stub/smoke + grpo_* + compare_* + sweep_* + bench_*
```

Two companion repos exist: the code repo (private) and a **private docs repo**
`shoprl-fabric-docs` (PROGRESS, RUN_ON_AWS, KAGGLE_RUN, MODAL_RUN, COMPARE_RUN,
BENCHMARK_RUN, ARCHITECTURE, RESULTS, EFFICIENCY).

---

## 9. How to run

```bash
# setup (brew Python 3.12; system python 3.9 is unusable)
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # add ".[hf]" for the ML path

# full test suite (166)
.venv/bin/python -m pytest -q

# end-to-end platform pipeline (no GPU / no model download needed)
.venv/bin/python -m shoprl.platform.pipeline --root runs/pipeline
#   -> jobs.db, trajectories.db, checkpoints/, recovery_events.jsonl, pipeline_metrics.jsonl

# render the platform dashboard from that persisted state
.venv/bin/python -m shoprl.platform.dashboard --root runs/pipeline
#   optionally overlay REAL RL metrics: --metrics runs/<exp>/metrics.jsonl

# real RL training / comparison (needs a GPU + HF token)
.venv/bin/python -m shoprl.rl.run --config configs/compare_rloo.yaml \
    --n-prompts 64 --num-samples 2 --out results/rloo.json
```

---

## 10. Test coverage of the platform layer

103 tests before the platform layer → **166 after**:
P1 +8, P2 +14 (incl. a real multi-process queue drain), P3 +11, P4 +10, P5 +11,
P6/end-to-end +9 (incl. the full pipeline through real OS processes + OOM
recovery, and dashboard corruption-flagging). Every pillar has unit +
integration + failure-injection tests; nothing advanced past a broken pillar.

---

## 11. What this project demonstrates (talking points)

- **RL post-training fundamentals**, built not imported: group-relative /
  leave-one-out / critic baselines, KL control against a LoRA-disabled reference,
  clipped surrogate, and a debugged real "it isn't learning" failure.
- **Honest experimental judgement:** reported KL-stability (RLOO ≪ GRPO ≪ PPO) as
  the real result on a saturated task, rather than headlining a noisy reward
  delta; reported a *negative* async/parallelism result; flagged the unmeasured
  vLLM leg.
- **Production systems thinking at laptop scale:** a job state machine with atomic
  transitions, queue-decoupled workers with leases + reaper + dead-letter +
  idempotency, provenance/lineage, atomic corruption-checked checkpoints,
  classified failure recovery with OOM-as-operational-event, and an observability
  surface — each with explicit failure modes and an honest real-vs-simulated line.
- **Discipline:** config-driven, 166 tests, cost-safe cloud usage, and a hard rule
  that **no reported metric is fabricated** and **everything simulated is
  labelled**.
```
