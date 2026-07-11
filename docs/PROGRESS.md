# ShopRL Fabric — Dev Progress Log

Running dev log. Newest-last: append one dated entry per session with the same
structure (Built / Key design decisions + why / Measured / Learned / Open+next).
Keep it concise and factual — only measured results, never invented numbers.

---

## 2026-07-11 — Session 1: foundation → reward → GRPO math core

### Built
- **Scaffold + config**: Python 3.12 venv, packaging, pydantic config loader
  (one YAML per experiment: `configs/dev.yaml`, `configs/stub.yaml`).
- **RolloutEngine interface**: `StubRolloutEngine` (deterministic, no deps) and
  `HFRolloutEngine` (transformers `generate` on Apple MPS, Qwen3-0.6B). vLLM
  left as a cloud-only drop-in behind the same interface.
- **Synthetic data generator** (`shoprl.data`): deterministic laptop catalog
  (JSONL) + prompt generator that stores each prompt's verifiable ground-truth
  answer set (catalog products satisfying its constraints).
- **Reward layer** (`shoprl.reward`): 4 pure functions (budget_compliance,
  catalog_groundedness + is_hallucinated, attribute_coverage, response_quality)
  + weighted composite. Parser extracts claimed SKUs/specs from free text.
- **GRPO math core** (`shoprl.grpo`): group-relative advantages, k3 KL penalty,
  clipped trust-region loss (all unit-tested; torch pieces run on CPU).
- **Live wiring + instrumentation**: `shoprl.task` (retrieve→shortlist→prompt
  builder) and `shoprl.eval.reward_report` (scores real rollouts; prints reward
  distribution, component means, rates, and **within-group reward std** — the
  GRPO signal — with `--dump` JSONL for auditing).
- Tests: 59 passing.

### Key design decisions (+ why)
- **Rule-based, verifiable reward** (no reward model): every score computed
  against the catalog ground truth, so metrics are objective and un-hackable by
  a learned judge. Budget/coverage check the *catalog's* true specs, never the
  specs the model states, so the policy can't pass a constraint by lying.
- **Group-relative, critic-free advantages**: the group mean is an unbiased,
  action-independent baseline, so we drop PPO's value network. Valid only when a
  group has reward spread → we measure within-group std before training.
- **`eps` in advantage standardization**: a flat group (std≈0) yields A≈0 (no
  gradient) instead of 0/0 NaN that would poison the batch.
- **k3 KL estimator** (`exp(Δ)-Δ-1`): always ≥0, unbiased, lower variance than
  the naive `logp_policy-logp_ref`; used as a per-token penalty in the loss.
- **KL leash to a frozen reference**: prevents reward hacking / degeneration by
  penalizing drift from the fluent starting model.
- **Clipped trust-region loss**: PPO-style `min(ρA, clip(ρ)A)` caps how far one
  batch of (possibly stale) rollouts can move the policy; masked token-mean so
  only generated completion tokens contribute.

### Measured (real, no fabrication)
- Real Qwen3-0.6B on grounded task prompts, n=40 completions (std-check run,
  max_new_tokens 96): reward mean +0.75, groundedness 0.91, format 0.90,
  coverage 0.82, comparison 0.46; hallucination fired 22.5%; 100% emitted a
  parseable SKU.
- **Within-group reward std: mean 0.182** (per-prompt: 0.221, 0.348, 0.009,
  0.009, 0.324). 3/5 prompts have strong gradient signal; 2/5 saturated ~0.935
  (flat → ~zero gradient, harmless).
- Parser fix (`LAP-\d{4}`→`LAP-\d+`) makes garbled ids like `LAP-01`/`LAP-00106`
  count as hallucinations directly; remaining flagged cases are genuine model
  degeneracies, not parser artifacts.

### Learned
- **Within-group variance IS the GRPO learning signal.** Advantages are relative
  to each prompt's own group mean, so global reward spread is irrelevant; a
  per-group-flat reward produces no gradient regardless of level.
- **Rollout dominates RL wall-clock**: 40 completions ≈ 28 min on M1/MPS (no
  training). Motivates pushing generation to GPU + vLLM, and async/batched
  rollout as its own optimization.

### Open / next
- **Go/no-go on within-group std: GO.** Signal is healthy (mean 0.18, majority
  of prompts non-flat). Cleared to build the training loop.
- Next: wire the live **rollout → reward → advantage → GRPO update** loop
  (compute per-token log-probs from the policy + frozen reference over rollout
  tokens; single update step on one batch; verify loss/KL/clip metrics move
  sensibly on M1 with the tiny model).
- Later (optional): harden shortlist task to un-saturate flat groups; then the
  multi-process fabric (queue + workers + learner) and checkpoint/resume.
