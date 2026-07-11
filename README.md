# ShopRL Fabric

Fault-tolerant RL post-training platform for shopping LLMs. Work in progress — actively building.

A from-scratch reinforcement-learning **post-training infrastructure** for a shopping
recommendation LLM: the full lifecycle — config → rollout → reward → GRPO update →
eval → checkpoint — built to be developed and debugged on a laptop (Apple M1, tiny
model) and scaled to cloud GPU in short bursts.

## Why this exists

A portfolio project modeling production-style RL training infrastructure:
verifiable rewards, a critic-free policy-gradient loop (GRPO), memory-frugal
training (LoRA), and an engine interface that swaps HF `generate` (laptop) for
vLLM (GPU) without touching the learner.

## Task & reward (verifiable, no reward model)

The policy recommends a laptop from a candidate shortlist given a customer's
constraints (budget / RAM / weight / battery). Every reward is computed against a
**synthetic catalog that is ground truth**, so scores are objective and un-hackable:

| Component | Checks |
|---|---|
| `budget_compliance` | recommended SKU's *true* price ≤ max_price |
| `catalog_groundedness` | claimed SKUs exist + stated specs match catalog (`is_hallucinated` flag) |
| `attribute_coverage` | recommended SKU satisfies all constraints |
| `response_quality` | (format, comparison) |

Composite: `0.25·budget + 0.25·ground + 0.25·coverage + 0.15·format + 0.10·comparison − 0.50·hallucinated`.

## Architecture

```
config (YAML) ─▶ RolloutEngine (HF/MPS ⇄ vLLM/GPU) ─▶ reward (vs catalog)
                          │                                    │
                          └──────▶ GRPO: group advantages ─ KL(k3) ─ clipped loss
                                            │
                                   LoRA policy update ─▶ checkpoint (→ HF Hub)
```

- `shoprl.config` — one pydantic-validated YAML per experiment.
- `shoprl.rollout` — `RolloutEngine` interface; `HFRolloutEngine` (M1) / vLLM (GPU).
- `shoprl.data` — deterministic catalog + prompts with verifiable answer sets.
- `shoprl.reward` — 4 pure reward functions + composite.
- `shoprl.grpo` — advantages, k3 KL, clipped loss, LoRA training loop.
- `shoprl.eval` — reward-distribution report + untrained baseline.
- `shoprl.task` — retrieve→shortlist→prompt builder.

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core + tests (model-free)
pip install -e ".[hf]"       # + torch/transformers/peft for real runs
```

## How to run

```bash
# 1. Generate the synthetic dataset (deterministic)
python -m shoprl.data.build --out data --n-catalog 300 --n-prompts 200

# 2. Inspect the reward distribution on real model output
python -m shoprl.eval.reward_report --config configs/dev.yaml --n-prompts 4 --num-samples 8

# 3. Baseline: score the UNTRAINED model on held-out prompts
python -m shoprl.eval.baseline --config configs/grpo_qwen_06b.yaml --out outputs/baseline.json

# 4. Launch the whole GRPO loop from one YAML
python -m shoprl.train --config configs/grpo_qwen_06b.yaml

# CPU smoke (proves the loop end-to-end on an 8GB M1)
python -m shoprl.train --config configs/smoke_cpu.yaml
```

## Tests & Docker

```bash
pytest -q                    # full suite (model-free, fast)
docker build -t shoprl-fabric . && docker run --rm shoprl-fabric   # tests in a clean env
```

## Roadmap

Single-node lifecycle (config-driven GRPO loop end to end, verifiable reward
layer, within-group-variance instrumentation, untrained baseline, Docker + tests)
is in place. Next: the first real GRPO training run on a cloud spot GPU, then a
multi-process "fabric" (queue + rollout workers + learner) with checkpoint/resume
for spot-interruption fault tolerance.
