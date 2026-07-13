# ShopRL Fabric

A from-scratch **RL post-training platform** for a shopping-recommendation LLM —
the full lifecycle (config → rollout → reward → optimize → eval → checkpoint) with
observability, alerting, and efficiency benchmarking. Built to develop on a laptop
(Apple M1, tiny model) and scale to cloud GPU **unchanged** behind stable
interfaces.

Model: `Qwen/Qwen3-0.6B` (small on purpose). Stack: PyTorch, HF Transformers,
PEFT (LoRA), vLLM (GPU rollout), spot GPU + HF Hub checkpoints.

## Why it's interesting
- **Verifiable, un-hackable reward** — scored against a synthetic catalog that *is*
  ground truth (no reward model); budget/coverage check the catalog's real specs,
  never the model's stated ones.
- **Three RL algorithms behind one interface** — GRPO, RLOO, PPO share rollout /
  reward / KL / clipped-loss, so comparisons isolate exactly the advantage/critic.
- **Runs on 8 GB M1** — LoRA (adapter = policy; disabled = frozen reference, no 2nd
  model), logsumexp log-probs (no full-vocab softmax), gradient checkpointing.
- **Real ops discipline** — spot GPUs via SSM, delete-on-terminate EBS, verified
  teardown every run, checkpoints to HF Hub, incident-response alerting.

## Key results (all measured — see the docs repo for full data)
- **The loop learns**: nonzero gradients, KL controlled, coherent generation
  (after fixing a rollout-in-train-mode bug that had silently zeroed learning).
- **Rigor over hype**: a 30-step GRPO run showed 0.851→0.930 held-out at n=16, but
  a proper **n=64** eval revised that to ~flat (0.841→0.839) — the task is
  near-saturated for the base model. Reported honestly; the substantive result is
  the comparison below.
- **Algorithm KL-stability (30 steps, identical settings, n=64 held-out):** reward
  gain ≈0 for all three, but stability differs sharply —

  | algo | held-out gain | final KL | max KL |
  |---|---|---|---|
  | RLOO | +0.002 | **0.015** | **0.22** |
  | GRPO | −0.002 | 0.58 | 0.99 |
  | PPO  | +0.001 | **6.78** | **6.78** |

  **RLOO is the most stable** (leave-one-out baseline, no critic); **PPO's fresh
  value-head critic destabilized** (KL blew up) with no reward benefit — the classic
  case for critic-free methods when a sample group is available.
- **Alerting validated on the real runs**: the KL-blow-up rule fires CRITICAL on
  PPO (×13) and stays silent on RLOO; reward-stall fires on the flat runs.
- **Efficiency (A10G, measured)**: **rollout = 82–84% of wall-clock**, optimize
  ~17%, reward ~0%. Async rollout and reward-worker parallelism gave no gain on a
  single GPU (measured negative result); the real lever is faster rollout (vLLM) —
  which is why rollout is decoupled in the scale design.

## Architecture
`config → RolloutEngine (HF ⇄ vLLM) → reward (vs catalog) → RLTrainer
(GRPO/RLOO/PPO) → eval → checkpoint`, with `metrics.jsonl → dashboard + alerting`.
Full diagram + the distributed-fabric scale target: docs repo `ARCHITECTURE.md`.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core + tests (model-free)
pip install -e ".[hf]"       # + torch/transformers/peft for real runs
```

## Run
```bash
python -m shoprl.data.build --out data                          # synth dataset
python -m shoprl.eval.baseline --config configs/grpo_qwen_06b.yaml   # untrained baseline
python -m shoprl.train --config configs/grpo_qwen_06b.yaml       # GRPO training (one YAML)
python -m shoprl.rl.run --config configs/compare_rloo.yaml --out results/rloo.json  # any algorithm
python -m shoprl.observability.alerts --result results/rloo.json # incident alerts
python -m shoprl.observability.dashboard --overlay results/*.json --out overlay.html # comparison view
python -m shoprl.bench.harness --config configs/bench_hf.yaml --steps 16 --out b.json # profiling

pytest -q                                        # full suite (model-free, fast)
docker build -t shoprl-fabric . && docker run --rm shoprl-fabric
```

## Status
Months 1–4 complete: single-node lifecycle, algorithm comparison, observability +
alerting, efficiency benchmarking. Next: land a working vLLM build (rollout
throughput) and the distributed fabric (rollout/reward workers + learner over a
queue, with checkpoint/resume). Dev log + design docs live in a separate repo.
