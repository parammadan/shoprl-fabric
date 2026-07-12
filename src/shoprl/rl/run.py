"""Run ONE algorithm end-to-end and write a metrics JSON for the comparison.

    python -m shoprl.rl.run --config configs/compare_grpo.yaml --out results/grpo.json

Does before-eval (untrained) -> train -> after-eval (trained) on the SAME
held-out split, reusing the trainer's own model (no second model in memory), and
records every comparison metric: held-out reward before/after + gain, final KL,
full KL trajectory, training time, peak GPU memory, tokens/sec, mean rollout
reward std (signal), and stability failures. All measured — nothing fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from shoprl.config import load_config
from shoprl.data import generate_prompts
from shoprl.reward import RewardContext, compute_reward
from shoprl.rl.factory import build_trainer
from shoprl.task import build_shortlist, build_task_prompt

HELDOUT_SEED_OFFSET = 777


def heldout_eval(trainer, n_prompts: int, num_samples: int) -> dict:
    """Score the trainer's CURRENT policy on a held-out split (seed disjoint
    from training). Uses the trainer's model in eval mode."""
    cfg = trainer.config
    seed = cfg.experiment.seed
    examples = generate_prompts(trainer.catalog, n=n_prompts, seed=seed + HELDOUT_SEED_OFFSET)
    task_prompts, contexts = [], []
    for ex in examples:
        sl = build_shortlist(ex, trainer.catalog, k=cfg.training.shortlist, seed=seed)
        task_prompts.append(build_task_prompt(ex, trainer.idx, sl))
        contexts.append(RewardContext(catalog=trainer.idx, constraints=ex.constraints))

    trainer.model.gradient_checkpointing_disable()
    trainer.model.eval()
    groups = trainer.engine.generate(task_prompts, num_samples, seed=seed)

    totals, halluc = [], 0
    comps = {k: [] for k in ("budget", "groundedness", "coverage",
                             "quality_format", "quality_comparison")}
    for ctx, group in zip(contexts, groups):
        for c in group.completions:
            r = compute_reward(c.text, ctx, weights=cfg.rewards.weights,
                               hallucination_penalty=cfg.rewards.hallucination_penalty)
            totals.append(r.total)
            halluc += int(r.hallucinated)
            for k in comps:
                comps[k].append(getattr(r, k))
    n = len(totals)
    return {
        "reward_mean": statistics.mean(totals),
        "reward_std": statistics.pstdev(totals),
        "reward_min": min(totals), "reward_max": max(totals),
        "components": {k: statistics.mean(v) for k, v in comps.items()},
        "hallucination_rate": halluc / n, "n_completions": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.rl.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-prompts", type=int, default=64, help="held-out eval prompts (>=50)")
    ap.add_argument("--num-samples", type=int, default=2, help="eval samples per prompt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    trainer = build_trainer(config)
    print(f"[run] algorithm={config.algorithm} steps={config.training.steps} "
          f"device={trainer.device}")

    before = heldout_eval(trainer, args.n_prompts, args.num_samples)
    print(f"[run] before: {before['reward_mean']:+.3f} ± {before['reward_std']:.3f}")

    step_metrics = trainer.train()
    trainer.save_checkpoint(config.training.steps)

    after = heldout_eval(trainer, args.n_prompts, args.num_samples)
    print(f"[run] after:  {after['reward_mean']:+.3f} ± {after['reward_std']:.3f}")

    kl_traj = [m["kl"] for m in step_metrics]
    step_reward_stds = [m["reward_std"] for m in step_metrics]
    result = {
        "algorithm": config.algorithm,
        "steps": config.training.steps,
        "regime": {"batch": config.training.prompts_per_step * config.rollout.num_samples,
                   "num_samples": config.rollout.num_samples,
                   "beta": config.training.beta, "lr": config.training.lr,
                   "dtype": config.model.dtype},
        "eval_n_completions": before["n_completions"],
        "held_out_before": before,
        "held_out_after": after,
        "reward_gain": after["reward_mean"] - before["reward_mean"],
        "final_kl": kl_traj[-1] if kl_traj else None,
        "max_kl": max(kl_traj) if kl_traj else None,
        "kl_trajectory": kl_traj,
        "train_time_s": getattr(trainer, "train_time_s", None),
        "tokens_per_sec": getattr(trainer, "tokens_per_sec", None),
        "peak_mem_gb": getattr(trainer, "peak_mem_gb", None),
        "mean_rollout_reward_std": statistics.mean(step_reward_stds) if step_reward_stds else 0.0,
        "stability_failures": trainer.stability_failures,
        "step_metrics": step_metrics,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[run] gain={result['reward_gain']:+.3f} final_kl={result['final_kl']} "
          f"time={result['train_time_s']:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
