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


def heldout_eval(trainer, n_prompts: int, num_samples: int,
                 collect_samples: bool = False) -> dict:
    """Score the trainer's CURRENT policy on a held-out split (seed disjoint
    from training). Uses the trainer's model in eval mode. With collect_samples,
    also returns per-completion records under '_samples' (for the platform to
    persist as policy-tagged trajectories)."""
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

    totals, halluc, samples = [], 0, []
    comps = {k: [] for k in ("budget", "groundedness", "coverage",
                             "quality_format", "quality_comparison")}
    for ex, ctx, group in zip(examples, contexts, groups):
        for c in group.completions:
            r = compute_reward(c.text, ctx, weights=cfg.rewards.weights,
                               hallucination_penalty=cfg.rewards.hallucination_penalty)
            totals.append(r.total)
            halluc += int(r.hallucinated)
            for k in comps:
                comps[k].append(getattr(r, k))
            if collect_samples:
                samples.append({"prompt": group.prompt, "response": c.text,
                                "reward": r.total, "components": r.as_dict(),
                                "prompt_id": ex.prompt_id})
    n = len(totals)
    out = {
        "reward_mean": statistics.mean(totals),
        "reward_std": statistics.pstdev(totals),
        "reward_min": min(totals), "reward_max": max(totals),
        "components": {k: statistics.mean(v) for k, v in comps.items()},
        "hallucination_rate": halluc / n, "n_completions": n,
    }
    if collect_samples:
        out["_samples"] = samples
    return out


def _build_result(config, before, after, step_metrics) -> dict:
    kl_traj = [m["kl"] for m in step_metrics]
    step_reward_stds = [m["reward_std"] for m in step_metrics]
    return {
        "algorithm": config.algorithm,
        "steps": config.training.steps,
        "regime": {"batch": config.training.prompts_per_step * config.rollout.num_samples,
                   "num_samples": config.rollout.num_samples,
                   "beta": config.training.beta, "lr": config.training.lr,
                   "dtype": config.model.dtype},
        "eval_n_completions": before["n_completions"],
        "held_out_before": before, "held_out_after": after,
        "reward_gain": after["reward_mean"] - before["reward_mean"],
        "final_kl": kl_traj[-1] if kl_traj else None,
        "max_kl": max(kl_traj) if kl_traj else None,
        "kl_trajectory": kl_traj,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.rl.run")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-prompts", type=int, default=64, help="held-out eval prompts (>=50)")
    ap.add_argument("--num-samples", type=int, default=2, help="eval samples per prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform-root", default=None,
                    help="platform stores dir (default runs/<name>/platform)")
    ap.add_argument("--gpu-mem-gb", type=float, default=None,
                    help="enforce the preflight memory estimate against this budget")
    ap.add_argument("--no-platform", action="store_true",
                    help="skip platform integration (pure comparison JSON only)")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    from shoprl.platform.integration import PlatformRun, cost_estimate
    from shoprl.platform.registry import RunStatus

    pr = None
    if not args.no_platform:
        root = args.platform_root or os.path.join("runs", config.experiment.name, "platform")
        pr = PlatformRun(config, root)
        if not args.skip_preflight:
            report = pr.preflight(gpu_mem_gb=args.gpu_mem_gb)
            for c in report.checks:
                print(f"[preflight] {'PASS' if c.ok else 'FAIL'} {c.name}: {c.detail}")
            report.raise_if_failed()          # abort before allocating anything
        pr.start(n_prompts=args.n_prompts)
        print(f"[run] registered run {pr.run.run_id} (config {pr.run.config_hash}, "
              f"dataset {pr.run.dataset_version})")

    try:
        trainer = build_trainer(config)
        print(f"[run] algorithm={config.algorithm} steps={config.training.steps} "
              f"device={trainer.device}")
        before = heldout_eval(trainer, args.n_prompts, args.num_samples)
        print(f"[run] before: {before['reward_mean']:+.3f} ± {before['reward_std']:.3f}")

        step_metrics = trainer.train()
        ckpt_dir = trainer.save_checkpoint(config.training.steps)

        after = heldout_eval(trainer, args.n_prompts, args.num_samples,
                             collect_samples=pr is not None)
        print(f"[run] after:  {after['reward_mean']:+.3f} ± {after['reward_std']:.3f}")

        result = _build_result(config, before,
                               {k: v for k, v in after.items() if k != "_samples"},
                               step_metrics)
        result.update({
            "train_time_s": getattr(trainer, "train_time_s", None),
            "tokens_per_sec": getattr(trainer, "tokens_per_sec", None),
            "peak_mem_gb": getattr(trainer, "peak_mem_gb", None),
            "mean_rollout_reward_std": statistics.mean(
                [m["reward_std"] for m in step_metrics]) if step_metrics else 0.0,
            "stability_failures": trainer.stability_failures,
            "step_metrics": step_metrics,
        })

        if pr is not None:
            manifest = pr.register_checkpoint(ckpt_dir)      # atomic + verified
            pv = pr.publish_policy(ckpt_dir, metadata={"step": config.training.steps,
                                                       "algorithm": config.algorithm})
            for s in after.get("_samples", []):              # tag trajectories -> vN
                pr.tag_trajectory(s["prompt"], s["response"], s["reward"],
                                  components=s["components"], prompt_id=s["prompt_id"])
            pr.finish(RunStatus.SUCCEEDED, eval_result=result,
                      best_checkpoint=manifest.ckpt_id,
                      cost_estimate=cost_estimate(result["train_time_s"]))
            print(f"[run] checkpoint {manifest.ckpt_id} · policy v{pv.version} · "
                  f"{len(after.get('_samples', []))} trajectories tagged v{pv.version}")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        t = result["train_time_s"]
        print(f"[run] gain={result['reward_gain']:+.3f} final_kl={result['final_kl']} "
              f"time={t:.0f}s -> {args.out}" if t else f"[run] -> {args.out}")
    except Exception as e:
        if pr is not None:
            pr.fail(repr(e))
        raise
    finally:
        if pr is not None:
            pr.close()


if __name__ == "__main__":
    main()
