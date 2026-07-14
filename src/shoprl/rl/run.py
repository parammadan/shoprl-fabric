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


def run_experiment(config, n_prompts: int, num_samples: int, resume_from: str | None = None) -> dict:
    """Pure training core (no platform coupling): build trainer -> before-eval
    -> train -> serialise the adapter to a TEMP dir -> after-eval with samples.
    Returns {checkpoint_dir (temp, caller owns/ingests), result, samples}. Called
    by the CLI and the control-plane worker so there is ONE real training path."""
    import json as _json
    import tempfile

    from shoprl.platform.gpu import gpu_telemetry

    trainer = build_trainer(config, resume_from=resume_from)
    steps = config.training.steps
    before = heldout_eval(trainer, n_prompts, num_samples)
    step_metrics = trainer.train()

    tmp = tempfile.mkdtemp(prefix="shoprl-ckpt-")
    trainer.model.save_pretrained(tmp)
    with open(f"{tmp}/train_state.json", "w") as f:
        _json.dump({"step": steps, "model": config.model.name}, f)

    after = heldout_eval(trainer, n_prompts, num_samples, collect_samples=True)
    result = _build_result(config, before,
                           {k: v for k, v in after.items() if k != "_samples"},
                           step_metrics)
    result.update({
        "train_time_s": getattr(trainer, "train_time_s", None),
        "tokens_per_sec": getattr(trainer, "tokens_per_sec", None),
        "peak_mem_gb": getattr(trainer, "peak_mem_gb", None),
        "gpu": gpu_telemetry(),
        "mean_rollout_reward_std": statistics.mean(
            [m["reward_std"] for m in step_metrics]) if step_metrics else 0.0,
        "stability_failures": trainer.stability_failures,
        "step_metrics": step_metrics,
    })
    return {"checkpoint_dir": tmp, "result": result, "samples": after.get("_samples", [])}


def run_through_platform(config, n_prompts: int, num_samples: int, root, *,
                         gpu_mem_gb=None, skip_preflight=False, runner=run_experiment,
                         resume_from=None) -> dict:
    """The ONE platform-wired training path (used by the CLI and the control
    worker): preflight -> register run + dataset -> [train via runner] ->
    register checkpoint (registry = sole writer) -> publish policy -> tag
    trajectories -> finish. Returns run refs + result."""
    import shutil

    from shoprl.platform.integration import PlatformRun, cost_estimate
    from shoprl.platform.registry import RunStatus

    pr = PlatformRun(config, root)
    out = None
    try:
        if not skip_preflight:
            pr.preflight(gpu_mem_gb=gpu_mem_gb).raise_if_failed()
        pr.start(n_prompts=n_prompts)
        try:
            out = runner(config, n_prompts, num_samples, resume_from)   # w/ resume support
        except TypeError:
            out = runner(config, n_prompts, num_samples)                # runner w/o resume arg

        manifest = pr.register_checkpoint(out["checkpoint_dir"], step=config.training.steps)
        pv = pr.publish_policy(out["checkpoint_dir"],
                               metadata={"step": config.training.steps,
                                         "algorithm": config.algorithm})
        for s in out["samples"]:                          # tag eval trajectories -> v{n}
            pr.tag_trajectory(s["prompt"], s["response"], s["reward"],
                              components=s["components"], prompt_id=s["prompt_id"])
        result = out["result"]
        pr.finish(RunStatus.SUCCEEDED, eval_result=result,
                  best_checkpoint=manifest.ckpt_id,
                  cost_estimate=cost_estimate(result.get("train_time_s")))
        return {"run_id": pr.run.run_id, "best_checkpoint": manifest.ckpt_id,
                "policy_version": pv.version, "result": result}
    except Exception as e:
        pr.fail(repr(e))
        raise
    finally:
        pr.close()
        if out and out.get("checkpoint_dir"):
            shutil.rmtree(out["checkpoint_dir"], ignore_errors=True)


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
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)

    # The ONLY execution path is through the platform. (For orchestration via
    # the job queue + scheduler, submit through the control plane / API instead;
    # this CLI is the direct single-run entry and uses the same platform path.)
    root = args.platform_root or os.path.join("runs", config.experiment.name, "platform")
    print(f"[run] algorithm={config.algorithm} steps={config.training.steps} "
          f"-> platform root {root}")
    ref = run_through_platform(config, args.n_prompts, args.num_samples, root,
                               gpu_mem_gb=args.gpu_mem_gb, skip_preflight=args.skip_preflight)
    result = ref["result"]
    print(f"[run] run {ref['run_id']} · checkpoint {ref['best_checkpoint']} · "
          f"policy v{ref['policy_version']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    t = result.get("train_time_s")
    print(f"[run] gain={result['reward_gain']:+.3f} final_kl={result['final_kl']} "
          + (f"time={t:.0f}s " if t else "") + f"-> {args.out}")


if __name__ == "__main__":
    main()
