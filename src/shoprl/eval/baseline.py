"""Baseline evaluation: score the UNTRAINED policy on held-out prompts.

    python -m shoprl.eval.baseline --config configs/grpo_qwen_06b.yaml

Establishes the before-training reference (mean reward + per-component breakdown)
so a later trained checkpoint can be compared against it. Held-out = prompts
generated with a seed disjoint from training's, so we never evaluate on prompts
the model trained on. All numbers are measured from real generations.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

from shoprl.config import load_config
from shoprl.data import generate_catalog, generate_prompts
from shoprl.data.catalog import catalog_index
from shoprl.reward import RewardContext, compute_reward
from shoprl.rollout.factory import build_engine
from shoprl.task import build_shortlist, build_task_prompt

HELDOUT_SEED_OFFSET = 777  # disjoint from training prompts (seed=experiment.seed)


def _trained_engine(config, adapter: str):
    """Build a rollout engine whose policy is base + trained LoRA adapter,
    so we can eval the TRAINED model on the same held-out split."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from shoprl.rollout.hf import HFRolloutEngine, _resolve_device, _resolve_dtype

    device = _resolve_device(config.model.device)
    dtype = _resolve_dtype(config.model.dtype, device)
    tok = AutoTokenizer.from_pretrained(config.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(config.model.name, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter).to(device)
    model.eval()
    return HFRolloutEngine(config, model=model, tokenizer=tok)


def evaluate(config, n_prompts, num_samples, max_new_tokens=None, out=None,
             adapter=None) -> dict:
    if max_new_tokens:
        config.rollout.max_new_tokens = max_new_tokens

    seed = config.experiment.seed
    catalog = generate_catalog(n=config.training.catalog_size, seed=seed)
    idx = catalog_index(catalog)
    # Held-out split: distinct seed -> disjoint from training prompts.
    examples = generate_prompts(catalog, n=n_prompts, seed=seed + HELDOUT_SEED_OFFSET)

    task_prompts, contexts = [], []
    for ex in examples:
        sl = build_shortlist(ex, catalog, k=config.training.shortlist, seed=seed)
        task_prompts.append(build_task_prompt(ex, idx, sl))
        contexts.append(RewardContext(catalog=idx, constraints=ex.constraints))

    phase = "trained" if adapter else "UNTRAINED"
    print(f"[baseline] model={config.model.name} ({phase}) held-out prompts="
          f"{len(examples)} samples/prompt={num_samples}")
    engine = _trained_engine(config, adapter) if adapter else build_engine(config)
    groups = engine.generate(task_prompts, num_samples, seed=seed)

    breakdowns = []
    for ctx, group in zip(contexts, groups):
        for comp in group.completions:
            breakdowns.append(compute_reward(
                comp.text, ctx,
                weights=config.rewards.weights,
                hallucination_penalty=config.rewards.hallucination_penalty,
            ))

    n = len(breakdowns)
    totals = [b.total for b in breakdowns]
    components = {
        k: statistics.mean(getattr(b, k) for b in breakdowns)
        for k in ("budget", "groundedness", "coverage",
                  "quality_format", "quality_comparison")
    }
    result = {
        "model": config.model.name,
        "phase": "trained" if adapter else "baseline_untrained",
        "adapter": adapter,
        "n_completions": n,
        "held_out_seed": seed + HELDOUT_SEED_OFFSET,
        "reward_mean": statistics.mean(totals),
        "reward_std": statistics.pstdev(totals),
        "reward_min": min(totals),
        "reward_max": max(totals),
        "components": components,
        "hallucination_rate": sum(b.hallucinated for b in breakdowns) / n,
    }

    print("\n=== BASELINE (untrained) ===")
    print(f"  reward mean {result['reward_mean']:+.3f} ± {result['reward_std']:.3f} "
          f"(min {result['reward_min']:+.3f}, max {result['reward_max']:+.3f})")
    for k, v in components.items():
        print(f"  {k:20s} {v:.3f}")
    print(f"  hallucination_rate   {result['hallucination_rate']:.3f}")

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[baseline] saved -> {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.eval.baseline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--out", default="outputs/baseline.json")
    ap.add_argument("--adapter", default=None,
                    help="path to a trained LoRA adapter -> eval the TRAINED model")
    args = ap.parse_args()
    config = load_config(args.config)
    evaluate(config, args.n_prompts, args.num_samples, args.max_new_tokens, args.out,
             adapter=args.adapter)


if __name__ == "__main__":
    main()
