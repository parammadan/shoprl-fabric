"""Inspect the reward distribution on live policy rollouts.

    python -m shoprl.eval.reward_report --config configs/dev.yaml \
        --n-prompts 4 --num-samples 8

Runs the configured rollout engine on grounded task prompts, scores every
completion with the composite reward, and prints a report so we can see —
before training — whether there is any signal to optimize: reward spread, how
often the model emits parseable SKUs, and whether hallucination penalties fire.

Every number here is measured from real generations. Nothing is fabricated.
"""
from __future__ import annotations

import argparse
import statistics

from shoprl.config import load_config
from shoprl.data import generate_catalog, generate_prompts
from shoprl.data.catalog import catalog_index
from shoprl.reward import RewardContext, compute_reward
from shoprl.reward.parse import parse_response
from shoprl.rollout.factory import build_engine
from shoprl.task import build_shortlist, build_task_prompt


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _bar(frac: float, width: int = 20) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.eval.reward_report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=None, help="overrides config")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="overrides config")
    ap.add_argument("--catalog-size", type=int, default=300)
    ap.add_argument("--shortlist", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=3, help="example completions to print")
    ap.add_argument("--dump", default=None, help="write all completions+rewards to JSONL for auditing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    num_samples = args.num_samples or cfg.rollout.num_samples
    if args.max_new_tokens:
        cfg.rollout.max_new_tokens = args.max_new_tokens

    # Same seed as generation -> catalog used for scoring matches the prompts.
    catalog = generate_catalog(n=args.catalog_size, seed=args.seed)
    idx = catalog_index(catalog)
    examples = generate_prompts(catalog, n=args.n_prompts, seed=args.seed)

    task_prompts, contexts = [], []
    for ex in examples:
        shortlist = build_shortlist(ex, catalog, k=args.shortlist, seed=args.seed)
        task_prompts.append(build_task_prompt(ex, idx, shortlist))
        contexts.append(RewardContext(catalog=idx, constraints=ex.constraints))

    print(f"[reward_report] model={cfg.model.name} engine={cfg.rollout.engine} "
          f"prompts={len(examples)} samples/prompt={num_samples}")
    print("generating (real rollouts; may take a few minutes on M1)...")
    engine = build_engine(cfg)
    groups = engine.generate(task_prompts, num_samples=num_samples)

    # Score every completion.
    records = []  # (example, ctx, completion_text, breakdown, n_claimed)
    for ex, ctx, group in zip(examples, contexts, groups):
        for comp in group.completions:
            bd = compute_reward(comp.text, ctx)
            n_claimed = len(parse_response(comp.text))
            records.append((ex, ctx, comp.text, bd, n_claimed))

    totals = [r[3].total for r in records]
    n = len(records)

    # --- aggregate report -------------------------------------------------
    print("\n" + "=" * 64)
    print(f"REWARD DISTRIBUTION  (n={n} completions)")
    print("=" * 64)
    print(f"  mean   {statistics.mean(totals):+.3f}")
    print(f"  stdev  {statistics.pstdev(totals):.3f}")
    print(f"  min    {min(totals):+.3f}")
    print(f"  median {statistics.median(totals):+.3f}")
    print(f"  max    {max(totals):+.3f}")

    print("\nComponent means:")
    for key in ("budget", "groundedness", "coverage", "quality_format", "quality_comparison"):
        vals = [getattr(r[3], key) for r in records]
        m = statistics.mean(vals)
        print(f"  {key:20s} {m:.3f}  {_bar(m)}")

    # Within-group (per-prompt) reward std — the quantity GRPO actually uses.
    # Advantages are computed relative to each prompt's own group mean, so a
    # prompt whose samples all score alike gives ~zero gradient regardless of
    # how high they score. Healthy per-group std is the real trainability check.
    group_totals: dict[str, list[float]] = {}
    for ex, _c, _t, bd, _nc in records:
        group_totals.setdefault(ex.prompt_id, []).append(bd.total)
    within_stds = [statistics.pstdev(v) for v in group_totals.values() if len(v) > 1]
    if within_stds:
        print("\nWithin-group reward std (GRPO signal):")
        print(f"  mean {statistics.mean(within_stds):.3f}  "
              f"min {min(within_stds):.3f}  max {max(within_stds):.3f}")
        for pid, v in group_totals.items():
            flag = "  <- ~flat, weak gradient" if statistics.pstdev(v) < 0.02 else ""
            print(f"    {pid}: mean {statistics.mean(v):+.3f} std {statistics.pstdev(v):.3f}{flag}")

    parseable = sum(1 for r in records if r[4] > 0) / n
    grounded_pos = sum(1 for r in records if r[3].groundedness > 0) / n
    halluc = sum(1 for r in records if r[3].hallucinated) / n
    print("\nRates:")
    print(f"  emits >=1 parseable SKU   {_pct(parseable)}")
    print(f"  groundedness > 0          {_pct(grounded_pos)}")
    print(f"  hallucination penalty fired {_pct(halluc)}")

    # Coarse histogram of totals.
    print("\nTotal-reward histogram:")
    edges = [(-0.6, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in edges:
        c = sum(1 for t in totals if lo <= t < hi)
        print(f"  [{lo:+.2f},{hi:+.2f})  {c:3d}  {_bar(c / n)}")

    # --- example completions ---------------------------------------------
    ordered = sorted(records, key=lambda r: r[3].total)
    picks = {"LOWEST": ordered[0], "HIGHEST": ordered[-1]}
    halluc_recs = [r for r in records if r[3].hallucinated]
    if halluc_recs:
        picks["HALLUCINATED"] = halluc_recs[0]

    print("\n" + "=" * 64)
    print("EXAMPLE COMPLETIONS")
    print("=" * 64)
    for label, (ex, _ctx, text, bd, nc) in picks.items():
        print(f"\n--- {label}  total={bd.total:+.3f}  hallucinated={bd.hallucinated} "
              f"claimed_skus={nc} ---")
        print(f"constraints: {ex.constraints}")
        snippet = text.strip().replace("\n", "\n    ")
        print("    " + snippet[:400])

    if args.dump:
        import json

        with open(args.dump, "w") as f:
            for ex, _c, text, bd, nc in records:
                f.write(json.dumps({
                    "prompt_id": ex.prompt_id,
                    "constraints": ex.constraints,
                    "n_claimed": nc,
                    "reward": bd.as_dict(),
                    "completion": text,
                }) + "\n")
        print(f"\n[dumped {len(records)} completions -> {args.dump}]")


if __name__ == "__main__":
    main()
