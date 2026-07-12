"""Build the PPO/GRPO/RLOO comparison table from measured result JSONs.

    python -m shoprl.rl.report results/grpo.json results/rloo.json results/ppo.json

Prints a markdown table straight from the numbers each run recorded — no
fabrication. Refuses to invent rows for runs that weren't provided.
"""
from __future__ import annotations

import argparse
import json


def _fmt(v, spec="{:.3f}"):
    return "n/a" if v is None else spec.format(v)


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.rl.report")
    ap.add_argument("results", nargs="+", help="result JSON files from shoprl.rl.run")
    args = ap.parse_args()

    rows = [json.load(open(p)) for p in args.results]

    cols = ["algo", "steps", "before", "after", "gain", "final_KL", "max_KL",
            "time_s", "tok/s", "peak_GB", "rollout_std", "fails"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        b = r["held_out_before"]["reward_mean"]
        a = r["held_out_after"]["reward_mean"]
        print("| " + " | ".join([
            r["algorithm"], str(r["steps"]),
            _fmt(b), _fmt(a), _fmt(r["reward_gain"], "{:+.3f}"),
            _fmt(r["final_kl"]), _fmt(r["max_kl"]),
            _fmt(r["train_time_s"], "{:.0f}"), _fmt(r["tokens_per_sec"], "{:.1f}"),
            _fmt(r["peak_mem_gb"], "{:.1f}"),
            _fmt(r["mean_rollout_reward_std"]), str(r["stability_failures"]),
        ]) + " |")

    print("\nNotes:")
    print("- before/after = held-out reward mean (n_completions "
          f"{rows[0].get('eval_n_completions','?')}); gain = after - before.")
    print("- final_KL / max_KL from the KL trajectory; watch KL vs gain.")
    print("- rollout_std = mean within-batch reward std during training (signal).")
    print("- fails = NaN/inf loss steps skipped (stability).")


if __name__ == "__main__":
    main()
