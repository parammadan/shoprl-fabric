"""Smoke CLI: config in -> sampled completions out.

    python -m shoprl.cli rollout --config configs/dev.yaml
    python -m shoprl.cli rollout --config configs/stub.yaml

This is the first end-to-end path: load a validated config, build the engine it
names, generate a group of completions per prompt, and print them. Later stages
(reward, learner, queue) hang off this same config + engine.
"""
from __future__ import annotations

import argparse

from shoprl.config import load_config
from shoprl.rollout.factory import build_engine

# A few recommendation-style prompts to exercise the engine. Real prompts come
# from the shopping dataset in a later increment.
_SAMPLE_PROMPTS = [
    "User wants wireless earbuds under $80 for running. Recommend one product.",
    "User needs a durable water bottle for hiking. Recommend one product.",
]


def _cmd_rollout(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(f"[experiment] {config.experiment.name} | engine={config.rollout.engine}")
    engine = build_engine(config)
    groups = engine.generate(_SAMPLE_PROMPTS, num_samples=config.rollout.num_samples)
    for g in groups:
        print(f"\nPROMPT: {g.prompt}")
        for i, c in enumerate(g.completions):
            print(f"  [{i}] {c.text}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="shoprl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_roll = sub.add_parser("rollout", help="Sample completions for sample prompts.")
    p_roll.add_argument("--config", required=True, help="Path to experiment YAML.")
    p_roll.set_defaults(func=_cmd_rollout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
