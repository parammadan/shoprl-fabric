"""Unified config-driven entry point for the whole training lifecycle.

    python -m shoprl.train --config configs/grpo_qwen_06b.yaml

One YAML specifies experiment / model / algorithm / rollout / training / rewards.
The `algorithm` field selects the RL algorithm; today only GRPO is implemented
(RLOO/PPO are on the roadmap and dispatch here when added).
"""
from __future__ import annotations

import argparse

from shoprl.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.train")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    print(f"[train] experiment={config.experiment.name} algorithm={config.algorithm}")

    if config.algorithm == "grpo":
        from shoprl.grpo.trainer import run_training

        run_training(config)
    else:
        raise NotImplementedError(
            f"algorithm '{config.algorithm}' not implemented yet (GRPO only)."
        )


if __name__ == "__main__":
    main()
