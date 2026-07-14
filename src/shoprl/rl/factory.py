"""Build the RLTrainer named by config.algorithm."""
from __future__ import annotations

from shoprl.config import Config
from shoprl.rl.base import RLTrainer


def build_trainer(config: Config, resume_from: str | None = None) -> RLTrainer:
    algo = config.algorithm
    if algo == "grpo":
        from shoprl.rl.grpo import GRPOTrainer
        return GRPOTrainer(config, resume_from)
    if algo == "rloo":
        from shoprl.rl.rloo import RLOOTrainer
        return RLOOTrainer(config, resume_from)
    if algo == "ppo":
        from shoprl.rl.ppo import PPOTrainer
        return PPOTrainer(config, resume_from)
    raise ValueError(f"unknown algorithm: {algo!r}")
