"""GRPO adapter.

Advantage = (r_i - mean_group) / (std_group + eps). The baseline is the group
mean (including sample i) and the advantage is standardized by the group's own
std. Critic-free: no value network to train. See shoprl.grpo.advantages.
"""
from __future__ import annotations

from shoprl.grpo.advantages import batch_group_advantages
from shoprl.rl.base import RLTrainer


class GRPOTrainer(RLTrainer):
    name = "grpo"

    def optimize(self, completions, rewards_per_group) -> dict:
        advantages = batch_group_advantages(rewards_per_group)  # group mean/std
        return self._critic_free_step(completions, rewards_per_group, advantages)
