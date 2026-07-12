"""RLOO adapter (REINFORCE Leave-One-Out).

How it differs from GRPO — the baseline:
  GRPO: baseline for sample i = mean of ALL G samples (includes i), then divide
        by the group std.
  RLOO: baseline for sample i = mean of the OTHER samples only (leave i out):
            b_i = (sum(r) - r_i) / (G - 1)
            A_i = r_i - b_i
        No std normalization.

Why leave-one-out? A baseline must not depend on the action it's scoring, or it
biases the gradient. GRPO's full-group mean technically includes r_i, so each
sample is being compared against a baseline it contributed to (a small bias,
shrinking with G); GRPO also rescales by std (variance reduction, but it's a
data-dependent transform some argue biases the objective). RLOO's LOO baseline
is provably unbiased and uses no std rescaling — arguably the "cleaner"
critic-free estimator. Cost/complexity is identical to GRPO (both just need a
group of samples). We keep the SAME clipped-surrogate + KL loss so the ONLY
difference measured against GRPO is this baseline choice.
"""
from __future__ import annotations

from shoprl.rl.base import RLTrainer


def rloo_advantages(rewards: list[float]) -> list[float]:
    """Leave-one-out advantages for one prompt's group."""
    g = len(rewards)
    if g <= 1:
        return [0.0] * g  # no siblings to form a baseline
    total = sum(rewards)
    return [r - (total - r) / (g - 1) for r in rewards]


class RLOOTrainer(RLTrainer):
    name = "rloo"

    def optimize(self, completions, rewards_per_group) -> dict:
        advantages = [rloo_advantages(g) for g in rewards_per_group]
        return self._critic_free_step(completions, rewards_per_group, advantages)
