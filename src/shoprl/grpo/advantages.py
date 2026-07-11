"""Group-relative advantages — GRPO's critic-free credit assignment.

For each prompt we sampled a GROUP of completions and scored them. The advantage
of completion i is how much better it did than its own group, standardized:

    A_i = (r_i - mean(group)) / (std(group) + eps)

Why this shape (see also the module docstring):
  - mean subtraction = the baseline. GRPO replaces PPO's learned value network
    with "the average sibling in this group". Unbiased (baseline doesn't depend
    on the sampled action) and it slashes gradient variance.
  - std division = scale normalization, so a prompt with a naturally wide reward
    spread doesn't dominate the update over a narrow-spread prompt. Every group
    contributes on the same scale (a per-prompt z-score).
  - eps = guard for a FLAT group (std ~ 0). Then the numerator is ~0 too, so
    A_i -> 0 (that group just adds no gradient) instead of 0/0 = NaN poisoning
    the batch. This is precisely the "within-group variance" we measure before
    training: low variance isn't a crash, it's wasted rollout compute.

Note: this is the DeepSeekMath-style GRPO estimator (full-group mean/std,
including sample i). RLOO is the leave-one-out variant; PPO uses a learned
critic instead. We keep this pure and framework-free so it's trivially testable
on M1 with no model in the loop.
"""
from __future__ import annotations

import statistics


def group_advantages(
    rewards: list[float],
    eps: float = 1e-4,
    standardize: bool = True,
) -> list[float]:
    """Advantages for ONE prompt's group of completion rewards.

    Returns a list aligned with `rewards`. With standardize=False, returns the
    raw mean-centered advantages (r_i - mean) — useful for ablations.
    """
    n = len(rewards)
    if n == 0:
        return []

    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if not standardize:
        return centered

    # Population std (divide by n), consistent with using the full-group mean.
    std = statistics.pstdev(rewards)
    return [c / (std + eps) for c in centered]


def batch_group_advantages(
    groups: list[list[float]],
    eps: float = 1e-4,
    standardize: bool = True,
) -> list[list[float]]:
    """Advantages for a batch of groups (one inner list per prompt)."""
    return [group_advantages(g, eps=eps, standardize=standardize) for g in groups]
