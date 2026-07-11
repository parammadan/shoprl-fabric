"""Rule-based, verifiable reward layer for the recommendation task.

Every score is computed against the catalog (ground truth) — no reward model,
no fabricated numbers. See composite.compute_reward for the scalar the learner
optimizes.
"""
from shoprl.reward.composite import (
    HALLUCINATION_PENALTY,
    WEIGHTS,
    RewardBreakdown,
    compute_reward,
)
from shoprl.reward.functions import (
    RewardContext,
    attribute_coverage,
    budget_compliance,
    catalog_groundedness,
    is_hallucinated,
    response_quality,
)

__all__ = [
    "RewardContext",
    "RewardBreakdown",
    "compute_reward",
    "WEIGHTS",
    "HALLUCINATION_PENALTY",
    "budget_compliance",
    "catalog_groundedness",
    "is_hallucinated",
    "attribute_coverage",
    "response_quality",
]
