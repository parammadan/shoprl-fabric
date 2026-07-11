"""Composite reward: the scalar the RL learner actually optimizes.

    total = 0.25*budget + 0.25*groundedness + 0.25*coverage
          + 0.15*format + 0.10*comparison
          - 0.50 * hallucinated

The positive weights sum to 1.0, so an honest, well-formed, fully-compliant
answer approaches +1.0. Any hallucination subtracts a flat 0.50 AND drags the
groundedness term down, so it can push total negative — a deliberately strong
signal that inventing products/specs is worse than a merely mediocre answer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from shoprl.reward.functions import (
    RewardContext,
    attribute_coverage,
    budget_compliance,
    catalog_groundedness,
    is_hallucinated,
    response_quality,
)

# Component weights (from spec). Kept here so a config can override them later.
WEIGHTS = {
    "budget": 0.25,
    "groundedness": 0.25,
    "coverage": 0.25,
    "quality_format": 0.15,
    "quality_comparison": 0.10,
}
HALLUCINATION_PENALTY = 0.50


@dataclass
class RewardBreakdown:
    budget: float
    groundedness: float
    coverage: float
    quality_format: float
    quality_comparison: float
    hallucinated: bool
    total: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_reward(response: str, ctx: RewardContext) -> RewardBreakdown:
    budget = budget_compliance(response, ctx)
    groundedness = catalog_groundedness(response, ctx)
    coverage = attribute_coverage(response, ctx)
    fmt, comparison = response_quality(response, ctx)
    hallucinated = is_hallucinated(response, ctx)

    total = (
        WEIGHTS["budget"] * budget
        + WEIGHTS["groundedness"] * groundedness
        + WEIGHTS["coverage"] * coverage
        + WEIGHTS["quality_format"] * fmt
        + WEIGHTS["quality_comparison"] * comparison
        - HALLUCINATION_PENALTY * (1.0 if hallucinated else 0.0)
    )

    return RewardBreakdown(
        budget=budget,
        groundedness=groundedness,
        coverage=coverage,
        quality_format=fmt,
        quality_comparison=comparison,
        hallucinated=hallucinated,
        total=total,
    )
