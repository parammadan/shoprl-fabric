"""Synthetic data for the recommendation task.

Everything here is deterministic given a seed. The catalog is the single source
of ground truth: reward functions verify the policy's recommendations against
it, so the reward is fully verifiable with no reward model.
"""
from shoprl.data.catalog import Product, generate_catalog
from shoprl.data.prompts import PromptExample, generate_prompts, satisfies

__all__ = [
    "Product",
    "generate_catalog",
    "PromptExample",
    "generate_prompts",
    "satisfies",
]
