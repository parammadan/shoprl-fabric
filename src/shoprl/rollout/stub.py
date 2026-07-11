"""Deterministic rollout engine with zero ML dependencies.

Purpose: exercise all the plumbing (config -> rollout -> reward -> queue ->
learner) and write fast, hermetic tests without downloading a model. Given the
same seed and prompts it always returns the same completions, so tests can
assert on exact output.
"""
from __future__ import annotations

import random

from shoprl.rollout.base import Completion, RolloutEngine, RolloutGroup

# A tiny canned vocabulary of recommendation-shaped phrases. This is NOT a
# model — it just produces plausible, varied, deterministic text so downstream
# stages have something realistic to chew on.
_CANNED = [
    "I recommend the {p} for its great value.",
    "Consider the {p}; it fits your budget and needs.",
    "The {p} is a top pick based on reviews.",
    "Go with the {p} — reliable and well rated.",
    "For you, the {p} is the best match.",
]
_PRODUCTS = ["AcmeWidget X1", "NovaBuds Pro", "TerraBottle 1L", "PixelCase", "GlowLamp Mini"]


class StubRolloutEngine(RolloutEngine):
    def __init__(self, seed: int = 0):
        self.seed = seed

    def generate(self, prompts: list[str], num_samples: int) -> list[RolloutGroup]:
        groups: list[RolloutGroup] = []
        for i, prompt in enumerate(prompts):
            # Seed per (engine seed, prompt index) so results are deterministic
            # yet differ across prompts.
            rng = random.Random((self.seed, i))
            comps: list[Completion] = []
            for _ in range(num_samples):
                template = rng.choice(_CANNED)
                product = rng.choice(_PRODUCTS)
                text = template.format(p=product)
                comps.append(Completion(prompt=prompt, text=text))
            groups.append(RolloutGroup(prompt=prompt, completions=comps))
        return groups
