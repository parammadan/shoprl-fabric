"""Prompt generator with verifiable ground truth.

Each example carries the constraint set AND the exact set of catalog SKUs that
satisfy it (`answer_skus`). Because we computed the answer from the catalog, the
reward layer can check any policy response against objective truth — no human
labels, no reward model.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel

from shoprl.data.catalog import Product

# The four constraint dimensions the prompts can impose. Each maps to a spec.
CONSTRAINT_KEYS = ("max_price", "min_ram", "max_weight", "min_battery")


class PromptExample(BaseModel):
    prompt_id: str
    prompt: str
    constraints: dict[str, float]  # active constraints only
    answer_skus: list[str]  # ground truth: catalog SKUs satisfying ALL of them


def satisfies(product: Product, constraints: dict[str, float]) -> bool:
    """True iff the product meets every active constraint. This IS the ground
    truth predicate — reward functions reuse it, so 'correct' means exactly this.
    """
    if "max_price" in constraints and product.price > constraints["max_price"]:
        return False
    if "min_ram" in constraints and product.ram_gb < constraints["min_ram"]:
        return False
    if "max_weight" in constraints and product.weight_lbs > constraints["max_weight"]:
        return False
    if "min_battery" in constraints and product.battery_hrs < constraints["min_battery"]:
        return False
    return True


def _sample_constraints(rng: random.Random) -> dict[str, float]:
    # Pick 1-4 dimensions, then a plausible threshold for each.
    k = rng.randint(1, 4)
    keys = rng.sample(CONSTRAINT_KEYS, k)
    out: dict[str, float] = {}
    for key in keys:
        if key == "max_price":
            out[key] = float(rng.choice([800, 1000, 1200, 1500, 2000]))
        elif key == "min_ram":
            out[key] = float(rng.choice([8, 16, 32]))
        elif key == "max_weight":
            out[key] = float(rng.choice([3.0, 3.5, 4.0, 4.5]))
        elif key == "min_battery":
            out[key] = float(rng.choice([8, 10, 12, 15]))
    return out


# Human-readable fragments for each constraint, templated into the prompt.
def _phrase(constraints: dict[str, float]) -> str:
    parts: list[str] = []
    if "max_price" in constraints:
        parts.append(f"under ${int(constraints['max_price'])}")
    if "min_ram" in constraints:
        parts.append(f"at least {int(constraints['min_ram'])}GB of RAM")
    if "max_weight" in constraints:
        parts.append(f"weighing no more than {constraints['max_weight']} lbs")
    if "min_battery" in constraints:
        parts.append(f"with {int(constraints['min_battery'])}+ hours of battery")
    # Join naturally: "A, B, and C".
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


_TEMPLATES = [
    "I'm shopping for a laptop {spec}. Recommend one and explain why.",
    "Looking for a laptop {spec}. Which would you suggest?",
    "Help me pick a laptop {spec}. Give your top recommendation.",
    "Need a new laptop {spec}. What should I buy?",
]


def generate_prompts(
    catalog: list[Product],
    n: int = 200,
    seed: int = 0,
    require_nonempty: bool = True,
    max_tries: int = 50,
) -> list[PromptExample]:
    """Generate `n` prompts, each with its ground-truth answer set.

    With require_nonempty, resample constraints until at least one catalog
    product satisfies them — otherwise reward signal (coverage/budget) would be
    undefined for that prompt.
    """
    rng = random.Random(f"prompts-{seed}")
    examples: list[PromptExample] = []
    for i in range(1, n + 1):
        for _ in range(max_tries):
            constraints = _sample_constraints(rng)
            answers = [p.sku for p in catalog if satisfies(p, constraints)]
            if answers or not require_nonempty:
                break
        template = rng.choice(_TEMPLATES)
        prompt = template.format(spec=_phrase(constraints))
        examples.append(
            PromptExample(
                prompt_id=f"P-{i:04d}",
                prompt=prompt,
                constraints=constraints,
                answer_skus=answers,
            )
        )
    return examples


def write_jsonl(examples: list[PromptExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump()) + "\n")
    return path
