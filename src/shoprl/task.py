"""Turn a PromptExample into the actual text the policy sees.

Grounded recommendation = retrieve candidates, then have the LLM pick/justify
from them. A 0.6B model can't know a 300-item catalog, so we put a candidate
shortlist right in the prompt (with SKUs + specs) and ask for the REC format.
This is what makes groundedness reachable and the reward learnable — and it's
the same prompt the GRPO rollouts will use.
"""
from __future__ import annotations

import random

from shoprl.data.catalog import Product
from shoprl.data.prompts import PromptExample, satisfies

REC_FORMAT = "REC: <SKU> | $<price> | <RAM>GB | <weight>lbs | <battery>hrs | <one-sentence reason>"


def build_shortlist(
    example: PromptExample,
    catalog: list[Product],
    k: int = 6,
    seed: int = 0,
) -> list[str]:
    """Pick `k` candidate SKUs with a guaranteed mix of constraint-satisfying
    and violating products, so budget/coverage rewards actually vary (a
    shortlist of all-good or all-bad items would make those signals constant)."""
    rng = random.Random(f"shortlist-{example.prompt_id}-{seed}")
    good = [p for p in catalog if satisfies(p, example.constraints)]
    bad = [p for p in catalog if not satisfies(p, example.constraints)]
    rng.shuffle(good)
    rng.shuffle(bad)

    n_good = min(len(good), max(1, k // 2))
    n_bad = min(len(bad), k - n_good)
    picks = good[:n_good] + bad[:n_bad]
    rng.shuffle(picks)
    return [p.sku for p in picks]


def build_task_prompt(
    example: PromptExample,
    catalog: dict[str, Product],
    shortlist: list[str],
) -> str:
    lines = [
        "You are a shopping assistant. Recommend a laptop for the customer, "
        "choosing ONLY from the catalog below.",
        "",
        f"Customer request: {example.prompt}",
        "",
        "Catalog:",
    ]
    for sku in shortlist:
        p = catalog[sku]
        lines.append(
            f"- {sku}: {p.name} | ${p.price:.0f} | {p.ram_gb}GB RAM | "
            f"{p.weight_lbs}lbs | {p.battery_hrs}hrs battery | {p.brand}"
        )
    # Concrete example + explicit stop instruction: cuts the hallucination tail
    # (garbles happen in rambling trailing text) and nudges toward a substantive
    # comparison (which the reward now grades).
    example_sku = shortlist[0]
    ep = catalog[example_sku]
    lines += [
        "",
        "Recommend the 2 best options. Copy each product's SKU and specs EXACTLY "
        "from the catalog, and in the reason compare them on price, RAM, weight, "
        "or battery. Output ONLY the REC lines, then stop.",
        "",
        "Format (one line per recommendation):",
        REC_FORMAT,
        "Example:",
        f"REC: {example_sku} | ${ep.price:.0f} | {ep.ram_gb}GB | {ep.weight_lbs}lbs "
        f"| {ep.battery_hrs}hrs | cheaper and lighter than the alternative",
    ]
    return "\n".join(lines)
