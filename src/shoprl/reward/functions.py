"""The four reward functions. Each is pure and deterministic: same
(response, context) -> same score, no I/O, no globals.

Convention: every component returns a scalar in [0, 1] (higher = better). The
composite applies the weights and the separate hallucination penalty.

Ground-truth rule for the numeric checks (budget, coverage): we look up the
recommended SKU's *true* specs in the catalog and check those. We never trust
the specs the response states — otherwise the policy could "pass" the budget by
lying about the price. Catching that lie is groundedness's job, not budget's.
"""
from __future__ import annotations

from dataclasses import dataclass

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.reward.parse import ParsedRec, parse_response

# Tolerances for judging whether a *stated* spec matches the catalog. Price and
# weight get slack for rounding ("$1299" vs 1299.00); RAM/battery are integers.
_PRICE_TOL = 1.0
_WEIGHT_TOL = 0.1

# Words that signal the response actually compared options (for quality).
_COMPARISON_WORDS = (
    "compared", "comparison", "whereas", "however", "better", "best",
    "versus", " vs", "trade-off", "tradeoff", "on the other hand", "while",
)

# Comparison score multiplier when no comparative language is present but the
# response still recommends products (partial credit).
_NO_COMPARISON_FACTOR = 0.6


@dataclass
class RewardContext:
    """Everything a reward needs to judge one response: the catalog (ground
    truth) and the active constraints for this prompt."""

    catalog: dict[str, Product]  # sku -> Product
    constraints: dict[str, float]


def _grounded(recs: list[ParsedRec], ctx: RewardContext) -> list[ParsedRec]:
    return [r for r in recs if r.sku in ctx.catalog]


def _specs_consistent(rec: ParsedRec, p: Product) -> bool:
    """True if every spec the response *stated* matches the catalog truth."""
    if rec.stated_price is not None and abs(rec.stated_price - p.price) > _PRICE_TOL:
        return False
    if rec.stated_ram is not None and rec.stated_ram != p.ram_gb:
        return False
    if rec.stated_weight is not None and abs(rec.stated_weight - p.weight_lbs) > _WEIGHT_TOL:
        return False
    if rec.stated_battery is not None and rec.stated_battery != p.battery_hrs:
        return False
    return True


# --- 1. budget_compliance -------------------------------------------------
def budget_compliance(response: str, ctx: RewardContext) -> float:
    """Fraction of recommended (grounded) products whose TRUE price is within
    the stated max_price. If the prompt has no price constraint, nothing to
    violate -> 1.0. If nothing valid was recommended -> 0.0."""
    if "max_price" not in ctx.constraints:
        return 1.0
    grounded = _grounded(parse_response(response), ctx)
    if not grounded:
        return 0.0
    cap = ctx.constraints["max_price"]
    ok = sum(1 for r in grounded if ctx.catalog[r.sku].price <= cap)
    return ok / len(grounded)


# --- 2. catalog_groundedness ---------------------------------------------
def catalog_groundedness(response: str, ctx: RewardContext) -> float:
    """Fraction of claims that are CLEAN: the SKU exists AND any stated specs
    match the catalog. 0.0 if nothing was claimed. Smooth signal; the flat
    penalty for lying lives in the composite via `is_hallucinated`."""
    recs = parse_response(response)
    if not recs:
        return 0.0
    clean = sum(
        1
        for r in recs
        if r.sku in ctx.catalog and _specs_consistent(r, ctx.catalog[r.sku])
    )
    return clean / len(recs)


def is_hallucinated(response: str, ctx: RewardContext) -> bool:
    """True if the response cited a non-existent SKU OR stated a spec that
    contradicts the catalog. Triggers the composite's -0.50 penalty."""
    for r in parse_response(response):
        if r.sku not in ctx.catalog:
            return True
        if not _specs_consistent(r, ctx.catalog[r.sku]):
            return True
    return False


# --- 3. attribute_coverage ------------------------------------------------
def attribute_coverage(response: str, ctx: RewardContext) -> float:
    """Fraction of recommended (grounded) products that satisfy ALL active
    constraints, judged against catalog truth via the same `satisfies`
    predicate that defined the ground-truth answer set."""
    grounded = _grounded(parse_response(response), ctx)
    if not grounded:
        return 0.0
    if not ctx.constraints:
        return 1.0
    ok = sum(1 for r in grounded if satisfies(ctx.catalog[r.sku], ctx.constraints))
    return ok / len(grounded)


# --- 4. response_quality (format + comparison) ---------------------------
def response_quality(response: str, ctx: RewardContext) -> tuple[float, float]:
    """Returns (format, comparison), each in [0, 1].

    format: are recommendations parseable and fully specified? Mean over recs
      of (specs_stated / 4). Rewards citing a SKU with all four specs — which
      is exactly what makes the response machine-checkable.
    comparison: did it help the user choose? Recommending >=2 products earns
      full breadth; comparative language earns the rest. A single bare pick
      scores low."""
    recs = parse_response(response)
    if not recs:
        return 0.0, 0.0

    fmt = sum(r.num_specs_stated / 4 for r in recs) / len(recs)

    grounded = _grounded(recs, ctx)
    breadth = min(len(grounded), 2) / 2  # 0.5 for one, 1.0 for two+
    has_comparison = any(w in response.lower() for w in _COMPARISON_WORDS)
    comparison = breadth * (1.0 if has_comparison else _NO_COMPARISON_FACTOR)

    return fmt, comparison
