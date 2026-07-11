from shoprl.reward import attribute_coverage
from shoprl.reward.functions import RewardContext


def test_satisfying_product_scores_full(ctx):
    # LAP-0001: $900 <= 1000 and 16GB >= 16 -> satisfies all constraints.
    assert attribute_coverage("I recommend LAP-0001.", ctx) == 1.0


def test_violating_product_scores_zero(ctx):
    # LAP-0002: $1500 and 8GB -> violates both constraints.
    assert attribute_coverage("Go with LAP-0002.", ctx) == 0.0


def test_partial_coverage(ctx):
    # LAP-0001 satisfies, LAP-0002 does not -> 0.5.
    assert attribute_coverage("Try LAP-0001 or LAP-0002.", ctx) == 0.5


def test_uses_catalog_truth_not_stated_specs(ctx):
    # Response lies that LAP-0002 has 32GB, but coverage checks the CATALOG
    # (8GB), so it still fails min_ram.
    resp = "REC: LAP-0002 | $1500 | 32GB | 5.0lbs | 6hrs | plenty of RAM"
    assert attribute_coverage(resp, ctx) == 0.0


def test_no_constraints_is_full(catalog):
    ctx = RewardContext(catalog=catalog, constraints={})
    assert attribute_coverage("I recommend LAP-0002.", ctx) == 1.0
