from shoprl.reward import response_quality


def test_fully_specified_format_scores_full(ctx):
    resp = "REC: LAP-0001 | $900 | 16GB | 3.0lbs | 12hrs | great pick"
    fmt, _ = response_quality(resp, ctx)
    assert fmt == 1.0


def test_bare_sku_has_low_format(ctx):
    # SKU cited but no specs stated -> 0/4 specs.
    fmt, _ = response_quality("I recommend LAP-0001.", ctx)
    assert fmt == 0.0


def test_partial_format(ctx):
    # Two of four specs stated -> 0.5.
    fmt, _ = response_quality("LAP-0001 costs $900 with 16GB.", ctx)
    assert fmt == 0.5


def test_no_recommendation_scores_zero(ctx):
    fmt, comp = response_quality("Laptops are nice.", ctx)
    assert fmt == 0.0 and comp == 0.0


def test_comparison_rewards_multiple_and_language(ctx):
    resp = "LAP-0001 is better for battery, whereas LAP-0003 is lighter."
    _, comp = response_quality(resp, ctx)
    assert comp == 1.0  # two grounded recs + comparative language


def test_single_pick_has_lower_comparison(ctx):
    _, comp = response_quality("I recommend LAP-0001.", ctx)
    # One rec (breadth 0.5), no comparison language (factor 0.6) -> 0.3.
    assert comp == 0.3
