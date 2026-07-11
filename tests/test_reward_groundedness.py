from shoprl.reward import catalog_groundedness, is_hallucinated


def test_grounded_product_scores_full(ctx):
    # LAP-0001 exists; no specs stated to contradict -> fully clean.
    assert catalog_groundedness("I recommend LAP-0001.", ctx) == 1.0
    assert is_hallucinated("I recommend LAP-0001.", ctx) is False


def test_grounded_with_correct_specs(ctx):
    resp = "REC: LAP-0001 | $900.00 | 16GB | 3.0lbs | 12hrs | great value"
    assert catalog_groundedness(resp, ctx) == 1.0
    assert is_hallucinated(resp, ctx) is False


def test_invented_sku_is_hallucination(ctx):
    # LAP-9999 is not in the catalog.
    assert catalog_groundedness("Buy the LAP-9999.", ctx) == 0.0
    assert is_hallucinated("Buy the LAP-9999.", ctx) is True


def test_spec_lie_about_real_product_is_hallucination(ctx):
    # LAP-0001 is real but has 16GB, not 64GB — an attribute lie.
    resp = "REC: LAP-0001 | $900 | 64GB | 3.0lbs | 12hrs | tons of memory"
    assert catalog_groundedness(resp, ctx) == 0.0
    assert is_hallucinated(resp, ctx) is True


def test_mixed_grounded_and_invented(ctx):
    # One real (LAP-0001), one invented (LAP-8888) -> half clean, still flagged.
    resp = "Options: LAP-0001 and LAP-8888."
    assert catalog_groundedness(resp, ctx) == 0.5
    assert is_hallucinated(resp, ctx) is True


def test_no_claim_is_not_hallucination(ctx):
    assert catalog_groundedness("Laptops are great.", ctx) == 0.0
    assert is_hallucinated("Laptops are great.", ctx) is False
