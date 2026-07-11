from shoprl.reward import budget_compliance
from shoprl.reward.functions import RewardContext


def test_within_budget_scores_full(ctx):
    # LAP-0001 is $900, under the $1000 cap.
    assert budget_compliance("I recommend LAP-0001.", ctx) == 1.0


def test_budget_violation_scores_zero(ctx):
    # LAP-0002 is $1500, over the cap — a budget violation.
    assert budget_compliance("Go with LAP-0002.", ctx) == 0.0


def test_partial_compliance(ctx):
    # One compliant (LAP-0001) + one over-budget (LAP-0002) -> 0.5.
    assert budget_compliance("Consider LAP-0001 or LAP-0002.", ctx) == 0.5


def test_no_price_constraint_is_vacuously_full(catalog):
    ctx = RewardContext(catalog=catalog, constraints={"min_ram": 16})
    assert budget_compliance("Go with LAP-0002.", ctx) == 1.0


def test_no_valid_recommendation_scores_zero(ctx):
    # Hallucinated SKU is not grounded -> nothing to price-check -> 0.0.
    assert budget_compliance("Buy the LAP-9999.", ctx) == 0.0
