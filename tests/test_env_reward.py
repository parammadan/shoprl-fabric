import math

import pytest

from shoprl.data.catalog import Product
from shoprl.env import EnvState, Goal, assign_credit, episode_reward

IDX = {
    "LAP-0001": Product(sku="LAP-0001", name="A", price=900.0, ram_gb=16, weight_lbs=3.0, battery_hrs=12, brand="Acer"),
    "LAP-0002": Product(sku="LAP-0002", name="B", price=1500.0, ram_gb=8, weight_lbs=5.0, battery_hrs=6, brand="Dell"),
    "LAP-0003": Product(sku="LAP-0003", name="C", price=700.0, ram_gb=32, weight_lbs=2.5, battery_hrs=15, brand="Apple"),
}
GOAL = Goal(budget=1200.0, constraints={"min_ram": 16}, target_items=1)


def _state(cart, checked_out=True):
    return EnvState(budget_remaining=0.0, cart=list(cart), checked_out=checked_out, done=True)


def test_reward_perfect_purchase():
    assert episode_reward(_state(["LAP-0001"]), GOAL, IDX) == pytest.approx(1.0)


def test_reward_constraint_violation():
    # LAP-0002 is 8GB < min_ram 16 -> constraints 0, count 1 -> 0.4
    assert episode_reward(_state(["LAP-0002"]), GOAL, IDX) == pytest.approx(0.4)


def test_reward_wrong_item_count():
    # two items, target 1 -> count_ok 0; both satisfy -> 0.6*1 + 0.4*0 = 0.6
    assert episode_reward(_state(["LAP-0001", "LAP-0003"]), GOAL, IDX) == pytest.approx(0.6)


def test_reward_no_checkout_is_zero():
    assert episode_reward(_state(["LAP-0001"], checked_out=False), GOAL, IDX) == 0.0


def test_reward_empty_cart_is_zero():
    assert episode_reward(_state([]), GOAL, IDX) == 0.0


def test_credit_uniform():
    assert assign_credit(1.0, 3) == [1.0, 1.0, 1.0]
    assert assign_credit(1.0, 3, baseline=0.5) == [0.5, 0.5, 0.5]


def test_credit_discounted_later_turns_get_more():
    adv = assign_credit(1.0, 3, scheme="discounted", gamma=0.9)
    assert adv[-1] == pytest.approx(1.0)          # last turn (nearest reward) full
    assert math.isclose(adv[0], 0.81, rel_tol=1e-6)  # earliest turn discounted
    assert adv[0] < adv[1] < adv[2]


def test_credit_edge_cases():
    assert assign_credit(1.0, 0) == []
    with pytest.raises(ValueError):
        assign_credit(1.0, 3, scheme="bogus")
