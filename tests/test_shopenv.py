import pytest

from shoprl.data.catalog import Product
from shoprl.env import Goal, ShopEnv, parse_action

CAT = [
    Product(sku="LAP-0001", name="A", price=900.0, ram_gb=16, weight_lbs=3.0, battery_hrs=12, brand="Acer"),
    Product(sku="LAP-0002", name="B", price=1500.0, ram_gb=8, weight_lbs=5.0, battery_hrs=6, brand="Dell"),
    Product(sku="LAP-0003", name="C", price=700.0, ram_gb=32, weight_lbs=2.5, battery_hrs=15, brand="Apple"),
]


def _env(budget=1200.0, constraints=None, max_turns=8):
    return ShopEnv(CAT, Goal(budget=budget, constraints=constraints or {"min_ram": 16}), max_turns)


def test_parse_action_forms():
    assert parse_action("ADD_TO_CART[LAP-0001]").kind == "ADD_TO_CART"
    assert parse_action("ADD_TO_CART[LAP-0001]").arg == "LAP-0001"
    assert parse_action("let's CHECKOUT now").kind == "CHECKOUT"
    assert parse_action("APPLY_FILTER[max_price=1200, min_ram=16]").kind == "APPLY_FILTER"
    assert parse_action("blah blah").kind == "INVALID"


def test_reset_and_context():
    e = _env()
    ctx = e.reset()
    assert "GOAL" in ctx and "budget" in ctx and e.state.turn == 0


def test_add_to_cart_deducts_budget():
    e = _env(); e.reset()
    _, done, info = e.step("ADD_TO_CART[LAP-0001]")
    assert info["valid"] and not done
    assert e.state.cart == ["LAP-0001"] and e.state.budget_remaining == 300.0


def test_add_over_budget_rejected():
    e = _env(budget=1000.0); e.reset()
    _, _, info = e.step("ADD_TO_CART[LAP-0002]")   # $1500 > $1000
    assert not info["valid"] and e.state.cart == [] and e.state.budget_remaining == 1000.0


def test_add_unknown_sku_rejected():
    e = _env(); e.reset()
    _, _, info = e.step("ADD_TO_CART[LAP-9999]")
    assert not info["valid"] and e.state.cart == []


def test_remove_refunds():
    e = _env(); e.reset()
    e.step("ADD_TO_CART[LAP-0001]")
    _, _, info = e.step("REMOVE[LAP-0001]")
    assert info["valid"] and e.state.cart == [] and e.state.budget_remaining == 1200.0


def test_apply_filter_changes_candidates():
    e = _env(); e.reset()
    e.step("APPLY_FILTER[min_ram=16]")
    skus = {p.sku for p in e.candidates()}
    assert skus == {"LAP-0001", "LAP-0003"}   # LAP-0002 (8GB) filtered out


def test_checkout_ends_episode():
    e = _env(); e.reset()
    _, done, info = e.step("CHECKOUT")
    assert done and e.state.checked_out and info["action"] == "CHECKOUT"


def test_max_turns_ends_without_checkout():
    e = _env(max_turns=2); e.reset()
    _, d1, _ = e.step("APPLY_FILTER[min_ram=16]")
    _, d2, _ = e.step("APPLY_FILTER[min_battery=10]")
    assert not d1 and d2 and not e.state.checked_out    # ended by max_turns


def test_step_after_done_is_noop():
    e = _env(); e.reset(); e.step("CHECKOUT")
    _, done, info = e.step("ADD_TO_CART[LAP-0001]")
    assert done and info["action"] == "NOOP"
