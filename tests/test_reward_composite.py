import pytest

from shoprl.reward import HALLUCINATION_PENALTY, WEIGHTS, compute_reward

PERFECT = (
    "REC: LAP-0001 | $900 | 16GB | 3.0lbs | 12hrs | best battery life\n"
    "REC: LAP-0003 | $700 | 32GB | 2.5lbs | 15hrs | lighter, whereas this wins on weight"
)


def test_positive_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_response_approaches_one(ctx):
    r = compute_reward(PERFECT, ctx)
    assert r.hallucinated is False
    assert r.budget == 1.0 and r.groundedness == 1.0 and r.coverage == 1.0
    assert r.quality_format == 1.0 and r.quality_comparison == 1.0
    assert r.total == pytest.approx(1.0)


def test_hallucination_drives_total_negative(ctx):
    r = compute_reward("Definitely buy the LAP-9999, it's amazing.", ctx)
    assert r.hallucinated is True
    assert r.groundedness == 0.0
    # Nothing grounded -> all positive terms zero; only the penalty remains.
    assert r.total == pytest.approx(-HALLUCINATION_PENALTY)
    assert r.total < 0


def test_spec_lie_penalized_even_with_valid_pick(ctx):
    # Real SKU, in budget, meets constraints — but lies about RAM (64 vs 16).
    liar = "REC: LAP-0001 | $900 | 64GB | 3.0lbs | 12hrs | tons of RAM"
    r = compute_reward(liar, ctx)
    assert r.hallucinated is True
    assert r.budget == 1.0  # true price still under cap
    assert r.groundedness == 0.0  # the lie makes the claim unclean
    assert r.total < 0.5  # penalty dominates
