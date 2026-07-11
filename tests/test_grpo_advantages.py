import math

from shoprl.grpo import batch_group_advantages, group_advantages


def test_advantages_are_mean_centered():
    adv = group_advantages([1.0, 2.0, 3.0])
    assert math.isclose(sum(adv), 0.0, abs_tol=1e-6)  # baseline removes the mean


def test_standardized_values_match_zscore():
    # rewards [1,2,3]: mean 2, pstdev sqrt(2/3)=0.8165. z-scores ~ [-1.22,0,1.22].
    adv = group_advantages([1.0, 2.0, 3.0], eps=0.0)
    assert math.isclose(adv[0], -1.224744871, rel_tol=1e-6)
    assert math.isclose(adv[1], 0.0, abs_tol=1e-9)
    assert math.isclose(adv[2], 1.224744871, rel_tol=1e-6)


def test_flat_group_yields_zero_not_nan():
    # The failure mode we measure: zero within-group variance -> zero gradient.
    adv = group_advantages([0.9, 0.9, 0.9, 0.9])
    assert all(a == 0.0 for a in adv)
    assert not any(math.isnan(a) for a in adv)


def test_ordering_preserved_and_sign_correct():
    adv = group_advantages([0.95, 0.2, 0.8, 0.3])
    # best sample positive, worst negative, order matches reward order.
    assert adv[0] > 0 and adv[1] < 0
    assert adv[0] == max(adv) and adv[1] == min(adv)


def test_non_standardized_is_raw_centered():
    adv = group_advantages([1.0, 2.0, 3.0], standardize=False)
    assert adv == [-1.0, 0.0, 1.0]


def test_single_sample_has_zero_advantage():
    # No siblings to compare against -> no signal.
    assert group_advantages([0.7]) == [0.0]


def test_empty_group():
    assert group_advantages([]) == []


def test_batch_maps_per_group():
    out = batch_group_advantages([[1.0, 2.0, 3.0], [0.5, 0.5]])
    assert len(out) == 2
    assert math.isclose(sum(out[0]), 0.0, abs_tol=1e-6)
    assert out[1] == [0.0, 0.0]  # flat group
