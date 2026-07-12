import math

from shoprl.grpo.advantages import group_advantages
from shoprl.rl.rloo import rloo_advantages


def test_rloo_leave_one_out_values():
    # rewards [1,2,3]: baseline_i = mean of the OTHER two.
    #  i0: (2+3)/2=2.5 -> 1-2.5=-1.5 ; i1: (1+3)/2=2 -> 0 ; i2: (1+2)/2=1.5 -> 1.5
    adv = rloo_advantages([1.0, 2.0, 3.0])
    assert adv == [-1.5, 0.0, 1.5]


def test_rloo_advantages_sum_to_zero():
    for rs in ([1.0, 2.0, 3.0], [0.9, 0.2, 0.8, 0.3], [5.0, 5.0]):
        assert math.isclose(sum(rloo_advantages(rs)), 0.0, abs_tol=1e-9)


def test_rloo_is_scaled_centered_reward():
    # RLOO advantage = (G/(G-1)) * (r_i - mean_group); i.e. GRPO-unstandardized x G/(G-1)
    rs = [0.2, 0.6, 1.0]
    g = len(rs)
    centered = group_advantages(rs, standardize=False)  # r_i - mean
    scaled = [c * g / (g - 1) for c in centered]
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
               for a, b in zip(rloo_advantages(rs), scaled))


def test_rloo_flat_group_zero():
    assert rloo_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0, 0.0]


def test_rloo_single_and_empty():
    assert rloo_advantages([0.7]) == [0.0]
    assert rloo_advantages([]) == []
