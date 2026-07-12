import time
import torch
from shoprl.bench.profiler import PhaseTimer, padding_waste


def test_phase_timer_accumulates_and_reports():
    pt = PhaseTimer()
    with pt.phase("rollout"): time.sleep(0.02)
    with pt.phase("rollout"): time.sleep(0.02)
    with pt.phase("optimize"): time.sleep(0.01)
    r = pt.report(total_tokens=1000)
    assert r["breakdown"]["rollout"]["calls"] == 2
    assert r["breakdown"]["rollout"]["seconds"] >= r["breakdown"]["optimize"]["seconds"]
    assert r["total_s"] > 0 and "tokens_per_sec" in r
    assert abs(sum(b["pct"] for b in r["breakdown"].values()) - 100.0) < 1.0


def test_padding_waste():
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])  # 6 real / 8 -> 0.25 waste
    assert abs(padding_waste(m) - 0.25) < 1e-6
    assert padding_waste(torch.ones(2, 5)) == 0.0
