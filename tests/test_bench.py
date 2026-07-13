import time
import torch
from shoprl.bench.profiler import PhaseTimer, padding_waste, rollout_metrics


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


def test_rollout_metrics_derivations():
    # 64 completions in 8s of rollout, 10 steps, 20s total, ttft supplied
    m = rollout_metrics(n_completions=64, rollout_seconds=8.0, steps=10,
                        total_s=20.0, ttft_ms=42.0)
    assert m["requests_per_sec"] == 8.0                 # 64 / 8
    assert m["rollout_latency_ms_per_request"] == 125.0  # 8000ms / 64
    assert m["iteration_time_s"] == 2.0                 # 20 / 10
    assert m["ttft_ms"] == 42.0


def test_rollout_metrics_guards_zeroes():
    m = rollout_metrics(n_completions=0, rollout_seconds=0.0, steps=0, total_s=0.0)
    assert m["requests_per_sec"] is None
    assert m["rollout_latency_ms_per_request"] is None
    assert m["iteration_time_s"] is None
    assert m["ttft_ms"] is None                          # absent, not fabricated
