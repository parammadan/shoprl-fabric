from shoprl.observability.alerts import (Level, Thresholds, check_run,
                                          check_step, max_level, summarize)


def test_clean_step_no_alerts():
    m = {"step": 0, "kl": 0.05, "entropy": 0.8, "grad_norm": 2.0, "loss": 0.01}
    assert check_step(m) == []


def test_kl_blowup_critical():
    a = check_step({"step": 5, "kl": 1.2, "entropy": 0.8, "grad_norm": 2.0})
    assert any(x.rule == "kl_blowup" and x.level == Level.CRITICAL for x in a)


def test_kl_high_warning():
    a = check_step({"step": 5, "kl": 0.6, "entropy": 0.8, "grad_norm": 2.0})
    assert any(x.rule == "kl_high" and x.level == Level.WARNING for x in a)


def test_entropy_collapse():
    a = check_step({"step": 5, "kl": 0.1, "entropy": 0.01, "grad_norm": 2.0})
    assert any(x.rule == "entropy_collapse" for x in a)


def test_nonfinite_grad_critical():
    a = check_step({"step": 5, "kl": 0.1, "entropy": 0.8, "grad_norm": float("nan")})
    assert any(x.rule == "nonfinite" and x.level == Level.CRITICAL for x in a)


def test_grad_spike():
    a = check_step({"step": 5, "kl": 0.1, "entropy": 0.8, "grad_norm": 80.0})
    assert any(x.rule == "grad_spike" for x in a)


def test_reward_regression_and_stall():
    reg = check_run({"held_out_before": {"reward_mean": 0.85},
                     "held_out_after": {"reward_mean": 0.70}, "step_metrics": []})
    assert any(x.rule == "reward_regression" for x in reg)
    stall = check_run({"held_out_before": {"reward_mean": 0.841},
                       "held_out_after": {"reward_mean": 0.841}, "step_metrics": []})
    assert any(x.rule == "reward_stall" for x in stall)


def test_stability_failures_critical():
    a = check_run({"stability_failures": 3, "step_metrics": []})
    assert any(x.rule == "stability_failures" and x.level == Level.CRITICAL for x in a)


def test_summarize_groups_by_rule():
    alerts = check_run({"step_metrics": [
        {"step": 1, "kl": 0.6, "entropy": 0.8, "grad_norm": 1.0},
        {"step": 2, "kl": 0.7, "entropy": 0.8, "grad_norm": 1.0},
    ]})
    s = summarize(alerts)
    assert s["kl_high"]["count"] == 2 and s["kl_high"]["first_step"] == 1
    assert max_level(alerts) == Level.WARNING
