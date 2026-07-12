import json

from shoprl.observability import load_metrics, render_dashboard
from shoprl.observability.dashboard import render_overlay


def _fake_metrics(n=5):
    return [{
        "step": s,
        "reward_mean": 0.50 + 0.05 * s, "reward_std": 0.2,
        "kl": 0.001 * s, "entropy": 0.12, "clip_frac": 0.0, "grad_norm": 1.0 + s,
        "reward_budget": 0.75, "reward_groundedness": 0.90, "reward_coverage": 0.80,
        "reward_quality_format": 0.90, "reward_quality_comparison": 0.30 + 0.04 * s,
        "hallucination_rate": 0.10, "value_loss": 0.5,
    } for s in range(n)]


def _fake_result(algo, n=5, gain=0.02):
    return {"algorithm": algo, "steps": n, "step_metrics": _fake_metrics(n),
            "reward_gain": gain, "final_kl": 0.3, "max_kl": 0.5, "train_time_s": 120.0,
            "peak_mem_gb": 7.3, "tokens_per_sec": 100.0, "mean_rollout_reward_std": 0.1,
            "stability_failures": 0}


def test_render_produces_all_panels(tmp_path):
    out = render_dashboard(_fake_metrics(), tmp_path / "d.html")
    txt = out.read_text()
    for title in ("Reward (mean", "Reward by component", "KL vs reference",
                  "Policy entropy", "Clip fraction", "Grad norm",
                  "Hallucination rate", "PPO value loss"):
        assert title in txt
    for lbl in ("budget", "groundedness", "coverage", "format", "comparison"):
        assert lbl in txt                          # component-panel series (legend)
    assert "#2a78d6" in txt and "DATA =" in txt    # validated palette + hover data


def test_overlay_multi_algorithm(tmp_path):
    out = render_overlay([_fake_result("grpo"), _fake_result("rloo"), _fake_result("ppo")],
                         tmp_path / "o.html")
    txt = out.read_text()
    assert "Run summary" in txt                       # summary table
    for algo in ("grpo", "rloo", "ppo"):
        assert algo in txt                            # all three overlaid
    assert txt.count("<svg") == 4                     # reward/kl/entropy/grad_norm
    assert "tok/s" in txt and "peak GB" in txt        # surfaced run-level metrics


def test_load_metrics_roundtrip(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _fake_metrics(3)) + "\n")
    m = load_metrics(p)
    assert len(m) == 3 and m[0]["step"] == 0


def test_single_step_does_not_crash(tmp_path):
    out = render_dashboard(_fake_metrics(1), tmp_path / "one.html")
    assert out.exists() and "<svg" in out.read_text()


def test_constant_series_scales(tmp_path):
    # flat metric (min == max) must not divide by zero
    rows = [{"step": s, "reward_mean": 0.9, "reward_std": 0.0, "kl": 0.0,
             "entropy": 0.0, "clip_frac": 0.0, "grad_norm": 0.0,
             "reward_budget": 0.9, "reward_groundedness": 0.9, "reward_coverage": 0.9,
             "reward_quality_format": 0.9, "reward_quality_comparison": 0.9,
             "hallucination_rate": 0.0} for s in range(3)]
    out = render_dashboard(rows, tmp_path / "flat.html")
    assert out.exists()
