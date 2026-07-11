import json

from shoprl.observability import load_metrics, render_dashboard


def _fake_metrics(n=5):
    return [{
        "step": s,
        "reward_mean": 0.50 + 0.05 * s, "reward_std": 0.2,
        "kl": 0.001 * s, "entropy": 0.12, "clip_frac": 0.0, "grad_norm": 1.0 + s,
        "reward_budget": 0.75, "reward_groundedness": 0.90, "reward_coverage": 0.80,
        "reward_quality_format": 0.90, "reward_quality_comparison": 0.30 + 0.04 * s,
        "hallucination_rate": 0.10,
    } for s in range(n)]


def test_render_produces_all_panels(tmp_path):
    out = render_dashboard(_fake_metrics(), tmp_path / "d.html")
    txt = out.read_text()
    for title in ("Reward (mean", "Reward by component", "KL vs reference",
                  "Policy entropy", "Clip fraction", "Grad norm"):
        assert title in txt
    # all five component labels present (legend + direct end-labels = relief rule)
    for lbl in ("budget", "groundedness", "coverage", "format", "comparison"):
        assert lbl in txt
    assert "<svg" in txt and "#2a78d6" in txt  # validated palette slot 1
    assert "step 0" in txt and "step 4" in txt
    assert "DATA =" in txt  # embedded hover data


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
