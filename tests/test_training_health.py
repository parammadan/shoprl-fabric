"""Training Health data path: comparison artifacts + metrics-run endpoints,
served read-only via the API (no store/trainer reads in the UI)."""
import json

import pytest
from fastapi.testclient import TestClient

from shoprl.platform.api import create_app


@pytest.fixture
def client(tmp_path):
    # a committed-style comparison dir + a run with metrics.jsonl
    comp = tmp_path / "comparisons"
    comp.mkdir()
    (comp / "rloo.json").write_text(json.dumps({
        "algorithm": "rloo", "final_kl": 0.015, "max_kl": 0.22, "reward_gain": 0.002,
        "stability_failures": 0,
        "step_metrics": [{"step": i, "kl": 0.01 + i * 0.0002, "entropy": 3.0,
                          "clip_frac": 0.1, "grad_norm": 2.0, "reward_mean": 0.8,
                          "reward_std": 0.1} for i in range(30)]}))
    (comp / "ppo.json").write_text(json.dumps({
        "algorithm": "ppo", "final_kl": 6.78, "max_kl": 6.78, "reward_gain": 0.0006,
        "stability_failures": 0,
        "step_metrics": [{"step": i, "kl": 6.78, "entropy": 2.0, "clip_frac": 0.3,
                          "grad_norm": 3.0, "reward_mean": 0.8, "reward_std": 0.1}
                         for i in range(30)]}))          # kl 6.78 >= 1.0 -> 30 kl_blowup
    runs = tmp_path / "runs"
    (runs / "smoke").mkdir(parents=True)
    (runs / "smoke" / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "kl": 0.01, "entropy": 3.0, "clip_frac": 0.1, "grad_norm": 2.0,
         "reward_mean": 0.8, "reward_std": 0.1, "reward_budget": 0.9},
        {"step": 1, "kl": 0.58, "entropy": 2.5, "clip_frac": 0.2, "grad_norm": 3.0,
         "reward_mean": 0.82, "reward_std": 0.09, "reward_budget": 0.92}]))
    return TestClient(create_app(tmp_path / "data", runs, comp))


def test_comparisons_endpoint_serves_real_artifacts_with_alerts(client):
    comps = client.get("/comparisons").json()
    by_algo = {c["algorithm"]: c for c in comps}
    assert set(by_algo) == {"rloo", "ppo"}
    assert by_algo["rloo"]["final_kl"] == 0.015          # measured, not fabricated
    assert by_algo["ppo"]["final_kl"] == 6.78
    # PPO critical KL alerts computed server-side over persisted step_metrics
    assert by_algo["ppo"]["alerts"]["critical"] == 30
    assert by_algo["ppo"]["alerts"]["by_rule"]["kl_blowup"] == 30
    assert by_algo["rloo"]["alerts"]["critical"] == 0    # RLOO quiet
    assert len(by_algo["rloo"]["step_metrics"]) == 30    # real time-series for overlays


def test_comparisons_absent_is_empty(tmp_path):
    c = TestClient(create_app(tmp_path / "d", tmp_path / "r", tmp_path / "none"))
    assert c.get("/comparisons").json() == []            # absent, not invented


def test_metrics_runs_lists_runs_with_metrics(client):
    assert client.get("/metrics-runs").json() == ["smoke"]


def test_single_run_metrics_and_alerts(client):
    m = client.get("/runs/smoke/metrics").json()
    assert m["n_steps"] == 2
    keys = m["metrics"][0].keys()
    assert "reward_std" in keys and "clip_frac" in keys and "reward_budget" in keys
    al = client.get("/runs/smoke/alerts").json()
    assert any(a["rule"] == "kl_high" for a in al["alerts"])   # kl 0.58 >= warn 0.5
