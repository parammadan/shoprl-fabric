"""Ops console tests: new API reads, ApiClient (in-process ASGI), dev controls,
and a headless render of the real console app."""
import json
import os

import pytest

from shoprl.platform.api_client import ApiClient
from shoprl.platform.pipeline import PipelineConfig, run_pipeline
from shoprl.platform.policy import PolicyRegistry
from shoprl.platform.registry import ExperimentRegistry, record_from_config
from shoprl.config import Config


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("ops")
    root = base / "data"
    run_pipeline(root, PipelineConfig(steps=1, prompts_per_step=2, num_samples=2,
                                      n_workers=2, oom_at_step=None))
    # a policy + a registered run + a run metrics file with a KL blowup
    PolicyRegistry(root / "policies").publish_state({"w": [1]}, metadata={"step": 0})
    reg = ExperimentRegistry(str(root / "registry.db"))
    reg.save(record_from_config(Config()))
    runs = base / "runs"
    (runs / "exp1").mkdir(parents=True)
    (runs / "exp1" / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "kl": 0.01, "entropy": 3.0, "grad_norm": 2.0, "reward_mean": 0.5},
        {"step": 1, "kl": 6.78, "entropy": 2.0, "grad_norm": 3.0, "reward_mean": 0.5}]))
    return root, runs


@pytest.fixture
def api(env):
    root, runs = env
    return ApiClient.in_process(root, runs)


# --- new operational reads (via the client -> API -> stores) ---------------
def test_overview_and_scheduler(api):
    ov = api.overview()
    assert ov["job_counts"]["succeeded"] > 0
    s = api.scheduler()
    assert set(s.keys()) >= {"gpu", "cpu", "running_total", "at_capacity"}


def test_jobs_checkpoints_trajectories(api):
    assert isinstance(api.jobs(), list) and len(api.jobs()) > 0
    cks = api.checkpoints()
    assert cks and all(c["integrity"] == "OK" for c in cks)
    trajs = api.trajectories(limit=10)
    assert trajs and "policy_id" in trajs[0]
    detail = api.trajectory(trajs[0]["id"])
    assert detail["kl"] is None                      # absent, not invented


def test_policies_and_health(api):
    assert len(api.policies()) == 1
    assert api.health()["status"] == "ok"


def test_run_metrics_and_alerts_fire_on_kl_blowup(api):
    m = api.run_metrics("exp1")
    assert m["n_steps"] == 2
    al = api.run_alerts("exp1")
    assert al["n_alerts"] >= 1
    assert al["max_level"] == "critical"             # kl 6.78 -> kl_blowup
    assert any(a["rule"] == "kl_blowup" for a in al["alerts"])


def test_run_metrics_absent_returns_none(api):
    assert api.run_metrics("ghost") is None          # honest absence, not fabricated


# --- dev-mode controls -----------------------------------------------------
def test_dev_kill_worker_and_replay(api):
    r = api.kill_worker()
    assert r["label"] == "SIMULATION" and r["resulting_state"] == "pending"
    tid = api.trajectories(limit=1)[0]["id"]
    rep = api.replay(tid)
    assert rep["label"] == "SIMULATION" and rep["duplicate_id"] != tid


def test_pause_resume_cancel_via_api(api):
    jid = api._post("/jobs", {"kind": "optimize"})["id"]
    assert api.pause(jid)["state"] == "paused"
    assert api.resume(jid)["state"] == "pending"
    assert api.cancel(jid)["state"] == "cancelled"


# --- the real console renders headless -------------------------------------
def test_ops_console_renders_without_exception(env):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    root, runs = env
    os.environ["SHOPRL_ROOT"] = str(root)
    os.environ["SHOPRL_RUNS"] = str(runs)
    try:
        at = AppTest.from_file("src/shoprl/platform/ops_console.py", default_timeout=90)
        at.run()
        assert not at.exception
        assert any("operations console" in t.value for t in at.title)
    finally:
        os.environ.pop("SHOPRL_ROOT", None)
        os.environ.pop("SHOPRL_RUNS", None)
