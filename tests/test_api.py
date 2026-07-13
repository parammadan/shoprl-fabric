"""Pillar (API) tests: validated HTTP boundary over the existing stores."""
import json

import pytest
from fastapi.testclient import TestClient

from shoprl.platform import dash_data
from shoprl.platform.api import create_app
from shoprl.platform.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    base = tmp_path_factory.mktemp("api")
    root = base / "data"
    run_pipeline(root, PipelineConfig(steps=1, prompts_per_step=2, num_samples=2,
                                      n_workers=2, oom_at_step=None))
    runs = base / "runs"
    (runs / "exp1").mkdir(parents=True)
    (runs / "exp1" / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "kl": 0.01, "entropy": 3.1}, {"step": 1, "kl": 0.58, "entropy": 2.0}]))
    client = TestClient(create_app(root, runs))
    return client, root


# --- create + validation ---------------------------------------------------
def test_create_job_201(ctx):
    client, _ = ctx
    r = client.post("/jobs", json={"kind": "rollout", "payload": {"p": 1}})
    assert r.status_code == 201
    body = r.json()
    assert body["kind"] == "rollout" and body["state"] == "pending" and body["id"]


def test_create_validation_errors_422(ctx):
    client, _ = ctx
    assert client.post("/jobs", json={"kind": ""}).status_code == 422        # empty kind
    assert client.post("/jobs", json={"kind": "x", "max_attempts": 0}).status_code == 422
    assert client.post("/jobs", json={"payload": {}}).status_code == 422       # kind missing


# --- get --------------------------------------------------------------------
def test_get_job_and_404(ctx):
    client, _ = ctx
    jid = client.post("/jobs", json={"kind": "reward"}).json()["id"]
    assert client.get(f"/jobs/{jid}").json()["id"] == jid
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404 and "not found" in r.json()["detail"]


# --- lifecycle: pause / resume / cancel ------------------------------------
def test_pause_resume_cancel_flow(ctx):
    client, _ = ctx
    jid = client.post("/jobs", json={"kind": "optimize"}).json()["id"]
    assert client.post(f"/jobs/{jid}/pause").json()["state"] == "paused"
    assert client.post(f"/jobs/{jid}/resume").json()["state"] == "pending"
    assert client.post(f"/jobs/{jid}/cancel").json()["state"] == "cancelled"


def test_illegal_transitions_return_409(ctx):
    client, _ = ctx
    jid = client.post("/jobs", json={"kind": "optimize"}).json()["id"]
    client.post(f"/jobs/{jid}/cancel")                       # now terminal
    assert client.post(f"/jobs/{jid}/pause").status_code == 409   # can't pause terminal
    assert client.post(f"/jobs/{jid}/cancel").status_code == 409  # can't cancel terminal
    # resume a job that isn't paused
    jid2 = client.post("/jobs", json={"kind": "reward"}).json()["id"]
    assert client.post(f"/jobs/{jid2}/resume").status_code == 409


def test_lifecycle_404_on_missing(ctx):
    client, _ = ctx
    for verb in ("pause", "resume", "cancel"):
        assert client.post(f"/jobs/nope/{verb}").status_code == 404


# --- runs metrics -----------------------------------------------------------
def test_run_metrics_ok_404_and_traversal_guard(ctx):
    client, _ = ctx
    r = client.get("/runs/exp1/metrics")
    assert r.status_code == 200
    assert r.json()["n_steps"] == 2 and r.json()["metrics"][1]["kl"] == 0.58
    assert client.get("/runs/ghost/metrics").status_code == 404
    assert client.get("/runs/..%2f..%2fetc/metrics").status_code in (400, 404)


# --- trajectories -----------------------------------------------------------
def test_trajectory_detail_and_404(ctx):
    client, root = ctx
    tid = dash_data.trajectories(root, 1)[0].id
    body = client.get(f"/trajectories/{tid}").json()
    assert body["id"] == tid
    assert "budget" in body["reward_components"]            # real components
    assert body["kl"] is None                              # absent, not invented
    assert client.get("/trajectories/nope").status_code == 404
