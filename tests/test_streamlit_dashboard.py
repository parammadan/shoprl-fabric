"""Tests for the live Streamlit dashboard: data access, DEV fault injection,
and an in-process headless render of the real app (no fabrication)."""
import json

import pytest

from shoprl.platform import dash_data
from shoprl.platform.pipeline import PipelineConfig, run_pipeline

_CFG = PipelineConfig(steps=2, prompts_per_step=2, num_samples=2,
                      n_workers=2, oom_at_step=1)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("live_ro")
    run_pipeline(r, _CFG)
    return r


@pytest.fixture
def fresh(tmp_path):
    r = tmp_path / "run"
    run_pipeline(r, _CFG)
    return r


# --- readers ---------------------------------------------------------------
def test_snapshot_reads_real_state(root):
    snap = dash_data.snapshot(root)
    assert snap["job_counts"]["succeeded"] == 6
    assert snap["reward_stats"]["count"] == 8
    assert all(c["integrity"] == "OK" for c in snap["checkpoints"])
    assert len(snap["recovery_events"]) == 1


def test_trajectory_detail_real_components_kl_absent(root):
    trajs = dash_data.trajectories(root)
    assert len(trajs) == 8
    d = dash_data.trajectory_detail(root, trajs[0].id)
    assert "budget" in (d["reward_components"] or {})   # real reward components
    assert d["advantage"] is not None                   # real group-relative advantage
    assert d["kl"] is None                              # absent, NOT invented
    assert d["policy_id"].startswith("step-")


def test_comparisons_absent_is_empty(tmp_path):
    assert dash_data.comparisons(tmp_path / "does_not_exist") == []


def test_comparisons_reads_real_result_files(tmp_path):
    (tmp_path / "rloo.json").write_text(json.dumps(
        {"algorithm": "rloo", "final_kl": 0.015, "max_kl": 0.22,
         "kl_trajectory": [0.01, 0.015], "reward_gain": 0.002}))
    (tmp_path / "notes.txt").write_text("ignored")      # non-json ignored
    comps = dash_data.comparisons(tmp_path)
    assert len(comps) == 1 and comps[0]["algorithm"] == "rloo"
    assert comps[0]["final_kl"] == 0.015


# --- DEV fault injection (SIMULATION) --------------------------------------
def test_sim_kill_worker_reaps_and_requeues(fresh):
    r = dash_data.sim_kill_worker(fresh)
    assert r["ok"] and r["reaped"] == 1 and r["label"] == "SIMULATION"
    assert r["resulting_state"] == "pending"            # reaper requeued it


def test_sim_oom_writes_real_recovery_event(fresh):
    n0 = len(dash_data.snapshot(fresh)["recovery_events"])
    r = dash_data.sim_oom(fresh)
    assert r["ok"] and r["simulated"] is True
    assert r["microbatch"] == "8->4"                    # real batch adjustment
    assert len(dash_data.snapshot(fresh)["recovery_events"]) == n0 + 1


def test_sim_duplicate_trajectory_links_lineage(fresh):
    r = dash_data.sim_duplicate_trajectory(fresh)
    assert r["ok"] and r["duplicate_id"] != r["parent_id"]
    child = dash_data.trajectory_detail(fresh, r["duplicate_id"])
    assert child["parent_id"] == r["parent_id"]         # provenance preserved


# --- the real app renders headless without error ---------------------------
def test_streamlit_app_renders_without_exception(root):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("src/shoprl/platform/streamlit_app.py", default_timeout=60)
    at.run()
    at.text_input[0].set_value(str(root)).run()         # point at the test run
    assert not at.exception
    assert any("ShopRL Fabric" in t.value for t in at.title)
