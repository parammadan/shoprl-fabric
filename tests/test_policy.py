"""Policy registry + weight synchronization + staleness tests."""
import pytest
from fastapi.testclient import TestClient

from shoprl.platform import (PolicyClient, PolicyRegistry, TrajectoryStore,
                             staleness, staleness_report)
from shoprl.platform.api import create_app
from shoprl.platform.policy import PolicyCorrupt, PolicyNotFound
from shoprl.platform.trajectory import Lineage, Trajectory, TrajectoryStep


def _pub(reg, step):
    return reg.publish_state({"step": step, "w": [step] * 3},
                             metadata={"step": step})


# --- publish / versioning --------------------------------------------------
def test_publish_increments_version_and_latest(tmp_path):
    reg = PolicyRegistry(tmp_path / "pol")
    assert reg.latest() is None
    v1 = _pub(reg, 0)
    v2 = _pub(reg, 1)
    assert (v1.version, v2.version) == (1, 2)
    assert reg.latest().version == 2
    assert [p.version for p in reg.list()] == [1, 2]


def test_load_state_roundtrip_and_missing(tmp_path):
    reg = PolicyRegistry(tmp_path / "pol")
    _pub(reg, 5)
    assert reg.load_state(1)["step"] == 5
    with pytest.raises(PolicyNotFound):
        reg.get(99)


def test_atomic_publish_and_corruption_detected(tmp_path):
    reg = PolicyRegistry(tmp_path / "pol")
    pv = _pub(reg, 0)
    reg.verify(1)                                    # clean
    (reg.root / "v1" / "policy_state.json").write_text("tampered")
    with pytest.raises(PolicyCorrupt):
        reg.verify(1)


def test_survives_restart(tmp_path):
    root = tmp_path / "pol"
    _pub(PolicyRegistry(root), 0)
    assert PolicyRegistry(root).latest().version == 1   # fresh instance sees it


# --- weight sync (worker side) --------------------------------------------
def test_worker_refresh_loads_latest_version(tmp_path):
    reg = PolicyRegistry(tmp_path / "pol")
    _pub(reg, 0); _pub(reg, 1)
    client = PolicyClient(reg)
    assert client.refresh() == 2                     # picks up latest
    assert client.policy_id() == "v2"
    _pub(reg, 2)                                     # trainer publishes v3
    assert client.refresh() == 3                     # worker syncs forward


def test_pinned_worker_is_stale_simulation(tmp_path):
    reg = PolicyRegistry(tmp_path / "pol")
    _pub(reg, 0); _pub(reg, 1); _pub(reg, 2)         # latest is v3
    lagging = PolicyClient(reg)
    lagging.pin(1)                                   # SIMULATION: stuck on v1
    assert lagging.refresh() == 1
    assert staleness(current_version=3, trajectory_version=1) == 2


# --- staleness over persisted trajectories --------------------------------
def _traj(policy_id):
    return Trajectory(prompt="p", reward=0.5,
                      steps=[TrajectoryStep(index=0, action="a")],
                      lineage=Lineage(policy_id=policy_id))


def test_staleness_report_from_trajectory_store(tmp_path):
    ts = TrajectoryStore(tmp_path / "t.db")
    for pid in ["v3", "v3", "v1", "v2"]:             # current will be v3
        ts.put(_traj(pid))
    rep = staleness_report(ts, current_version=3)
    assert rep["n"] == 4
    assert rep["on_policy_count"] == 2               # the two v3 trajectories
    assert rep["stale_count"] == 2                   # v1 (stale 2) + v2 (stale 1)
    assert rep["max_staleness"] == 2


def test_unversioned_trajectories_reported_not_counted(tmp_path):
    ts = TrajectoryStore(tmp_path / "t.db")
    ts.put(_traj("step-000"))                        # not a v{n} tag
    rep = staleness_report(ts, current_version=1)
    assert rep["n"] == 0 and rep["unversioned"] == 1


# --- API -------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    root = tmp_path / "data"
    reg = PolicyRegistry(root / "policies")
    _pub(reg, 0); _pub(reg, 1)
    # a couple of trajectories tagged with versions, for the staleness endpoint
    ts = TrajectoryStore(root / "trajectories.db")
    ts.put(_traj("v2")); ts.put(_traj("v1")); ts.close()
    return TestClient(create_app(root, tmp_path / "runs"))


def test_api_policies(client):
    assert len(client.get("/policies").json()) == 2
    assert client.get("/policies/latest").json()["version"] == 2
    assert client.get("/policies/1").json()["version"] == 1
    assert client.get("/policies/99").status_code == 404


def test_api_staleness(client):
    body = client.get("/policies/staleness").json()
    assert body["current_version"] == 2
    assert body["n"] == 2 and body["max_staleness"] == 1   # v1 is 1 behind v2
