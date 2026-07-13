"""Retention/GC tests: bounded growth for checkpoints and trajectories."""
import time

import pytest

from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.traj_store import TrajectoryStore
from shoprl.platform.trajectory import Lineage, Trajectory, TrajectoryStep


def _src(tmp_path, name, content):
    d = tmp_path / name
    d.mkdir()
    (d / "w").write_bytes(content)
    return d


def test_checkpoint_prune_keeps_last_n(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ck")
    ids = [reg.save(_src(tmp_path, f"s{i}", bytes([i])), step=i).ckpt_id
           for i in range(5)]
    removed = reg.prune(keep_last_n=2)
    assert set(removed) == set(ids[:3])              # oldest 3 gone
    assert {m.ckpt_id for m in reg.list()} == set(ids[3:])
    for r in removed:                                # actually deleted from disk
        assert not (reg.root / r).exists()


def test_checkpoint_prune_protects_best(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ck")
    ids = [reg.save(_src(tmp_path, f"s{i}", bytes([i])), step=i).ckpt_id
           for i in range(5)]
    best = ids[0]                                    # oldest, but protected
    removed = reg.prune(keep_last_n=2, protect=[best])
    assert best not in removed
    assert best in {m.ckpt_id for m in reg.list()}


def test_checkpoint_prune_noop_when_under_limit(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ck")
    reg.save(_src(tmp_path, "s0", b"x"), step=0)
    assert reg.prune(keep_last_n=5) == []
    with pytest.raises(ValueError):
        reg.prune(keep_last_n=0)


def _traj(reward):
    return Trajectory(prompt="p", reward=reward,
                      steps=[TrajectoryStep(index=0, action="a")],
                      lineage=Lineage(policy_id="v1"))


def test_trajectory_prune_keeps_last_n(tmp_path):
    ts = TrajectoryStore(tmp_path / "t.db")
    for i in range(10):
        ts.put(_traj(i / 10))
        time.sleep(0.001)                            # distinct created_at ordering
    deleted = ts.prune(keep_last_n=3)
    assert deleted == 7 and ts.count() == 3
    kept = {round(t.reward, 3) for t in ts.recent(10)}
    assert kept == {0.7, 0.8, 0.9}                   # the three most recent


def test_trajectory_prune_zero_clears(tmp_path):
    ts = TrajectoryStore(tmp_path / "t.db")
    ts.put(_traj(0.5))
    assert ts.prune(keep_last_n=0) == 1 and ts.count() == 0
