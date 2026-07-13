"""Pillar 3 tests: trajectory schema validation + lineage persistence."""
import pytest
from pydantic import ValidationError

from shoprl.platform import (Lineage, Trajectory, TrajectoryStep,
                             TrajectoryStore)
from shoprl.platform.traj_store import TrajectoryNotFound


def _traj(**kw):
    base = dict(
        prompt="find a laptop under $1200",
        steps=[TrajectoryStep(index=0, action="ADD_TO_CART LAP-0001", reward=1.0)],
        reward=1.0,
        lineage=Lineage(policy_id="step-000", job_id="J1", prompt_id="P1", seed=42),
    )
    base.update(kw)
    return Trajectory(**base)


# --- schema validation -----------------------------------------------------
def test_valid_trajectory_roundtrips_json():
    t = _traj()
    again = Trajectory.model_validate_json(t.model_dump_json())
    assert again == t and again.num_steps == 1


def test_empty_steps_rejected():
    with pytest.raises(ValidationError):
        _traj(steps=[])


def test_non_contiguous_step_indices_rejected():
    # a dropped/duplicated turn would silently misalign credit assignment
    steps = [TrajectoryStep(index=0, action="a"), TrajectoryStep(index=2, action="b")]
    with pytest.raises(ValidationError):
        _traj(steps=steps)


def test_non_finite_reward_rejected():
    with pytest.raises(ValidationError):
        _traj(reward=float("nan"))
    with pytest.raises(ValidationError):
        _traj(reward=float("inf"))


def test_from_completion_builds_single_turn():
    from shoprl.rollout.base import Completion
    c = Completion(prompt="p", text="ADD_TO_CART X")
    t = Trajectory.from_completion(
        c, reward=0.6, lineage=Lineage(policy_id="step-003"), prompt_id="P9")
    assert t.kind == "single_turn" and t.num_steps == 1
    assert t.steps[0].action == "ADD_TO_CART X"
    assert t.reward == 0.6 and t.lineage.prompt_id == "P9"


# --- persistence -----------------------------------------------------------
def test_put_get_roundtrip(tmp_path):
    s = TrajectoryStore(tmp_path / "t.db")
    t = _traj()
    s.put(t)
    assert s.get(t.id) == t


def test_missing_raises(tmp_path):
    s = TrajectoryStore(tmp_path / "t.db")
    with pytest.raises(TrajectoryNotFound):
        s.get("nope")


def test_survives_restart(tmp_path):
    path = tmp_path / "t.db"
    s1 = TrajectoryStore(path)
    t = _traj()
    s1.put(t)
    s1.close()
    s2 = TrajectoryStore(path)                       # reopen -> data recovered
    assert s2.get(t.id).reward == 1.0 and s2.count() == 1


# --- lineage queries -------------------------------------------------------
def test_query_by_job_and_policy(tmp_path):
    s = TrajectoryStore(tmp_path / "t.db")
    s.put(_traj(lineage=Lineage(policy_id="step-000", job_id="A")))
    s.put(_traj(lineage=Lineage(policy_id="step-000", job_id="A")))
    s.put(_traj(lineage=Lineage(policy_id="step-001", job_id="B")))
    assert len(s.by_job("A")) == 2
    assert len(s.by_policy("step-000")) == 2
    assert len(s.by_policy("step-001")) == 1


def test_derive_sets_parent_and_children_query(tmp_path):
    s = TrajectoryStore(tmp_path / "t.db")
    parent = _traj()
    child = parent.derive(policy_id="step-001")      # e.g. re-scored under new policy
    s.put(parent); s.put(child)
    assert child.id != parent.id
    assert child.lineage.parent_id == parent.id
    kids = s.children(parent.id)
    assert [k.id for k in kids] == [child.id]


def test_ancestry_walks_full_chain(tmp_path):
    s = TrajectoryStore(tmp_path / "t.db")
    a = _traj()
    b = a.derive(policy_id="step-001")
    c = b.derive(policy_id="step-002")
    for t in (a, b, c):
        s.put(t)
    chain = s.ancestry(c.id)
    assert [t.id for t in chain] == [a.id, b.id, c.id]   # root-first
