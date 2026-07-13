"""Control-plane tests: real RL training routed through job -> scheduler ->
worker -> platform. Uses a fake runner (no model) so the FULL route to the
registries is exercised offline; a separate real CPU smoke covers the trainer."""
import json
import pathlib
import tempfile

import pytest
from fastapi.testclient import TestClient

from shoprl.platform import (ArtifactRegistry, ArtifactType, JobStore,
                             PolicyRegistry, ResourceConfig, Scheduler,
                             TrajectoryStore)
from shoprl.platform.api import create_app
from shoprl.platform.control import TRAIN_KIND, serve_pending, submit_training
from shoprl.platform.jobs import JobState
from shoprl.platform.registry import ExperimentRegistry, RunStatus

CFG = "configs/smoke_cpu.yaml"


def fake_runner(config, n_prompts, num_samples) -> dict:
    """Stands in for run_experiment: produces a real temp checkpoint dir + a
    result + eval samples, WITHOUT building a model."""
    d = tempfile.mkdtemp(prefix="fakeckpt-")
    (pathlib.Path(d) / "adapter_model.safetensors").write_bytes(b"\x00weights")
    (pathlib.Path(d) / "train_state.json").write_text('{"step": 2}')
    result = {"algorithm": config.algorithm, "reward_gain": 0.01, "final_kl": 0.02,
              "max_kl": 0.03, "train_time_s": 12.0, "gpu": {"available": False},
              "held_out_before": {"reward_mean": 0.80, "n_completions": 8},
              "held_out_after": {"reward_mean": 0.81},
              "step_metrics": [{"kl": 0.02, "reward_std": 0.1}]}
    samples = [{"prompt": "find a laptop", "response": "ADD_TO_CART LAP-1",
                "reward": 0.5, "components": {"total": 0.5, "budget": 1.0},
                "prompt_id": "P-1"}]
    return {"checkpoint_dir": d, "result": result, "samples": samples}


def _sched(root, gpu_slots=1):
    store = JobStore(str(root / "jobs.db"))
    return store, Scheduler(store, ResourceConfig(gpu_slots=gpu_slots))


def test_submit_creates_pending_gpu_job(tmp_path):
    store, _ = _sched(tmp_path)
    job = submit_training(store, CFG, n_prompts=8, num_samples=2,
                          platform_root=str(tmp_path))
    assert job.state is JobState.PENDING and job.resource == "gpu"
    assert job.kind == TRAIN_KIND


def test_full_route_populates_every_registry(tmp_path):
    store, sch = _sched(tmp_path)
    job = submit_training(store, CFG, n_prompts=8, num_samples=2,
                          platform_root=str(tmp_path))
    results = serve_pending(sch, runner=fake_runner)            # admit + execute

    assert results and results[0]["status"] == "succeeded"
    assert store.get(job.id).state is JobState.SUCCEEDED        # job lifecycle closed

    # experiment registry populated by the real route
    reg = ExperimentRegistry(str(tmp_path / "registry.db"))
    runs = reg.list()
    assert len(runs) == 1 and runs[0].status is RunStatus.SUCCEEDED
    assert runs[0].policy_version == 1 and runs[0].best_checkpoint
    assert runs[0].eval_result["final_kl"] == 0.02

    # policy published, checkpoint in the registry, artifacts + lineage, tagged traj
    assert PolicyRegistry(tmp_path / "policies").latest().version == 1
    areg = ArtifactRegistry(str(tmp_path / "artifacts.db"))
    assert {a.type for a in areg.list(run_id=runs[0].run_id)} == {
        ArtifactType.PROMPT_DATASET, ArtifactType.CHECKPOINT,
        ArtifactType.POLICY, ArtifactType.EVAL_REPORT}
    ts = TrajectoryStore(str(tmp_path / "trajectories.db"))
    assert ts.count() == 1 and ts.recent(1)[0].lineage.policy_id == "v1"


def test_gpu_slot_serializes_two_jobs(tmp_path):
    store, sch = _sched(tmp_path, gpu_slots=1)
    j1 = submit_training(store, CFG, platform_root=str(tmp_path), priority=1)
    j2 = submit_training(store, CFG, platform_root=str(tmp_path), priority=5)
    r1 = serve_pending(sch, runner=fake_runner)                # only 1 gpu slot
    assert len(r1) == 1 and r1[0]["job_id"] == j2.id           # higher priority first
    assert store.get(j1.id).state is JobState.PENDING          # j1 still queued
    r2 = serve_pending(sch, runner=fake_runner)
    assert r2[0]["job_id"] == j1.id
    assert store.get(j1.id).state is JobState.SUCCEEDED


def test_training_failure_marks_job_and_run_failed(tmp_path):
    store, sch = _sched(tmp_path)
    job = submit_training(store, CFG, platform_root=str(tmp_path))

    def boom(config, n, ns):
        raise RuntimeError("trainer exploded")

    results = serve_pending(sch, runner=boom)
    assert results[0]["status"] == "failed"
    assert store.get(job.id).state is JobState.DEAD_LETTER or \
        store.get(job.id).state is JobState.PENDING            # bounded retry then DL
    reg = ExperimentRegistry(str(tmp_path / "registry.db"))
    assert reg.list()[0].status is RunStatus.FAILED            # run marked FAILED


# --- API submit path -------------------------------------------------------
def test_api_submit_and_list_training_jobs(tmp_path):
    client = TestClient(create_app(tmp_path / "data", tmp_path / "runs"))
    r = client.post("/training-jobs", json={"config_path": CFG, "n_prompts": 8})
    assert r.status_code == 201
    body = r.json()
    assert body["kind"] == "train" and body["resource"] == "gpu" and body["state"] == "pending"
    assert len(client.get("/training-jobs").json()) == 1


def test_api_training_validation(tmp_path):
    client = TestClient(create_app(tmp_path / "data", tmp_path / "runs"))
    assert client.post("/training-jobs", json={}).status_code == 422
    assert client.post("/training-jobs", json={"config_path": "", }).status_code == 422
