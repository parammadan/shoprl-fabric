"""Artifact registry tests: unified metadata + cross-artifact lineage."""
import pytest
from fastapi.testclient import TestClient

from shoprl.platform import ArtifactRegistry, ArtifactType
from shoprl.platform.api import create_app
from shoprl.platform.artifacts import (ArtifactNotFound, register_checkpoint,
                                       register_prompt_dataset)
from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.prompt_registry import PromptRegistry


# --- core registry ---------------------------------------------------------
def test_register_get_list_by_type_and_run(tmp_path):
    reg = ArtifactRegistry(tmp_path / "a.db")
    reg.register(ArtifactType.CHECKPOINT, "step-000030-abc", run_id="R1")
    reg.register(ArtifactType.PROMPT_DATASET, "cat300-n64-seed0", run_id="R1")
    reg.register(ArtifactType.CHECKPOINT, "step-000060-def", run_id="R2")
    assert len(reg.list()) == 3
    assert len(reg.list(type=ArtifactType.CHECKPOINT)) == 2
    assert len(reg.list(run_id="R1")) == 2


def test_missing_raises(tmp_path):
    with pytest.raises(ArtifactNotFound):
        ArtifactRegistry(tmp_path / "a.db").get("nope")


def test_by_ref_returns_latest(tmp_path):
    reg = ArtifactRegistry(tmp_path / "a.db")
    reg.register(ArtifactType.POLICY, "v1")
    a = reg.by_ref(ArtifactType.POLICY, "v1")
    assert a is not None and a.ref == "v1"
    assert reg.by_ref(ArtifactType.POLICY, "v9") is None


def test_survives_restart(tmp_path):
    path = tmp_path / "a.db"
    aid = ArtifactRegistry(path).register(ArtifactType.BENCHMARK, "vllm-vs-hf").artifact_id
    assert ArtifactRegistry(path).get(aid).ref == "vllm-vs-hf"


# --- cross-artifact lineage ------------------------------------------------
def test_lineage_dag_ancestors_and_children(tmp_path):
    reg = ArtifactRegistry(tmp_path / "a.db")
    ds = reg.register(ArtifactType.PROMPT_DATASET, "cat300-n64-seed0", run_id="R1")
    pol = reg.register(ArtifactType.POLICY, "v3", run_id="R1")
    # a checkpoint derived from BOTH the dataset and the policy
    ck = reg.register(ArtifactType.CHECKPOINT, "step-000030-abc", run_id="R1",
                      parents=[ds.artifact_id, pol.artifact_id])
    # an eval report derived from the checkpoint
    ev = reg.register(ArtifactType.EVAL_REPORT, "eval-R1", run_id="R1",
                      parents=[ck.artifact_id])

    anc = {a.ref for a in reg.ancestors(ev.artifact_id)}
    assert anc == {"step-000030-abc", "cat300-n64-seed0", "v3"}  # full DAG upward
    assert [c.ref for c in reg.children(ck.artifact_id)] == ["eval-R1"]

    lin = reg.lineage(ev.artifact_id)
    assert lin["artifact"]["ref"] == "eval-R1"
    assert len(lin["ancestors"]) == 3


# --- thin registrars reuse the specialized registries ----------------------
def test_registrars_index_real_manifests(tmp_path):
    areg = ArtifactRegistry(tmp_path / "a.db")
    # a real checkpoint + a real prompt dataset from their own registries
    creg = CheckpointRegistry(tmp_path / "ck")
    src = tmp_path / "src"; src.mkdir(); (src / "w").write_bytes(b"weights")
    manifest = creg.save(src, step=30, policy_id="v3")
    preg = PromptRegistry(tmp_path / "pr")
    meta = preg.materialize(catalog_size=120, n=8, seed=0)

    a_ck = register_checkpoint(areg, manifest, run_id="R1")
    a_ds = register_prompt_dataset(areg, meta, run_id="R1")
    assert a_ck.type is ArtifactType.CHECKPOINT and a_ck.ref == manifest.ckpt_id
    assert a_ck.metadata["step"] == 30
    assert a_ds.type is ArtifactType.PROMPT_DATASET and a_ds.hash == meta.hash


# --- API -------------------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "data", tmp_path / "runs"))


def test_api_artifact_crud_and_lineage(client):
    ds = client.post("/artifacts", json={"type": "prompt_dataset",
                                         "ref": "cat300-n64-seed0", "run_id": "R1"}).json()
    ck = client.post("/artifacts", json={"type": "checkpoint", "ref": "step-30",
                                         "run_id": "R1",
                                         "parents": [ds["artifact_id"]]}).json()
    assert client.get(f"/artifacts/{ck['artifact_id']}").json()["ref"] == "step-30"
    lin = client.get(f"/artifacts/{ck['artifact_id']}/lineage").json()
    assert [a["ref"] for a in lin["ancestors"]] == ["cat300-n64-seed0"]
    assert len(client.get("/artifacts", params={"run_id": "R1"}).json()) == 2
    assert len(client.get("/artifacts", params={"type": "checkpoint"}).json()) == 1


def test_api_validation_and_404(client):
    assert client.post("/artifacts", json={"type": "checkpoint"}).status_code == 422
    assert client.post("/artifacts", json={"type": "bogus", "ref": "x"}).status_code == 422
    assert client.get("/artifacts/ghost").status_code == 404
    assert client.get("/artifacts/ghost/lineage").status_code == 404
