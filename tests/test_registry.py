"""Experiment registry tests: run records, provenance, comparison, API."""
import pytest
from fastapi.testclient import TestClient

from shoprl.config import Config, TrainingConfig
from shoprl.platform.api import create_app
from shoprl.platform.registry import (ExperimentRegistry, RunNotFound,
                                       RunStatus, config_hash, dataset_version,
                                       record_from_config, register_rl_result,
                                       reward_version)


# --- provenance helpers ----------------------------------------------------
def test_config_hash_is_stable_and_sensitive():
    a = Config()
    b = Config()
    c = Config(training=TrainingConfig(lr=9e-9))
    assert config_hash(a) == config_hash(b)          # same config -> same hash
    assert config_hash(a) != config_hash(c)          # a change moves the hash


def test_dataset_and_reward_versions():
    cfg = Config()
    assert dataset_version(cfg) == f"catalog{cfg.training.catalog_size}-seed{cfg.experiment.seed}"
    assert reward_version(cfg).startswith("rw-")


def test_record_from_config_populates_provenance():
    rec = record_from_config(Config())
    assert rec.algorithm == "grpo" and rec.model.startswith("Qwen")
    assert rec.config_hash and rec.dataset_version and rec.reward_version
    assert rec.status is RunStatus.CREATED


# --- persistence + lifecycle ----------------------------------------------
def test_save_get_list_and_filter(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    reg.save(record_from_config(Config()))
    reg.save(record_from_config(Config()).model_copy(update={"algorithm": "rloo"}))
    assert len(reg.list()) == 2
    assert len(reg.list(algorithm="rloo")) == 1


def test_missing_raises(tmp_path):
    with pytest.raises(RunNotFound):
        ExperimentRegistry(tmp_path / "r.db").get("nope")


def test_start_finish_lifecycle(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    rec = reg.save(record_from_config(Config()))
    reg.start(rec.run_id, now=100.0)
    done = reg.finish(rec.run_id, RunStatus.SUCCEEDED, now=200.0,
                      eval_result={"final_kl": 0.015, "reward_gain": 0.002},
                      best_checkpoint="step-000030-abc",
                      cost_estimate={"usd": 0.3})
    assert done.status is RunStatus.SUCCEEDED
    assert done.started_at == 100.0 and done.ended_at == 200.0
    assert done.eval_result["final_kl"] == 0.015
    assert done.best_checkpoint == "step-000030-abc"


def test_survives_restart(tmp_path):
    path = tmp_path / "r.db"
    rid = ExperimentRegistry(path).save(record_from_config(Config())).run_id
    assert ExperimentRegistry(path).get(rid).run_id == rid


# --- comparison ------------------------------------------------------------
def test_compare_flags_comparable_runs(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    ids = []
    for algo, kl in [("grpo", 0.58), ("rloo", 0.015), ("ppo", 6.78)]:
        rec = record_from_config(Config()).model_copy(update={"algorithm": algo})
        reg.save(rec)
        reg.finish(rec.run_id, RunStatus.SUCCEEDED,
                   eval_result={"final_kl": kl, "reward_gain": 0.0})
        ids.append(rec.run_id)
    cmp = reg.compare(ids)
    assert cmp["comparable"] is True                 # same dataset + reward version
    kls = {r["algorithm"]: r["final_kl"] for r in cmp["rows"]}
    assert kls == {"grpo": 0.58, "rloo": 0.015, "ppo": 6.78}


def test_compare_detects_incomparable(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    a = reg.save(record_from_config(Config()))
    b = reg.save(record_from_config(Config(training=TrainingConfig(catalog_size=999))))
    assert reg.compare([a.run_id, b.run_id])["comparable"] is False  # dataset differs


def test_register_rl_result_imports_real_fields(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.db")
    result = {"algorithm": "rloo", "reward_gain": 0.002, "final_kl": 0.015,
              "max_kl": 0.22, "held_out_before": {"reward_mean": 0.85},
              "held_out_after": {"reward_mean": 0.852}, "stability_failures": 0}
    rec = register_rl_result(reg, result, cfg=Config(), cost_estimate={"usd": 0.3})
    assert rec.status is RunStatus.SUCCEEDED
    assert rec.eval_result["final_kl"] == 0.015
    assert rec.eval_result["reward_after"] == 0.852


# --- through the API -------------------------------------------------------
@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "data", tmp_path / "runs"))


def _reg_body(**over):
    b = {"algorithm": "grpo", "model": "Qwen/Qwen3-0.6B", "config_hash": "abc123",
         "dataset_version": "catalog300-seed0", "reward_version": "rw-deadbeef"}
    b.update(over)
    return b


def test_api_run_crud_and_lifecycle(client):
    r = client.post("/runs", json=_reg_body())
    assert r.status_code == 201
    rid = r.json()["run_id"]
    assert client.get(f"/runs/{rid}").json()["algorithm"] == "grpo"
    assert client.post(f"/runs/{rid}/start").json()["status"] == "running"
    fin = client.post(f"/runs/{rid}/finish",
                      json={"status": "succeeded", "eval_result": {"final_kl": 0.02}})
    assert fin.json()["status"] == "succeeded" and fin.json()["ended_at"]
    assert len(client.get("/runs").json()) == 1


def test_api_run_validation_and_404(client):
    assert client.post("/runs", json={"algorithm": "grpo"}).status_code == 422
    assert client.get("/runs/ghost").status_code == 404
    assert client.post("/runs/ghost/start").status_code == 404


def test_api_compare(client):
    ids = []
    for algo in ("grpo", "rloo"):
        rid = client.post("/runs", json=_reg_body(algorithm=algo)).json()["run_id"]
        client.post(f"/runs/{rid}/finish",
                    json={"status": "succeeded", "eval_result": {"final_kl": 0.1}})
        ids.append(rid)
    body = client.get("/runs/compare", params={"ids": ",".join(ids)}).json()
    assert body["comparable"] is True and len(body["rows"]) == 2
    assert client.get("/runs/compare", params={"ids": ""}).status_code == 400
