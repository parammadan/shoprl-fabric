"""Integration tests: a training run threaded through the whole platform.

Uses PlatformRun directly with a fake checkpoint dir + fake eval data, so the
full platform path (preflight -> register -> checkpoint -> policy -> tag -> finish
-> artifacts+lineage) is exercised without torch or a model."""
import pytest

from shoprl.config import Config, RolloutConfig
from shoprl.platform import ArtifactRegistry, ArtifactType, PlatformRun, cost_estimate
from shoprl.platform.integration import PlatformRun as PR
from shoprl.platform.policy import staleness_report
from shoprl.platform.preflight import PreflightError
from shoprl.platform.registry import ExperimentRegistry, RunStatus
from shoprl.platform.traj_store import TrajectoryStore


def _adapter(tmp_path):
    d = tmp_path / "ckpt-out"
    d.mkdir(exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(b"\x00\x01weights")
    (d / "train_state.json").write_text('{"step": 3}')
    return d


def test_cost_estimate():
    assert cost_estimate(3600)["usd"] == 0.4 and cost_estimate(3600)["gpu_hours"] == 1.0
    assert cost_estimate(None) is None


def test_preflight_gate_blocks_bad_config(tmp_path):
    cfg = Config(rollout=RolloutConfig(engine="stub", num_samples=1))  # group of 1
    pr = PR(cfg, tmp_path / "plat")
    try:
        with pytest.raises(PreflightError):
            pr.preflight().raise_if_failed()
    finally:
        pr.close()


def test_full_run_flows_through_every_component(tmp_path):
    cfg = Config(rollout=RolloutConfig(engine="stub", num_samples=2))
    pr = PlatformRun(cfg, tmp_path / "plat")
    try:
        pr.preflight().raise_if_failed()              # 1. fail fast (passes)
        run = pr.start(n_prompts=8)                   # 2. register + dataset
        assert run.status is RunStatus.RUNNING
        assert run.dataset_version.startswith("cat")

        manifest = pr.register_checkpoint(_adapter(tmp_path))   # 3. checkpoint
        pv = pr.publish_policy(_adapter(tmp_path),              # 4. policy
                               metadata={"step": cfg.training.steps})
        assert pv.version == 1

        # 5. tag trajectories with the policy version
        for i in range(3):
            pr.tag_trajectory(f"prompt {i}", "ADD_TO_CART X", 0.5,
                              components={"total": 0.5}, prompt_id=f"P-{i}")

        done = pr.finish(RunStatus.SUCCEEDED,          # 6. finish
                         eval_result={"reward_gain": 0.01, "final_kl": 0.02},
                         best_checkpoint=manifest.ckpt_id,
                         cost_estimate=cost_estimate(120.0))
    finally:
        pr.close()

    # --- verify the run record captured everything ---
    reg = ExperimentRegistry(str((tmp_path / "plat") / "registry.db"))
    rec = reg.get(done.run_id)
    assert rec.status is RunStatus.SUCCEEDED
    assert rec.best_checkpoint == manifest.ckpt_id
    assert rec.policy_version == 1
    assert rec.eval_result["final_kl"] == 0.02
    assert rec.cost_estimate["usd"] is not None

    # --- artifacts + cross-artifact lineage ---
    areg = ArtifactRegistry(str((tmp_path / "plat") / "artifacts.db"))
    kinds = {a.type for a in areg.list(run_id=done.run_id)}
    assert kinds == {ArtifactType.PROMPT_DATASET, ArtifactType.CHECKPOINT,
                     ArtifactType.POLICY, ArtifactType.EVAL_REPORT}
    ev = next(a for a in areg.list(type=ArtifactType.EVAL_REPORT))
    anc_types = {a.type for a in areg.ancestors(ev.artifact_id)}
    assert ArtifactType.CHECKPOINT in anc_types            # eval -> checkpoint
    assert ArtifactType.PROMPT_DATASET in anc_types        # -> dataset (via ckpt)

    # --- trajectories tagged with the policy version, staleness computable ---
    ts = TrajectoryStore(str((tmp_path / "plat") / "trajectories.db"))
    rep = staleness_report(ts, current_version=1)
    assert rep["n"] == 3 and rep["on_policy_count"] == 3   # all tagged v1


def test_failure_marks_run_failed(tmp_path):
    cfg = Config(rollout=RolloutConfig(engine="stub", num_samples=2))
    pr = PlatformRun(cfg, tmp_path / "plat")
    try:
        pr.start(n_prompts=8)
        rec = pr.fail("boom")
        assert rec.status is RunStatus.FAILED
        assert rec.eval_result["error"] == "boom"
    finally:
        pr.close()


def test_platform_package_imports_cleanly():
    import importlib
    import shoprl.platform as p
    importlib.reload(p)                                    # no circular import
    assert hasattr(p, "PlatformRun")
