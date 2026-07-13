"""Platform integration — make the training entrypoint flow THROUGH the platform.

The platform components (preflight, experiment/prompt/checkpoint/policy/artifact
registries) were built and tested in isolation. This orchestrator threads a real
training run through them so they are load-bearing, not adjacent:

    preflight (fail fast)                          -> don't allocate a doomed run
    register run + materialize prompt dataset      -> reproducible, tracked
    [ train ]                                      -> the RLTrainer, unchanged
    register checkpoint (atomic, verified)         -> resumable, corruption-checked
    publish policy version                         -> on-policy lifecycle
    tag trajectories with the policy version       -> staleness computable
    record artifacts + finish run                  -> lineage + comparison

`PlatformRun` is deliberately decoupled from the trainer: it takes directories
and plain data, so it is fully testable without torch/a model (see tests). The
entrypoint (`shoprl.rl.run`) supplies the real trainer's outputs.
"""
from __future__ import annotations

from pathlib import Path

from shoprl.config import Config
from shoprl.platform.artifacts import (ArtifactRegistry, register_checkpoint,
                                        register_eval_report, register_policy,
                                        register_prompt_dataset)
from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.policy import PolicyRegistry
from shoprl.platform.preflight import run_preflight
from shoprl.platform.prompt_registry import PromptRegistry
from shoprl.platform.registry import ExperimentRegistry, RunStatus, record_from_config
from shoprl.platform.traj_store import TrajectoryStore
from shoprl.platform.trajectory import Lineage, Trajectory, TrajectoryStep


def cost_estimate(train_time_s: float | None, rate_usd_per_hr: float = 0.40) -> dict | None:
    """Rough, LABELLED cost estimate from measured train time (A10G spot ~$0.40/hr)."""
    if not train_time_s:
        return None
    hrs = train_time_s / 3600.0
    return {"gpu_hours": round(hrs, 4), "usd": round(hrs * rate_usd_per_hr, 4),
            "rate_usd_per_hr": rate_usd_per_hr, "note": "ESTIMATE"}


class PlatformRun:
    def __init__(self, config: Config, root: str | Path):
        self.cfg = config
        self.root = Path(root)
        self.exp = ExperimentRegistry(str(self.root / "registry.db"))
        self.prompts = PromptRegistry(self.root / "prompts")
        self.ckpts = CheckpointRegistry(self.root / "checkpoints")
        self.policies = PolicyRegistry(self.root / "policies")
        self.artifacts = ArtifactRegistry(str(self.root / "artifacts.db"))
        self.traj = TrajectoryStore(str(self.root / "trajectories.db"))
        self.run = None
        self.policy_version: int | None = None
        self._dataset_art = None
        self._ckpt_art = None

    # 1. fail fast --------------------------------------------------------
    def preflight(self, gpu_mem_gb: float | None = None, rollout_fn=None,
                  backward_fn=None):
        return run_preflight(self.cfg, gpu_mem_gb=gpu_mem_gb,
                             rollout_fn=rollout_fn, backward_fn=backward_fn)

    # 2. register + materialize data -------------------------------------
    def start(self, n_prompts: int = 64) -> "RunRecord":
        meta = self.prompts.materialize(
            catalog_size=self.cfg.training.catalog_size, n=n_prompts,
            seed=self.cfg.experiment.seed)
        self.run = record_from_config(self.cfg)
        self.run = self.run.model_copy(update={"dataset_version": meta.dataset_version})
        self.exp.save(self.run)
        self.exp.start(self.run.run_id)
        self._dataset_art = register_prompt_dataset(self.artifacts, meta,
                                                    run_id=self.run.run_id)
        return self.exp.get(self.run.run_id)

    # 3. checkpoint (atomic, verified) -----------------------------------
    def register_checkpoint(self, checkpoint_dir: str | Path, step: int | None = None):
        step = self.cfg.training.steps if step is None else step
        manifest = self.ckpts.save(checkpoint_dir, step=step,
                                   policy_id=f"step-{step}")
        parents = [self._dataset_art.artifact_id] if self._dataset_art else None
        self._ckpt_art = register_checkpoint(self.artifacts, manifest,
                                             run_id=self.run.run_id, parents=parents)
        return manifest

    # 4. publish policy version ------------------------------------------
    def publish_policy(self, adapter_dir: str | Path, metadata: dict | None = None):
        pv = self.policies.publish(adapter_dir, metadata=metadata)
        self.policy_version = pv.version
        parents = [self._ckpt_art.artifact_id] if self._ckpt_art else None
        register_policy(self.artifacts, pv, run_id=self.run.run_id, parents=parents)
        # reflect the policy version on the run record
        self.run = self.exp.get(self.run.run_id).model_copy(
            update={"policy_version": pv.version})
        self.exp.save(self.run)
        return pv

    # 5. tag trajectories with the policy version ------------------------
    def tag_trajectory(self, prompt: str, response: str, reward: float, *,
                       components: dict | None = None, advantage: float | None = None,
                       prompt_id: str | None = None, generated_version: int | None = None,
                       max_staleness: int = 0, stale_mode: str = "warn") -> Trajectory:
        """Persist a trajectory tagged with the policy version that GENERATED it
        (defaults to the current published version — i.e. on-policy). Runs the
        staleness gate so a lagging rollout is flagged/rejected."""
        from shoprl.platform.policy import staleness_gate
        gen_v = self.policy_version if generated_version is None else generated_version
        gen_v = 0 if gen_v is None else gen_v
        s, warn = staleness_gate(self.policy_version or gen_v, gen_v,
                                 max_staleness=max_staleness, mode=stale_mode)
        traj = Trajectory(
            kind="single_turn", prompt=prompt, reward=reward,
            steps=[TrajectoryStep(index=0, action=response, reward=reward)],
            lineage=Lineage(policy_id=f"v{gen_v}", job_id=self.run.run_id,
                            prompt_id=prompt_id, seed=self.cfg.experiment.seed),
            meta={"reward_components": components, "advantage": advantage,
                  "staleness": s, "staleness_warning": warn})
        return self.traj.put(traj)

    # 6. finish + record eval report -------------------------------------
    def finish(self, status: RunStatus, *, eval_result: dict | None = None,
               best_checkpoint: str | None = None, cost_estimate: dict | None = None):
        self.run = self.exp.finish(self.run.run_id, status, eval_result=eval_result,
                                   best_checkpoint=best_checkpoint,
                                   cost_estimate=cost_estimate)
        if status is RunStatus.SUCCEEDED and eval_result is not None:
            parents = [self._ckpt_art.artifact_id] if self._ckpt_art else None
            register_eval_report(self.artifacts, f"eval-{self.run.run_id}",
                                 run_id=self.run.run_id, parents=parents,
                                 metadata=eval_result)
        return self.run

    def fail(self, error: str):
        if self.run is not None:
            self.run = self.exp.finish(self.run.run_id, RunStatus.FAILED,
                                       eval_result={"error": error})
        return self.run

    def close(self) -> None:
        self.exp.close()
        self.traj.close()
        self.artifacts.close()
