"""End-to-end pipeline: the six pillars wired into one runnable flow.

    enqueue rollout jobs  (Pillar 1 state machine + persistence)
      -> local worker processes claim + run them  (Pillar 2)
        -> each produces reward-scored, lineage-tagged trajectories  (Pillar 3)
          -> an optimize job checkpoints the step atomically  (Pillar 4)
            -> a deliberately-injected OOM is recovered automatically  (Pillar 5)
              -> everything persists to disk for the dashboard  (Pillar 6)

This is real platform plumbing end to end. To run on a laptop with no GPU and
no model download it uses the deterministic StubRolloutEngine for text
generation and the project's REAL rule-based reward (compute_reward against the
synthetic catalog ground truth) — so the reward numbers are genuine, only the
token generation is a stub. There is no policy-gradient update here, so this
pipeline does NOT produce KL/entropy; those come from the real RL trainer's
metrics.jsonl, which the dashboard shows alongside this operational state.

Honest scope: workers are local processes on one machine; the OOM is triggered
via SimulatedOOM and labelled; the recovery (batch shrink + checkpoint restore
+ requeue) is real.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from shoprl.data import generate_catalog, generate_prompts
from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.failures import RecoveryController, SimulatedOOM
from shoprl.platform.jobs import JobState
from shoprl.platform.store import JobStore
from shoprl.platform.traj_store import TrajectoryStore
from shoprl.platform.trajectory import Lineage, Trajectory
from shoprl.platform.workers import run_local_pool
from shoprl.reward.composite import compute_reward
from shoprl.reward.functions import RewardContext
from shoprl.rollout.stub import StubRolloutEngine

_CATALOG_N = 120


@dataclass
class PipelineConfig:
    steps: int = 3
    prompts_per_step: int = 6
    num_samples: int = 4
    seed: int = 0
    n_workers: int = 3
    oom_at_step: int | None = 1        # inject one OOM to exercise recovery
    recovery_enabled: bool = True


@dataclass
class Paths:
    root: Path
    jobs_db: str = field(init=False)
    traj_db: str = field(init=False)
    ckpt_root: str = field(init=False)
    events_path: str = field(init=False)
    metrics_path: str = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_db = str(self.root / "jobs.db")
        self.traj_db = str(self.root / "trajectories.db")
        self.ckpt_root = str(self.root / "checkpoints")
        self.events_path = str(self.root / "recovery_events.jsonl")
        self.metrics_path = str(self.root / "pipeline_metrics.jsonl")


# --- the rollout handler run inside each worker PROCESS --------------------
def rollout_handler(job) -> dict:
    """Runs in a spawned worker process. Rebuilds deterministic state from the
    payload (no shared globals), generates + scores completions with the REAL
    reward, and persists lineage-tagged trajectories."""
    p = job.payload
    catalog = generate_catalog(n=_CATALOG_N, seed=p["seed"])
    cat_by_sku = {x.sku: x for x in catalog}
    ctx = RewardContext(catalog=cat_by_sku, constraints=p["constraints"])
    engine = StubRolloutEngine(seed=p["seed"])
    group = engine.generate([p["prompt"]], num_samples=p["num_samples"],
                            seed=p["seed"] + p["step"])[0]
    ts = TrajectoryStore(p["traj_db"])
    rewards = []
    try:
        for comp in group.completions:
            r = compute_reward(comp.text, ctx).total
            rewards.append(r)
            ts.put(Trajectory.from_completion(
                comp, reward=r,
                lineage=Lineage(policy_id=p["policy_id"], job_id=job.id,
                                prompt_id=p["prompt_id"], seed=p["seed"]),
                prompt_id=p["prompt_id"]))
    finally:
        ts.close()
    return {"prompt_id": p["prompt_id"],
            "reward_mean": statistics.fmean(rewards), "n": len(rewards)}


class Pipeline:
    def __init__(self, cfg: PipelineConfig, root: str | Path):
        self.cfg = cfg
        self.paths = Paths(Path(root))
        self.store = JobStore(self.paths.jobs_db)
        self.traj = TrajectoryStore(self.paths.traj_db)
        self.registry = CheckpointRegistry(self.paths.ckpt_root)
        self.recovery = RecoveryController(
            self.store, registry=self.registry,
            enabled=cfg.recovery_enabled, events_path=self.paths.events_path)
        catalog = generate_catalog(n=_CATALOG_N, seed=cfg.seed)
        self.prompts = generate_prompts(catalog, n=cfg.prompts_per_step, seed=cfg.seed)

    def _enqueue_rollouts(self, step: int, policy_id: str) -> None:
        for ex in self.prompts:
            self.store.create("rollout", {
                "prompt_id": ex.prompt_id, "prompt": ex.prompt,
                "constraints": ex.constraints, "num_samples": self.cfg.num_samples,
                "seed": self.cfg.seed, "step": step, "policy_id": policy_id,
                "traj_db": self.paths.traj_db})

    def _run_optimize(self, step: int, policy_id: str) -> dict:
        """Aggregate the step's trajectory rewards and atomically checkpoint.
        On the configured step, deliberately OOM to exercise recovery, then let
        the retried job succeed."""
        job = self.store.create("optimize", {
            "step": step, "policy_id": policy_id,
            "microbatch_size": 8, "grad_accum_steps": 1})
        self.store.claim(kinds=["optimize"])

        oom_here = self.cfg.oom_at_step == step
        if oom_here:
            # Pillar 5: this raises, the controller shrinks the batch + selects
            # a checkpoint to restore + requeues; then we re-run the retry.
            self.recovery.handle(self.store.get(job.id), SimulatedOOM("cuda oom"))
            self.store.claim(kinds=["optimize"])       # pick the requeued job up

        rewards = [t.reward for t in self.traj.by_policy(policy_id)
                   if t.reward is not None]
        reward_mean = statistics.fmean(rewards) if rewards else 0.0
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        job = self.store.get(job.id)                   # reflect any shrink
        self.registry.save_state(
            {"step": step, "policy_id": policy_id, "reward_mean": reward_mean,
             "microbatch_size": job.payload["microbatch_size"]},
            step=step, policy_id=policy_id,
            metadata={"reward_mean": reward_mean, "n_trajectories": len(rewards)})
        self.store.complete(job.id)

        row = {"step": step, "policy_id": policy_id, "reward_mean": reward_mean,
               "reward_std": reward_std, "n_trajectories": len(rewards),
               "recovered_oom": oom_here}
        with open(self.paths.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def run(self) -> dict:
        open(self.paths.metrics_path, "w").close()
        per_step = []
        for step in range(self.cfg.steps):
            policy_id = f"step-{step:03d}"
            self._enqueue_rollouts(step, policy_id)
            run_local_pool(self.paths.jobs_db, n_workers=self.cfg.n_workers,
                           handlers={"rollout": "shoprl.platform.pipeline:rollout_handler"})
            per_step.append(self._run_optimize(step, policy_id))

        summary = {
            "steps": self.cfg.steps,
            "job_counts": self.store.counts(),
            "trajectories": self.traj.count(),
            "checkpoints": [m.ckpt_id for m in self.registry.list()],
            "recovery_events": _count_lines(self.paths.events_path),
            "per_step": per_step,
            "paths": self.paths.__dict__ | {"root": str(self.paths.root)},
        }
        return summary

    def close(self) -> None:
        self.store.close()
        self.traj.close()


def _count_lines(path: str) -> int:
    p = Path(path)
    return sum(1 for _ in p.open()) if p.exists() else 0


def run_pipeline(root: str | Path, cfg: PipelineConfig | None = None) -> dict:
    pipe = Pipeline(cfg or PipelineConfig(), root)
    try:
        return pipe.run()
    finally:
        pipe.close()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the ShopRL platform pipeline end to end")
    ap.add_argument("--root", default="runs/pipeline")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompts-per-step", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-recovery", action="store_true",
                    help="disable OOM recovery (plain bounded retry)")
    args = ap.parse_args()
    cfg = PipelineConfig(steps=args.steps, prompts_per_step=args.prompts_per_step,
                         n_workers=args.workers, recovery_enabled=not args.no_recovery)
    summary = run_pipeline(args.root, cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
