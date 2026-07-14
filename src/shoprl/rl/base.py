"""Common RL trainer interface + the shared training loop.

The interface the user asked for:
  - generate_rollouts(step)   -> sample a group of completions per prompt (SHARED)
  - calculate_rewards(...)     -> verifiable composite reward per completion (SHARED)
  - optimize(...)              -> the learner step; THIS is where algorithms differ
  - save_checkpoint(step)      -> persist the LoRA adapter (SHARED)

Everything except `optimize` is identical across GRPO/RLOO/PPO, so a fair
comparison isolates exactly the advantage-estimation mechanism. The shared loop
also records the comparison metrics (KL, entropy, grad-norm, tokens/sec, peak
GPU memory, reward variance, stability failures).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from abc import ABC, abstractmethod

import torch

from shoprl.config import Config
from shoprl.data import generate_catalog, generate_prompts
from shoprl.data.catalog import catalog_index
from shoprl.grpo.logprobs import build_batch, mean_entropy, token_logprobs
from shoprl.grpo.loss import grpo_loss
from shoprl.grpo.trainer import build_policy, save_checkpoint
from shoprl.reward import RewardContext, compute_reward
from shoprl.rollout.base import Completion
from shoprl.rollout.hf import HFRolloutEngine
from shoprl.task import build_shortlist, build_task_prompt


class RLTrainer(ABC):
    name = "rl"

    def __init__(self, config: Config):
        self.config = config
        self.tr = config.training
        self.model, self.tokenizer, self.device = build_policy(config)
        self.engine = HFRolloutEngine(config, model=self.model, tokenizer=self.tokenizer)
        self.catalog = generate_catalog(n=self.tr.catalog_size, seed=config.experiment.seed)
        self.idx = catalog_index(self.catalog)
        self.examples = generate_prompts(self.catalog, n=64, seed=config.experiment.seed)
        self.policy_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.policy_params, lr=self.tr.lr)
        self.stability_failures = 0  # NaN/inf losses skipped

    # --- SHARED: rollout -------------------------------------------------
    def generate_rollouts(self, step: int):
        tr = self.tr
        picks = [self.examples[(step * tr.prompts_per_step + j) % len(self.examples)]
                 for j in range(tr.prompts_per_step)]
        task_prompts, contexts = [], []
        for ex in picks:
            sl = build_shortlist(ex, self.catalog, k=tr.shortlist, seed=self.config.experiment.seed)
            task_prompts.append(build_task_prompt(ex, self.idx, sl))
            contexts.append(RewardContext(catalog=self.idx, constraints=ex.constraints))
        # rollout in eval mode + checkpointing OFF (the bug fix)
        self.model.gradient_checkpointing_disable()
        self.model.eval()
        groups = self.engine.generate(task_prompts, self.config.rollout.num_samples,
                                      seed=self.config.experiment.seed + step + 1)
        return groups, contexts

    # --- SHARED: reward --------------------------------------------------
    def calculate_rewards(self, groups, contexts):
        completions: list[Completion] = []
        rewards_per_group: list[list[float]] = []
        breakdowns = []
        for ctx, group in zip(contexts, groups):
            grp = []
            for comp in group.completions:
                r = compute_reward(comp.text, ctx,
                                   weights=self.config.rewards.weights,
                                   hallucination_penalty=self.config.rewards.hallucination_penalty)
                grp.append(r.total)
                breakdowns.append(r)
                completions.append(comp)
            rewards_per_group.append(grp)
        return completions, rewards_per_group, breakdowns

    @staticmethod
    def _reward_components(breakdowns) -> dict:
        """Per-component reward means + hallucination rate, for the dashboard's
        component/hallucination panels (all algorithms)."""
        n = len(breakdowns)
        keys = ("budget", "groundedness", "coverage", "quality_format", "quality_comparison")
        out = {f"reward_{k}": sum(getattr(b, k) for b in breakdowns) / n for k in keys}
        out["hallucination_rate"] = sum(b.hallucinated for b in breakdowns) / n
        return out

    # --- ALGORITHM-SPECIFIC ---------------------------------------------
    @abstractmethod
    def optimize(self, completions, rewards_per_group) -> dict:
        """Run the learner step; return a metrics dict for this step."""

    # --- shared helpers used by the critic-free algorithms ---------------
    def _forward_logps(self, completions):
        """Build the batch and return (logp_policy[grad], logp_ref[detached],
        mask, entropy, logits) with checkpointing ON + train mode."""
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.train()
        input_ids, attn, cmask = build_batch(completions, self.tokenizer.pad_token_id, self.device)
        mask_shift = cmask[:, 1:]
        with torch.no_grad():
            with self.model.disable_adapter():
                ref_logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                logp_ref = token_logprobs(ref_logits, input_ids)
                del ref_logits
        logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        logp_policy = token_logprobs(logits, input_ids)
        entropy = mean_entropy(logits, mask_shift)
        return logp_policy, logp_ref, mask_shift, entropy, input_ids, attn

    def _critic_free_step(self, completions, rewards_per_group, advantages_nested) -> dict:
        """The shared learner step for GRPO & RLOO: same clipped-surrogate + KL
        loss; the ONLY difference between the two is `advantages_nested`."""
        flat_adv = [a for g in advantages_nested for a in g]
        advantages = torch.tensor(flat_adv, dtype=torch.float32, device=self.device)
        logp_policy, logp_ref, mask, entropy, _ids, _attn = self._forward_logps(completions)
        logp_old = logp_policy.detach()  # single update/batch -> ratio 1
        loss, stats = grpo_loss(logp_policy, logp_old, logp_ref, advantages, mask,
                                clip_eps=self.tr.clip_eps, beta=self.tr.beta)
        gn = self._step(loss)
        flat_r = [r for g in rewards_per_group for r in g]
        return {
            "reward_mean": statistics.mean(flat_r),
            "reward_std": statistics.pstdev(flat_r) if len(flat_r) > 1 else 0.0,
            "kl": stats.mean_kl, "entropy": entropy, "grad_norm": gn,
            "loss": stats.loss, "clip_frac": stats.clip_fraction, "ratio": stats.mean_ratio,
        }

    def _step(self, loss):
        """Backward + clip + step, with a NaN/inf guard (stability tracking)."""
        if not torch.isfinite(loss):
            self.stability_failures += 1
            self.optimizer.zero_grad()
            return float("nan")
        self.optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.policy_params, self.tr.max_grad_norm)
        self.optimizer.step()
        return float(gn)

    def save_checkpoint(self, step: int) -> str:
        return save_checkpoint(self.model, self.config, step)

    # --- SHARED: the loop ------------------------------------------------
    def train(self) -> list[dict]:
        metrics = []
        n_tokens = 0
        # metrics.jsonl (one row per step, appended live) — same schema the
        # legacy trainer wrote, so the dashboard + alerts work for all algorithms.
        run_dir = os.path.join("runs", self.config.experiment.name)
        os.makedirs(run_dir, exist_ok=True)
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        open(self.metrics_path, "w").close()
        t0 = time.time()
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(self.tr.steps):
            groups, contexts = self.generate_rollouts(step)
            completions, rewards_per_group, breakdowns = self.calculate_rewards(groups, contexts)
            n_tokens += sum(len(c.completion_token_ids) for c in completions)
            m = self.optimize(completions, rewards_per_group)
            m.update(self._reward_components(breakdowns))
            m["step"] = step
            # REAL measured GPU memory polled from the device this step (not an
            # estimate). Only present when training on CUDA; omitted on CPU/MPS
            # so the dashboard shows it as absent rather than faking a number.
            if self.device == "cuda":
                m["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
                m["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
                m["gpu_mem_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            metrics.append(m)
            with open(self.metrics_path, "a") as f:
                f.write(json.dumps(m) + "\n")
            print(f"[{self.name}] step {step:>2} | reward {m['reward_mean']:+.3f}"
                  f"±{m['reward_std']:.3f} | kl {m['kl']:.4f} | entropy {m['entropy']:.3f}"
                  f" | grad_norm {m['grad_norm']:.3f}")
            # NOTE: no mid-training save here — the CheckpointRegistry is the sole
            # authoritative checkpoint writer (final checkpoint is registered by
            # run_through_platform after train()). Periodic spot-interruption
            # resume points are a documented scale-out concern, not a second
            # writer that bypasses the registry.
        elapsed = time.time() - t0
        self.train_time_s = elapsed
        self.tokens_per_sec = n_tokens / elapsed if elapsed > 0 else 0.0
        self.peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9
                            if self.device == "cuda" else float("nan"))
        return metrics
