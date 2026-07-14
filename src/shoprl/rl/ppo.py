"""PPO adapter — the critic-based one.

How it differs from GRPO/RLOO — the baseline is LEARNED, not computed from the
group:
  GRPO/RLOO: baseline = a statistic of the sampled group's rewards (mean, or
             leave-one-out mean). Free, but needs a group of samples per prompt.
  PPO:       baseline = a value network V(s) trained to predict the reward.
             Advantage = reward - V(s). The critic is optimized alongside the
             policy (value loss = MSE(V, reward)).

Tradeoffs:
  + PPO doesn't need a group — it can learn from a single sample per prompt,
    because the critic provides the baseline.
  + A good critic can lower gradient variance more than a group mean.
  - You now train a SECOND network: more parameters, more compute/memory, and a
    new failure mode (if the critic is wrong, advantages are wrong -> instability).
    GRPO/RLOO deleted exactly this machinery, which is their whole appeal.

Simplification for this task: rewards are a single scalar per completion (no
per-token or intermediate rewards, no bootstrapping), so GAE collapses to
A = reward - V(state). We use a lightweight value head on the shared backbone's
last hidden state (standard actor-critic-with-shared-body), value = mean over
completion-token hidden states. The policy uses the same clipped-surrogate + KL
loss as the others (so the comparison isolates baseline mechanism + the critic).
"""
from __future__ import annotations

import statistics

import torch

from shoprl.grpo.logprobs import build_batch, mean_entropy, token_logprobs
from shoprl.grpo.loss import grpo_loss
from shoprl.rl.base import RLTrainer

_VALUE_COEF = 0.5


class PPOTrainer(RLTrainer):
    name = "ppo"

    def __init__(self, config, resume_from=None):
        super().__init__(config, resume_from)
        hidden = self.model.config.hidden_size
        dtype = next(self.model.parameters()).dtype
        # The critic: a scalar value head on the shared backbone.
        self.value_head = torch.nn.Linear(hidden, 1).to(self.device, dtype=dtype)
        # Train the value head alongside the LoRA policy params.
        self.optimizer.add_param_group({"params": list(self.value_head.parameters()),
                                        "lr": self.tr.lr})

    def optimize(self, completions, rewards_per_group) -> dict:
        flat_r = [r for g in rewards_per_group for r in g]
        R = torch.tensor(flat_r, dtype=torch.float32, device=self.device)  # [B]

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

        out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        logits = out.logits
        hidden = out.hidden_states[-1]                       # [B, T, H]
        logp_policy = token_logprobs(logits, input_ids)
        entropy = mean_entropy(logits, mask_shift)
        logp_old = logp_policy.detach()

        # Critic: per-sequence value = mean value over completion tokens.
        v_tok = self.value_head(hidden).squeeze(-1)          # [B, T]
        v_shift = v_tok[:, :-1]
        v_seq = (v_shift * mask_shift).sum(-1) / mask_shift.sum(-1).clamp(min=1.0)  # [B]

        # Advantage = reward - V(state), standardized across the batch (PPO norm).
        adv = R - v_seq.detach().float()
        adv = (adv - adv.mean()) / (adv.std() + 1e-4)

        # Policy: same clipped-surrogate + KL as GRPO/RLOO, with the critic advantage.
        policy_loss, stats = grpo_loss(logp_policy, logp_old, logp_ref, adv, mask_shift,
                                       clip_eps=self.tr.clip_eps, beta=self.tr.beta)
        # Critic: fit V to the observed reward.
        value_loss = torch.nn.functional.mse_loss(v_seq.float(), R)
        loss = policy_loss + _VALUE_COEF * value_loss

        # step both policy + value head
        self.policy_params_all = self.policy_params + list(self.value_head.parameters())
        if not torch.isfinite(loss):
            self.stability_failures += 1
            self.optimizer.zero_grad()
            gn = float("nan")
        else:
            self.optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(self.policy_params_all, self.tr.max_grad_norm))
            self.optimizer.step()

        return {
            "reward_mean": statistics.mean(flat_r),
            "reward_std": statistics.pstdev(flat_r) if len(flat_r) > 1 else 0.0,
            "kl": stats.mean_kl, "entropy": entropy, "grad_norm": gn,
            "loss": stats.loss, "value_loss": float(value_loss.detach()),
            "clip_frac": stats.clip_fraction, "ratio": stats.mean_ratio,
        }
