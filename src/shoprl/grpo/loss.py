"""The GRPO loss: clipped policy-gradient surrogate + KL penalty.

Ties the pieces together. Per completion token t of sample i:

    ratio   = exp(logp_policy - logp_old)          # vs behavior policy
    surr    = min(ratio * A_i, clip(ratio, 1-e, 1+e) * A_i)   # PPO trust region
    kl      = token_kl(logp_policy, logp_ref)       # k3, vs frozen reference
    per_tok = surr - beta * kl

    loss    = - sum(mask * per_tok) / sum(mask)     # mean over completion tokens

Design choices / WHY:
  - A_i is per-SEQUENCE (outcome reward), broadcast across that sample's tokens.
    Advantages are detached: rewards are constants w.r.t. the policy.
  - ratio uses logp_old (the policy weights at sampling time), enabling multiple
    optimizer steps per rollout batch. On the first step logp_old == logp_policy
    so ratio == 1.
  - clip + elementwise min = PPO clipping; caps how far one batch can move the
    policy (works for both signs of A because min picks the pessimistic branch).
  - mask zeroes prompt + padding tokens: we only train on generated tokens.
  - We return metrics too (mean KL, ratio, clip fraction) for observability.

Length-bias note: dividing by total completion-token count (not per-sequence
length then averaging) is the simple, common choice; Dr.GRPO argues about the
bias here. We keep the straightforward token-mean for the reference impl.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from shoprl.grpo.kl import token_kl


@dataclass
class GRPOStats:
    loss: float
    mean_kl: float
    mean_ratio: float
    clip_fraction: float
    mean_advantage: float


def grpo_loss(
    logp_policy: torch.Tensor,  # [B, T]  current policy log-probs (grad)
    logp_old: torch.Tensor,     # [B, T]  behavior policy at sampling time (detached)
    logp_ref: torch.Tensor,     # [B, T]  frozen reference (detached)
    advantages: torch.Tensor,   # [B]     per-sequence, group-standardized (detached)
    mask: torch.Tensor,         # [B, T]  1 = completion token, 0 = prompt/pad
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> tuple[torch.Tensor, GRPOStats]:
    logp_old = logp_old.detach()
    advantages = advantages.detach()
    mask = mask.to(logp_policy.dtype)

    ratio = torch.exp(logp_policy - logp_old)          # [B, T]
    adv = advantages.unsqueeze(1)                       # [B, 1] -> broadcast

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(unclipped, clipped)           # pessimistic branch

    kl = token_kl(logp_policy, logp_ref)                # [B, T], >= 0
    per_token = surrogate - beta * kl

    denom = mask.sum().clamp(min=1.0)
    loss = -(per_token * mask).sum() / denom

    # --- metrics (no grad) ---
    with torch.no_grad():
        m = mask
        mean_kl = (kl * m).sum() / denom
        mean_ratio = (ratio * m).sum() / denom
        # clip active where the clipped branch was chosen (differs from unclipped)
        clipped_active = (unclipped != clipped).to(logp_policy.dtype)
        clip_fraction = (clipped_active * m).sum() / denom
        mean_adv = advantages.mean()

    stats = GRPOStats(
        loss=float(loss.detach()),
        mean_kl=float(mean_kl),
        mean_ratio=float(mean_ratio),
        clip_fraction=float(clip_fraction),
        mean_advantage=float(mean_adv),
    )
    return loss, stats
