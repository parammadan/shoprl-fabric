"""Bridge from rollout token ids to the per-token log-probs the loss needs.

The learner must know, for each sampled completion token, its log-prob under the
current policy (with grad) and the frozen reference (no grad). We get that by a
forward pass over [prompt + completion] and reading off the log-prob of the token
that was actually sampled at each position.

Memory note (this matters on 8GB): Qwen3's vocab is ~152k, so a full
log_softmax over [B, T, V] is ~0.5GB per copy. We avoid it two ways:
  - token log-prob via the logsumexp identity:
        log p(x_t) = logits[x_t] - logsumexp(logits)
    which needs only [B, T] reductions, no [B, T, V] softmax tensor.
  - entropy (logging only) computed on JUST the completion-token positions
    (a compact [N, V]), never the whole batch-time grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from shoprl.rollout.base import Completion


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of each *next* token, aligned to input_ids[:, 1:].

    logits[:, t] predicts input_ids[:, t+1], so we compare logits[:, :-1] to
    input_ids[:, 1:]. Returns [B, T-1]. Uses the logsumexp identity to skip the
    full-vocab softmax.
    """
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    gathered = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    logZ = torch.logsumexp(logits, dim=-1)                          # [B, T-1]
    return gathered - logZ


@torch.no_grad()
def mean_entropy(logits: torch.Tensor, mask_shift: torch.Tensor) -> float:
    """Mean predictive entropy over completion tokens only (for logging).

    mask_shift is the completion mask aligned to logits[:, :-1] (i.e. mask[:, 1:]).
    We gather only those positions so the full-vocab softmax is [N, V], tiny.
    """
    sel = logits[:, :-1, :][mask_shift.bool()]      # [N, V]
    if sel.numel() == 0:
        return 0.0
    logp = F.log_softmax(sel.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)              # [N]
    return float(ent.mean())


def build_batch(
    completions: list[Completion],
    pad_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack completions into padded tensors.

    Returns (input_ids [B,T], attention_mask [B,T], completion_mask [B,T]).
    completion_mask is 1 on generated tokens, 0 on prompt + padding — so the
    loss trains only on what the policy produced.
    """
    seqs, cmasks = [], []
    for c in completions:
        full = list(c.prompt_token_ids) + list(c.completion_token_ids)
        seqs.append(full)
        cmasks.append([0] * len(c.prompt_token_ids) + [1] * len(c.completion_token_ids))

    B = len(seqs)
    T = max(len(s) for s in seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention = torch.zeros((B, T), dtype=torch.long)
    completion = torch.zeros((B, T), dtype=torch.float)
    for i, (s, m) in enumerate(zip(seqs, cmasks)):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
        attention[i, :L] = 1
        completion[i, :L] = torch.tensor(m, dtype=torch.float)

    return input_ids.to(device), attention.to(device), completion.to(device)
