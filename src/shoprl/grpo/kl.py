"""KL penalty against a frozen reference policy (Schulman's k3 estimator).

GRPO adds a per-token KL penalty straight into the loss to keep the policy from
drifting away from the reference (the frozen starting model). This is the leash
against reward hacking / degeneration.

Estimator (k3), per token:
    delta = logp_ref - logp_policy
    kl    = exp(delta) - delta - 1        # always >= 0, unbiased, low variance

Contrast with the naive single-sample estimate (logp_policy - logp_ref): that is
unbiased for the mean KL but can be negative on a given token and has higher
variance. k3 fixes both, which matters when it's a per-token penalty summed over
long sequences.

torch (not pure Python) because the gradient must flow through logp_policy: the
KL is part of the differentiable objective. logp_ref is detached (the reference
is frozen — it contributes no gradient).
"""
from __future__ import annotations

import torch


def token_kl(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token k3 KL estimate. Shapes match; result is elementwise, >= 0.

    logp_policy: log-prob of each sampled token under the CURRENT policy (grad).
    logp_ref:    log-prob of the SAME tokens under the frozen reference (no grad).
    """
    # Reference is frozen: never backprop into it.
    logp_ref = logp_ref.detach()
    delta = logp_ref - logp_policy
    return torch.exp(delta) - delta - 1.0
