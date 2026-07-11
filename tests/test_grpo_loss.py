import torch

from shoprl.grpo.loss import grpo_loss


def _base(logp):
    """Helper: policy == old == ref (ratio 1, kl 0)."""
    return dict(logp_policy=logp.clone().requires_grad_(True),
                logp_old=logp.clone(), logp_ref=logp.clone())


def test_baseline_loss_equals_negative_mean_advantage():
    # ratio=1, kl=0 -> per_token = A_i; loss = -mean over tokens of A.
    logp = torch.full((2, 3), -0.5)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 3)
    loss, stats = grpo_loss(advantages=adv, mask=mask, beta=0.04, **_base(logp))
    # tokens: seq0 all +1, seq1 all -1 -> mean 0 -> loss 0.
    assert abs(loss.item()) < 1e-6
    assert abs(stats.mean_kl) < 1e-6
    assert abs(stats.mean_ratio - 1.0) < 1e-6


def test_positive_advantage_gradient_increases_logprob():
    logp = torch.full((1, 4), -0.8)
    b = _base(logp)
    adv = torch.tensor([2.0])
    mask = torch.ones(1, 4)
    loss, _ = grpo_loss(advantages=adv, mask=mask, **b)
    loss.backward()
    # descent step = -grad; loss should push logp_policy UP for positive adv.
    assert torch.all(b["logp_policy"].grad < 0)


def test_mask_excludes_prompt_and_pad_tokens():
    logp = torch.full((1, 4), -0.5)
    adv = torch.tensor([1.0])
    full = grpo_loss(advantages=adv, mask=torch.ones(1, 4), **_base(logp))[0]
    half = grpo_loss(advantages=adv, mask=torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
                     **_base(logp))[0]
    # Same per-token value everywhere, so masking changes denom not the mean.
    assert torch.allclose(full, half, atol=1e-6)

    # Changing a MASKED token's logp must not affect the loss.
    lp = torch.tensor([[-0.5, -0.5, -0.5, -0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    a = grpo_loss(logp_policy=lp.clone().requires_grad_(True), logp_old=lp.clone(),
                  logp_ref=lp.clone(), advantages=adv, mask=mask)[0]
    lp2 = lp.clone(); lp2[0, 3] = -5.0
    b = grpo_loss(logp_policy=lp2.clone().requires_grad_(True), logp_old=lp.clone(),
                  logp_ref=lp.clone(), advantages=adv, mask=mask)[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_kl_penalty_raises_loss():
    logp_pol = torch.full((1, 3), -0.5, requires_grad=True)
    logp_old = torch.full((1, 3), -0.5)
    logp_ref = torch.full((1, 3), -2.0)  # policy diverged from reference
    adv = torch.tensor([0.0])            # isolate the KL term
    mask = torch.ones(1, 3)
    loss, stats = grpo_loss(logp_pol, logp_old, logp_ref, adv, mask, beta=0.1)
    assert stats.mean_kl > 0
    assert loss.item() > 0  # with A=0, loss = +beta*mean_kl


def test_clip_fraction_reported():
    # Large ratio (policy far above old) with positive adv triggers clipping.
    logp_pol = torch.full((1, 4), 0.0, requires_grad=True)
    logp_old = torch.full((1, 4), -2.0)  # ratio = exp(2) ~ 7.4 >> 1+eps
    logp_ref = torch.full((1, 4), 0.0)
    adv = torch.tensor([1.0])
    _, stats = grpo_loss(logp_pol, logp_old, logp_ref, adv, torch.ones(1, 4))
    assert stats.clip_fraction == 1.0
