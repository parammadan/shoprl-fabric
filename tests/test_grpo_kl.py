import torch

from shoprl.grpo.kl import token_kl


def test_zero_when_policy_matches_reference():
    lp = torch.tensor([-0.5, -1.2, -0.1])
    kl = token_kl(lp.clone(), lp.clone())
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_is_nonnegative():
    torch.manual_seed(0)
    lp_pol = torch.randn(1000)
    lp_ref = torch.randn(1000)
    kl = token_kl(lp_pol, lp_ref)
    assert torch.all(kl >= -1e-6)


def test_grows_with_divergence():
    lp_ref = torch.tensor([-1.0])
    near = token_kl(torch.tensor([-1.1]), lp_ref)
    far = token_kl(torch.tensor([-3.0]), lp_ref)
    assert far.item() > near.item()


def test_gradient_flows_to_policy_only():
    lp_pol = torch.tensor([-0.7], requires_grad=True)
    lp_ref = torch.tensor([-1.3], requires_grad=True)
    token_kl(lp_pol, lp_ref).sum().backward()
    assert lp_pol.grad is not None
    assert lp_ref.grad is None  # reference is detached / frozen
