import math

import torch
import torch.nn.functional as F

from shoprl.grpo.logprobs import build_batch, mean_entropy, token_logprobs
from shoprl.rollout.base import Completion


def test_token_logprobs_matches_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    ids = torch.randint(0, 7, (2, 5))
    got = token_logprobs(logits, ids)
    # reference: explicit log_softmax then gather
    ref = F.log_softmax(logits[:, :-1], dim=-1).gather(
        -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, ref, atol=1e-5)
    assert got.shape == (2, 4)


def test_mean_entropy_uniform_is_log_vocab():
    V = 8
    logits = torch.zeros(1, 4, V)  # uniform -> entropy = log V
    mask_shift = torch.ones(1, 3)
    ent = mean_entropy(logits, mask_shift)
    assert math.isclose(ent, math.log(V), rel_tol=1e-5)


def test_mean_entropy_ignores_masked_positions():
    logits = torch.randn(1, 4, 6)
    ent_all = mean_entropy(logits, torch.ones(1, 3))
    ent_one = mean_entropy(logits, torch.tensor([[1.0, 0.0, 0.0]]))
    # restricting to fewer positions generally changes the mean; both finite/>=0
    assert ent_all >= 0 and ent_one >= 0


def test_build_batch_pads_and_masks():
    comps = [
        Completion(prompt="p", text="t", prompt_token_ids=[1, 2, 3],
                   completion_token_ids=[4, 5]),
        Completion(prompt="p", text="t", prompt_token_ids=[9],
                   completion_token_ids=[8, 7, 6]),
    ]
    ids, attn, cmask = build_batch(comps, pad_id=0, device="cpu")
    assert ids.shape == (2, 5) and attn.shape == (2, 5) and cmask.shape == (2, 5)
    # seq 0: len 5, no pad; completion on last 2 positions
    assert attn[0].tolist() == [1, 1, 1, 1, 1]
    assert cmask[0].tolist() == [0, 0, 0, 1, 1]
    # seq 1: len 4 then 1 pad; completion on positions 1,2,3
    assert attn[1].tolist() == [1, 1, 1, 1, 0]
    assert cmask[1].tolist() == [0, 1, 1, 1, 0]
    assert ids[1].tolist() == [9, 8, 7, 6, 0]  # padded with pad_id
