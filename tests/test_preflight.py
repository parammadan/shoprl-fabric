"""Preflight tests: fail-fast ordered validation before GPU launch."""
import pytest

from shoprl.config import Config, RolloutConfig, TrainingConfig
from shoprl.platform.preflight import (PreflightError, _infer_param_count,
                                       estimate_peak_mem_gb, run_preflight)


def _cfg(**over) -> Config:
    roll = RolloutConfig(engine="stub", num_samples=over.pop("num_samples", 2),
                         max_new_tokens=over.pop("max_new_tokens", 64))
    train = TrainingConfig(prompts_per_step=over.pop("prompts_per_step", 1),
                           catalog_size=120, lr=over.pop("lr", 1e-5),
                           beta=over.pop("beta", 0.04))
    return Config(algorithm="grpo", rollout=roll, training=train)


# --- config check ----------------------------------------------------------
def test_group_size_one_fails_config_first():
    r = run_preflight(_cfg(num_samples=1))
    assert not r.ok
    assert r.first_failure.name == "config"
    assert "group" in r.first_failure.detail.lower()
    assert len(r.checks) == 1                       # stopped immediately


def test_bad_lr_fails_config():
    r = run_preflight(_cfg(lr=0.0))
    assert not r.ok and r.first_failure.name == "config"


# --- happy path ------------------------------------------------------------
def test_good_config_passes_all_in_order():
    r = run_preflight(_cfg())
    assert r.ok
    assert [c.name for c in r.checks] == \
        ["config", "dataset", "memory", "rollout", "reward", "backward"]


def test_dataset_and_reward_are_real():
    r = run_preflight(_cfg())
    ds = next(c for c in r.checks if c.name == "dataset")
    assert ds.ok and ds.data["with_answers"] > 0
    rw = next(c for c in r.checks if c.name == "reward")
    assert rw.ok and rw.data["mean"] is not None


# --- memory ----------------------------------------------------------------
def test_memory_budget_enforced_and_fails_fast():
    r = run_preflight(_cfg(), gpu_mem_gb=0.5)        # 0.6B weights alone exceed this
    assert not r.ok and r.first_failure.name == "memory"
    assert len(r.checks) == 3                        # rollout/reward/backward skipped
    assert "OOM" in r.first_failure.detail


def test_memory_estimate_logits_dominate():
    est = estimate_peak_mem_gb(0.6e9, batch=64, seq_len=1024, dtype="bfloat16")
    assert est["logits_gb"] > est["weights_gb"]      # the real OOM lesson
    assert est["total_gb"] > est["logits_gb"]


def test_no_budget_memory_is_skip_not_fail():
    r = run_preflight(_cfg())                         # no gpu_mem_gb
    mem = next(c for c in r.checks if c.name == "memory")
    assert mem.ok and mem.skipped


def test_param_inference():
    assert _infer_param_count("Qwen/Qwen3-0.6B") == pytest.approx(0.6e9)
    assert _infer_param_count("meta/Llama-7B") == pytest.approx(7e9)
    assert _infer_param_count("tiny-125M") == pytest.approx(125e6)


# --- backward injection ----------------------------------------------------
def test_backward_fn_injection_detects_dead_gradient():
    r = run_preflight(_cfg(), backward_fn=lambda: {"loss": 1.0, "grad_norm": 0.0})
    assert not r.ok and r.first_failure.name == "backward"


def test_backward_fn_healthy_passes():
    r = run_preflight(_cfg(), backward_fn=lambda: {"loss": 0.5, "grad_norm": 2.8})
    assert r.ok


# --- raise helper ----------------------------------------------------------
def test_raise_if_failed():
    with pytest.raises(PreflightError) as ei:
        run_preflight(_cfg(num_samples=1)).raise_if_failed()
    assert "config" in str(ei.value)
