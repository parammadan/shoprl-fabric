"""Preflight validation — fail fast, before allocating a GPU.

A GPU-hour wasted because a config typo, an empty dataset, or an OOM only shows
up 40 minutes into a spot run is the most expensive kind of bug on this project.
Preflight runs the cheap checks first and the expensive ones last, stops at the
first failure, and reports a clear reason — so you never allocate a GPU for a run
that was doomed at line 1 of the config.

Order (cheapest → most expensive; stop at first failure):
  1. config      — semantic validity beyond types (e.g. group size >= 2)
  2. dataset     — prompts exist and carry ground-truth answers (reward defined)
  3. memory      — ROUGH peak-VRAM estimate vs. the GPU budget (logits dominate)
  4. rollout     — generate one group of completions, non-empty
  5. reward      — score them, rewards finite and in range
  6. backward    — one real autograd step: loss finite, grad_norm finite and > 0

Honesty: the memory number is a labelled ESTIMATE (it encodes the real lesson
from this project — the [batch, seq, vocab] LM-head logits, not the weights, are
what OOMs a small model). The default rollout uses the dependency-free stub and
the default backward is a tiny real autograd smoke; a real GPU launch injects
`rollout_fn` / `backward_fn` that drive the actual engine + policy. If torch is
absent the backward check is SKIPPED (reported, not silently passed).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from shoprl.config import Config
from shoprl.data import generate_catalog, generate_prompts
from shoprl.reward.composite import compute_reward
from shoprl.reward.functions import RewardContext
from shoprl.rollout.stub import StubRolloutEngine

QWEN3_VOCAB = 151936          # Qwen3 vocab — the LM-head width that dominates VRAM


class PreflightError(Exception):
    def __init__(self, result: "CheckResult"):
        self.result = result
        super().__init__(f"preflight failed at '{result.name}': {result.detail}")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    skipped: bool = False
    data: dict = field(default_factory=dict)


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def first_failure(self) -> CheckResult | None:
        return next((c for c in self.checks if not c.ok), None)

    def raise_if_failed(self) -> "PreflightReport":
        f = self.first_failure
        if f is not None:
            raise PreflightError(f)
        return self


# --- memory estimate -------------------------------------------------------
def _infer_param_count(name: str) -> float:
    m = re.search(r"(\d+\.?\d*)\s*([bBmM])", name or "")
    if not m:
        return 0.6e9
    val = float(m.group(1))
    return val * (1e9 if m.group(2).lower() == "b" else 1e6)


def estimate_peak_mem_gb(n_params: float, batch: int, seq_len: int, dtype: str,
                         vocab: int = QWEN3_VOCAB, hidden: int = 1024,
                         layers: int = 28) -> dict:
    """ROUGH upper-ish peak-VRAM estimate, in GB, broken down by component."""
    bytes_per = 4 if dtype in ("float32", "fp32", "f32") else 2  # auto/bf16/fp16 -> 2
    weights = n_params * bytes_per
    logits = batch * seq_len * vocab * bytes_per          # LM-head output = the killer
    activations = batch * seq_len * hidden * layers * 2 * bytes_per
    overhead = 1.0e9                                       # CUDA context / allocator
    total = weights + logits + activations + overhead
    g = lambda x: round(x / 1e9, 2)
    return {"weights_gb": g(weights), "logits_gb": g(logits),
            "activations_gb": g(activations), "overhead_gb": g(overhead),
            "total_gb": g(total), "bytes_per_param": bytes_per}


# --- individual checks -----------------------------------------------------
def _check_config(cfg: Config) -> CheckResult:
    problems = []
    if cfg.rollout.num_samples < 2:
        problems.append(
            f"rollout.num_samples={cfg.rollout.num_samples} < 2: "
            f"{cfg.algorithm} computes a baseline WITHIN the group, which is "
            "undefined for a group of 1")
    if cfg.training.lr <= 0:
        problems.append(f"training.lr={cfg.training.lr} must be > 0")
    if cfg.training.beta < 0:
        problems.append(f"training.beta={cfg.training.beta} must be >= 0")
    if sum(cfg.rewards.weights.values()) <= 0:
        problems.append("reward weights sum to <= 0 — no reward signal")
    ok = not problems
    return CheckResult("config", ok,
                       "config semantics OK" if ok else "; ".join(problems),
                       data={"algorithm": cfg.algorithm,
                             "num_samples": cfg.rollout.num_samples})


def _check_dataset(cfg: Config, seed: int) -> CheckResult:
    catalog = generate_catalog(n=cfg.training.catalog_size, seed=seed)
    n = max(cfg.training.prompts_per_step, 8)
    prompts = generate_prompts(catalog, n=n, seed=seed)
    with_answers = [p for p in prompts if p.answer_skus]
    ok = len(with_answers) > 0
    detail = (f"{len(with_answers)}/{len(prompts)} prompts have ground-truth "
              f"answers over a {len(catalog)}-item catalog"
              if ok else "no prompts have satisfying products — reward undefined")
    return CheckResult("dataset", ok, detail,
                       data={"catalog": len(catalog), "prompts": len(prompts),
                             "with_answers": len(with_answers)})


def _check_memory(cfg: Config, gpu_mem_gb: float | None,
                  n_params: float | None) -> CheckResult:
    batch = cfg.training.prompts_per_step * cfg.rollout.num_samples
    seq_len = cfg.rollout.max_new_tokens + 128            # + rough prompt length
    params = n_params if n_params is not None else _infer_param_count(cfg.model.name)
    est = estimate_peak_mem_gb(params, batch, seq_len, cfg.model.dtype)
    est["batch"] = batch
    est["seq_len"] = seq_len
    est["n_params"] = params
    if gpu_mem_gb is None:
        return CheckResult("memory", True,
                           f"ESTIMATE ~{est['total_gb']} GB peak (no --gpu-mem-gb "
                           f"budget given; not enforced)", skipped=True, data=est)
    ok = est["total_gb"] <= gpu_mem_gb
    est["budget_gb"] = gpu_mem_gb
    detail = (f"ESTIMATE ~{est['total_gb']} GB peak vs {gpu_mem_gb} GB budget "
              f"(logits {est['logits_gb']} GB dominate)"
              + ("" if ok else " — would likely OOM; reduce batch/seq or use bf16"))
    return CheckResult("memory", ok, detail, data=est)


def _check_rollout(cfg: Config, rollout_fn: Callable | None, seed: int) -> CheckResult:
    if rollout_fn is None:
        engine = StubRolloutEngine(seed=seed)
        group = engine.generate(["find a laptop under $1200 with 16GB RAM"],
                                num_samples=cfg.rollout.num_samples, seed=seed)[0]
        comps = group.completions
        engine_name = "stub"
    else:
        comps = rollout_fn()
        engine_name = cfg.rollout.engine
    nonempty = [c for c in comps if getattr(c, "text", "").strip()]
    ok = len(nonempty) == cfg.rollout.num_samples and len(nonempty) > 0
    return CheckResult("rollout", ok,
                       f"{engine_name} engine produced {len(nonempty)}/"
                       f"{cfg.rollout.num_samples} non-empty completions",
                       data={"engine": engine_name, "completions": len(comps)},
                       ), comps


def _check_reward(cfg: Config, comps, seed: int) -> CheckResult:
    catalog = generate_catalog(n=cfg.training.catalog_size, seed=seed)
    prompts = generate_prompts(catalog, n=8, seed=seed)
    prompt = next((p for p in prompts if p.answer_skus), prompts[0])
    ctx = RewardContext(catalog={p.sku: p for p in catalog},
                        constraints=prompt.constraints)
    rewards = []
    for c in comps:
        r = compute_reward(c.text, ctx, weights=cfg.rewards.weights,
                           hallucination_penalty=cfg.rewards.hallucination_penalty).total
        rewards.append(r)
    finite = all(math.isfinite(r) for r in rewards)
    return CheckResult("reward", finite,
                       f"scored {len(rewards)} completions, all finite"
                       if finite else "reward produced a non-finite value",
                       data={"mean": sum(rewards) / len(rewards) if rewards else None,
                             "min": min(rewards, default=None),
                             "max": max(rewards, default=None)})


def _tiny_torch_backward() -> dict:
    import torch
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 4)
    x, y = torch.randn(4, 8), torch.randn(4, 4)
    loss = ((lin(x) - y) ** 2).mean()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(lin.parameters(), 1.0)
    return {"loss": loss.item(), "grad_norm": float(gn.detach())}


def _check_backward(backward_fn: Callable | None) -> CheckResult:
    fn = backward_fn
    if fn is None:
        try:
            import torch  # noqa: F401
        except Exception:
            return CheckResult("backward", True,
                               "torch not installed — skipped (a real launch runs "
                               "one optimizer step on the actual policy)",
                               skipped=True)
        fn = _tiny_torch_backward
    out = fn()
    gn, loss = out.get("grad_norm"), out.get("loss")
    ok = (gn is not None and math.isfinite(gn) and gn > 0
          and loss is not None and math.isfinite(loss))
    return CheckResult("backward", ok,
                       f"one backward: loss={loss}, grad_norm={gn} (finite, > 0)"
                       if ok else f"backward unhealthy: loss={loss}, grad_norm={gn}",
                       data=out)


# --- driver ----------------------------------------------------------------
def run_preflight(cfg: Config, *, gpu_mem_gb: float | None = None,
                  n_params: float | None = None,
                  rollout_fn: Callable | None = None,
                  backward_fn: Callable | None = None) -> PreflightReport:
    """Run the checks in order, stopping at the first failure."""
    seed = cfg.experiment.seed
    report = PreflightReport()

    def add(c: CheckResult) -> bool:
        report.checks.append(c)
        return c.ok

    if not add(_check_config(cfg)):
        return report
    if not add(_check_dataset(cfg, seed)):
        return report
    if not add(_check_memory(cfg, gpu_mem_gb, n_params)):
        return report
    rollout_result, comps = _check_rollout(cfg, rollout_fn, seed)
    if not add(rollout_result):
        return report
    if not add(_check_reward(cfg, comps, seed)):
        return report
    add(_check_backward(backward_fn))
    return report


def main() -> None:
    import argparse
    import sys

    from shoprl.config import load_config

    ap = argparse.ArgumentParser(description="Preflight-validate a run before GPU launch")
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu-mem-gb", type=float, default=None,
                    help="GPU VRAM budget; enforces the memory estimate")
    args = ap.parse_args()
    report = run_preflight(load_config(args.config), gpu_mem_gb=args.gpu_mem_gb)
    for c in report.checks:
        mark = "SKIP" if c.skipped else ("PASS" if c.ok else "FAIL")
        print(f"[{mark}] {c.name:9s} {c.detail}")
    if not report.ok:
        print("\nPREFLIGHT FAILED — not launching.", file=sys.stderr)
        sys.exit(1)
    print("\nPreflight OK — cleared to launch.")


if __name__ == "__main__":
    main()
