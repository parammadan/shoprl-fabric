"""Benchmark harness — measure the same run under different efficiency configs.

    python -m shoprl.bench.harness --config configs/bench_base.yaml --steps 8 \
        --out results/bench_base.json [--reward-workers 4] [--pack] [--async-rollout]

Levers measured (all against the SAME task/model, so deltas are attributable):
  (a) sync vs async rollout  — prefetch the next batch's rollout in a thread while
      the learner optimizes the current batch; reported as rollout_wait vs rollout.
  (b) sequence packing        — length-bucket the batch to cut pad tokens; reported
      as padding_waste vs padding_waste_packed.
  (c) reward-worker parallelism — score completions in a thread pool.
  (d) profiling               — wall-clock breakdown across rollout / reward /
      optimize, plus tokens/sec and peak GPU memory.

Drives the RLTrainer interface (generate_rollouts / calculate_rewards / optimize)
so it works for any algorithm and needs no trainer changes. Numbers are measured;
this file never invents throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

import torch

from shoprl.bench.profiler import PhaseTimer, padding_waste, rollout_metrics
from shoprl.config import load_config
from shoprl.grpo.logprobs import build_batch
from shoprl.reward import compute_reward
from shoprl.rl.factory import build_trainer


def _parallel_rewards(groups, contexts, cfg, workers):
    """Reward scoring across a thread pool (lever c). Returns the same shape as
    the trainer's calculate_rewards."""
    jobs = []  # (group_idx, comp)
    for gi, (ctx, group) in enumerate(zip(contexts, groups)):
        for comp in group.completions:
            jobs.append((gi, ctx, comp))

    def score(job):
        _, ctx, comp = job
        return compute_reward(comp.text, ctx, weights=cfg.rewards.weights,
                              hallucination_penalty=cfg.rewards.hallucination_penalty)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(score, jobs))

    completions, rpg = [], [[] for _ in groups]
    for (gi, _ctx, comp), r in zip(jobs, results):
        completions.append(comp)
        rpg[gi].append(r.total)
    return completions, rpg


def benchmark(config, steps: int, reward_workers: int = 1,
              pack: bool = False, async_rollout: bool = False,
              rollout_only: bool = False) -> dict:
    trainer = build_trainer(config)
    # For a non-hf rollout engine (vLLM), swap in a factory-built engine so the
    # rollout phase actually measures that backend. vLLM keeps its own base-model
    # copy (the documented on-policy weight-sync gap) — valid for a rollout
    # THROUGHPUT benchmark; the LoRA optimize still runs on the HF model.
    if config.rollout.engine != "hf":
        from shoprl.rollout.factory import build_engine
        trainer.engine = build_engine(config)
    pt = PhaseTimer()
    pad_id = trainer.tokenizer.pad_token_id
    device = trainer.device
    total_tokens = 0
    n_completions = 0
    pad_def, pad_pack = [], []
    ex = ThreadPoolExecutor(max_workers=1) if async_rollout else None
    prefetch = None

    for step in range(steps):
        if async_rollout and prefetch is not None:
            with pt.phase("rollout_wait"):
                groups, contexts = prefetch.result()
        else:
            with pt.phase("rollout"):
                groups, contexts = trainer.generate_rollouts(step)
        if async_rollout:
            prefetch = ex.submit(trainer.generate_rollouts, step + 1)

        with pt.phase("reward"):
            if reward_workers > 1:
                completions, rpg = _parallel_rewards(groups, contexts, config, reward_workers)
            else:
                completions, rpg, _bd = trainer.calculate_rewards(groups, contexts)

        total_tokens += sum(len(c.completion_token_ids) for c in completions)
        n_completions += len(completions)

        # padding-waste: default order vs length-bucketed (lever b)
        _, attn, _ = build_batch(completions, pad_id, device)
        pad_def.append(padding_waste(attn))
        packed = sorted(completions,
                        key=lambda c: len(c.prompt_token_ids) + len(c.completion_token_ids))
        _, attn_p, _ = build_batch(packed, pad_id, device)
        pad_pack.append(padding_waste(attn_p))

        if not rollout_only:
            with pt.phase("optimize"):
                trainer.optimize(completions, rpg)

    if ex:
        ex.shutdown(wait=False)

    rep = pt.report(total_tokens)
    # rollout throughput/latency for the HF-vs-vLLM comparison
    rollout_seconds = pt.times.get("rollout", 0.0) + pt.times.get("rollout_wait", 0.0)
    rep.update(rollout_metrics(
        n_completions, rollout_seconds, steps, rep["total_s"],
        ttft_ms=getattr(trainer.engine, "last_ttft_ms", None)))
    rep["padding_waste"] = round(statistics.mean(pad_def), 3)
    rep["padding_waste_packed"] = round(statistics.mean(pad_pack), 3)
    rep["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None
    rep["config"] = {
        "algorithm": config.algorithm, "steps": steps,
        "batch": config.training.prompts_per_step * config.rollout.num_samples,
        "engine": config.rollout.engine, "dtype": config.model.dtype, "device": device,
        "reward_workers": reward_workers, "pack": pack, "async_rollout": async_rollout,
        "rollout_only": rollout_only,
    }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.bench.harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--reward-workers", type=int, default=1)
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--async-rollout", action="store_true")
    ap.add_argument("--rollout-only", action="store_true",
                    help="skip the optimize phase — pure rollout-throughput "
                         "benchmark (lets the vLLM leg run in an isolated env)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = load_config(args.config)
    rep = benchmark(config, args.steps, reward_workers=args.reward_workers,
                    pack=args.pack, async_rollout=args.async_rollout,
                    rollout_only=args.rollout_only)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[bench] total {rep['total_s']}s | tok/s {rep.get('tokens_per_sec')} | "
          f"req/s {rep.get('requests_per_sec')} | "
          f"lat {rep.get('rollout_latency_ms_per_request')}ms/req | "
          f"iter {rep.get('iteration_time_s')}s | ttft {rep.get('ttft_ms')} | "
          f"mem {rep.get('peak_mem_gb')}GB | "
          f"pad {rep['padding_waste']}->{rep['padding_waste_packed']} -> {args.out}")
    for name, b in rep["breakdown"].items():
        print(f"    {name:14s} {b['seconds']:>8.2f}s  {b['pct']:>5.1f}%")


if __name__ == "__main__":
    main()
