"""Phase timing + padding-waste measurement — the measurement core.

PhaseTimer accumulates wall-clock per named phase (rollout / reward / optimize /
eval) so a run's time breakdown is measured, not guessed. padding_waste reports
the fraction of batch tokens that are pad (the thing sequence-packing removes).
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class PhaseTimer:
    def __init__(self):
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def report(self, total_tokens: int | None = None) -> dict:
        total = sum(self.times.values())
        breakdown = {
            name: {
                "seconds": round(sec, 3),
                "pct": round(100 * sec / total, 1) if total else 0.0,
                "calls": self.counts[name],
            }
            for name, sec in sorted(self.times.items(), key=lambda kv: -kv[1])
        }
        out = {"total_s": round(total, 3), "breakdown": breakdown}
        if total_tokens is not None and total > 0:
            out["tokens_per_sec"] = round(total_tokens / total, 1)
        return out


def rollout_metrics(n_completions: int, rollout_seconds: float, steps: int,
                    total_s: float, ttft_ms: float | None = None) -> dict:
    """Rollout throughput/latency metrics for a HF-vs-vLLM comparison. All
    derived from measured counts + times — nothing invented. TTFT is reported
    only when the engine supplies it (vLLM); HF batch-generate leaves it None."""
    return {
        "n_completions": n_completions,
        "requests_per_sec": round(n_completions / rollout_seconds, 2)
        if rollout_seconds > 0 else None,
        "rollout_latency_ms_per_request": round(rollout_seconds / n_completions * 1000, 1)
        if n_completions else None,
        "iteration_time_s": round(total_s / steps, 3) if steps else None,
        "ttft_ms": ttft_ms,
    }


def padding_waste(attention_mask) -> float:
    """Fraction of batch positions that are padding (0 = perfectly packed).
    attention_mask: [B, T] tensor with 1 = real token, 0 = pad."""
    total = attention_mask.numel()
    real = float(attention_mask.sum())
    return 1.0 - real / total if total else 0.0
