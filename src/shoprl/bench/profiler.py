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


def padding_waste(attention_mask) -> float:
    """Fraction of batch positions that are padding (0 = perfectly packed).
    attention_mask: [B, T] tensor with 1 = real token, 0 = pad."""
    total = attention_mask.numel()
    real = float(attention_mask.sum())
    return 1.0 - real / total if total else 0.0
