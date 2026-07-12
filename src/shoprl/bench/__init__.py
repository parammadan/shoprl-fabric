"""Efficiency benchmarking: measure the same run under different rollout/batch
configs and profile where wall-clock actually goes (rollout vs compute vs eval).

The recurring project finding is that ROLLOUT dominates RL wall-clock; this
harness quantifies that and lets us A/B the levers (async rollout, sequence
packing, reward-worker parallelism, vLLM) with measured numbers — never guessed.
"""
from shoprl.bench.profiler import PhaseTimer, padding_waste

__all__ = ["PhaseTimer", "padding_waste"]
