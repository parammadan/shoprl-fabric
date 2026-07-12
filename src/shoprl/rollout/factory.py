"""Build a RolloutEngine from config.

The HF import is lazy (inside the branch) so that using the stub engine — and
thus the whole plumbing/test path — never requires torch/transformers.
"""
from __future__ import annotations

from shoprl.config import Config
from shoprl.rollout.base import RolloutEngine
from shoprl.rollout.stub import StubRolloutEngine


def build_engine(config: Config) -> RolloutEngine:
    kind = config.rollout.engine
    if kind == "stub":
        return StubRolloutEngine(seed=config.experiment.seed)
    if kind == "hf":
        from shoprl.rollout.hf import HFRolloutEngine  # lazy: needs torch

        return HFRolloutEngine(config)
    if kind == "vllm":
        from shoprl.rollout.vllm import VLLMRolloutEngine  # lazy: GPU-only (needs vllm)

        return VLLMRolloutEngine(config)
    raise ValueError(f"Unknown rollout engine: {kind!r}")
