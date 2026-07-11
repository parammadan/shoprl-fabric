"""Rollout: turning prompts into sampled completions.

The RolloutEngine interface is the seam that lets us develop the whole platform
on an M1 (HF `generate`) and swap in vLLM on cloud GPU without touching the
learner, reward, or orchestration code.
"""
from shoprl.rollout.base import Completion, RolloutEngine, RolloutGroup
from shoprl.rollout.stub import StubRolloutEngine

__all__ = ["Completion", "RolloutEngine", "RolloutGroup", "StubRolloutEngine"]
