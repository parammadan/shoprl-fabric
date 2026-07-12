"""Pluggable RL post-training algorithms behind one interface.

All three algorithms share the same rollout, reward, batching, KL, and clipped
policy-gradient machinery; they differ ONLY in how the advantage baseline is
formed (and PPO additionally learns a value network):

  - GRPO: baseline = group mean, standardized by group std (critic-free).
  - RLOO: baseline = mean of the OTHER samples in the group (leave-one-out,
          unbiased; no std normalization). Critic-free.
  - PPO:  baseline = a learned value network V(s); advantage = reward - V.
          Adds a critic (value head + value loss) -> ~2x params/compute.

See base.RLTrainer for the shared interface: generate_rollouts,
calculate_rewards, optimize, save_checkpoint.
"""
from shoprl.rl.base import RLTrainer
from shoprl.rl.factory import build_trainer

__all__ = ["RLTrainer", "build_trainer"]
