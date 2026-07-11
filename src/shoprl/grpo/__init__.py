"""GRPO core: the RL math that turns rewards into a policy update.

Built piece by piece with the math explained in comments (WHY, not just how):
  1. advantages  - group-relative, standardized (this file's first component)
  2. kl          - penalty against a frozen reference policy (next)
  3. loss        - the GRPO objective the learner backprops (after that)
"""
from shoprl.grpo.advantages import batch_group_advantages, group_advantages
from shoprl.grpo.kl import token_kl
from shoprl.grpo.loss import GRPOStats, grpo_loss

__all__ = [
    "group_advantages",
    "batch_group_advantages",
    "token_kl",
    "grpo_loss",
    "GRPOStats",
]
