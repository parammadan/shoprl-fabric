"""Multi-turn Shopping Environment Simulator for trajectory-level RL.

A practice environment on top of the completed single-turn pipeline: the model
takes multiple actions (add/remove/filter/checkout) across turns, and the reward
is assigned at the END of the episode (delayed / trajectory-level) — the setting
where credit assignment and reward variance actually get hard.

This module is the SIMULATOR only (state, actions, transitions, episode). The
trajectory reward + credit assignment is built separately (taught step by step);
nothing here is wired into training yet.
"""
from shoprl.env.reward import assign_credit, episode_reward
from shoprl.env.shopenv import Action, EnvState, Goal, ShopEnv, parse_action

__all__ = ["Action", "EnvState", "Goal", "ShopEnv", "parse_action",
           "episode_reward", "assign_credit"]
