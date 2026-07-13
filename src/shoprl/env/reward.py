"""Trajectory-level reward + credit assignment (no training wiring).

Two pieces:
  1. episode_reward(state, goal, idx) -> one scalar scoring the COMPLETED episode
     (valid purchase within budget/constraints + cart quality). This is the
     delayed, end-of-episode signal.
  2. assign_credit(reward, n_turns, scheme) -> per-turn advantages, i.e. how that
     one scalar is spread across the episode's turns. Two schemes implemented:
       - "uniform": every turn gets (reward - baseline)  [outcome-supervised;
         GRPO/RLOO/PPO lifted to the trajectory level]
       - "discounted": turn t gets gamma^(T-1-t) * (reward - baseline)  [later
         turns, nearer the reward, get more credit]

Kept pure + framework-free so it's unit-testable and, later, droppable into a
trajectory trainer. Nothing here touches the training loop yet.
"""
from __future__ import annotations

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies
from shoprl.env.shopenv import EnvState, Goal


def episode_reward(state: EnvState, goal: Goal, idx: dict[str, Product]) -> float:
    """Score a finished episode in [0, 1].

    Gates: must have CHECKED OUT with a non-empty cart (else 0 — no purchase).
    Budget is env-enforced (over-budget adds are rejected), so a checked-out cart
    is always within budget. Score then blends:
      - constraint satisfaction: fraction of cart items meeting the goal's
        constraints (the catalog's TRUE specs — verifiable, un-hackable)
      - item-count match: how close |cart| is to the target count
    """
    if not state.checked_out or not state.cart:
        return 0.0
    constraints_ok = sum(satisfies(idx[s], goal.constraints) for s in state.cart) / len(state.cart)
    if goal.target_items > 0:
        count_ok = max(0.0, 1.0 - abs(len(state.cart) - goal.target_items) / goal.target_items)
    else:
        count_ok = 1.0
    return round(0.6 * constraints_ok + 0.4 * count_ok, 4)


def assign_credit(reward: float, n_turns: int, scheme: str = "uniform",
                  gamma: float = 0.99, baseline: float = 0.0) -> list[float]:
    """Spread the trajectory reward across `n_turns` as per-turn advantages.

    The advantage a turn receives is what the policy gradient actually uses; the
    baseline (e.g. mean reward of sibling episodes for the same goal) is what
    turns the raw reward into a low-variance, centered signal.
    """
    if n_turns <= 0:
        return []
    adv = reward - baseline
    if scheme == "uniform":
        return [adv] * n_turns
    if scheme == "discounted":
        return [round(gamma ** (n_turns - 1 - t) * adv, 6) for t in range(n_turns)]
    raise ValueError(f"unknown credit scheme: {scheme!r}")
