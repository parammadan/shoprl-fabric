"""The shopping environment: state, actions, transitions, episode.

One episode = the model shopping toward a goal across turns. Each turn it emits
an action string; the env applies it, updates state, and returns the new context
string the model sees next. The episode ends on CHECKOUT or max_turns. The
trajectory reward (scoring the completed episode) lives in reward.py — built
separately, taught step by step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from shoprl.data.catalog import Product
from shoprl.data.prompts import satisfies

# ADD_TO_CART[LAP-0007] | APPLY_FILTER[max_price=1200,min_ram=16] | REMOVE[LAP-0007] | CHECKOUT
_ACTION_RE = re.compile(r"(ADD_TO_CART|APPLY_FILTER|REMOVE|CHECKOUT)(?:\s*\[(.*?)\])?", re.IGNORECASE)
_CONSTRAINT_KEYS = ("max_price", "min_ram", "max_weight", "min_battery")


@dataclass
class Action:
    kind: str          # ADD_TO_CART | APPLY_FILTER | REMOVE | CHECKOUT | INVALID
    arg: str = ""


def parse_action(text: str) -> Action:
    """First recognized action verb in the model's output. INVALID if none."""
    m = _ACTION_RE.search(text or "")
    if not m:
        return Action("INVALID")
    return Action(m.group(1).upper(), (m.group(2) or "").strip())


def _parse_criteria(arg: str) -> dict[str, float]:
    """'max_price=1200, min_ram=16' -> {'max_price':1200.0,'min_ram':16.0}."""
    out: dict[str, float] = {}
    for part in re.split(r"[,;]", arg):
        m = re.match(r"\s*([a-z_]+)\s*[=:<>]+\s*([\d.]+)", part, re.IGNORECASE)
        if m and m.group(1).lower() in _CONSTRAINT_KEYS:
            out[m.group(1).lower()] = float(m.group(2))
    return out


@dataclass
class Goal:
    """What the shopper wants: a budget, target constraints, and item count."""
    budget: float
    constraints: dict[str, float] = field(default_factory=dict)
    target_items: int = 1


@dataclass
class EnvState:
    budget_remaining: float
    cart: list[str] = field(default_factory=list)      # SKUs in the cart
    filters: dict[str, float] = field(default_factory=dict)
    turn: int = 0
    done: bool = False
    checked_out: bool = False


class ShopEnv:
    def __init__(self, catalog: list[Product], goal: Goal, max_turns: int = 8):
        self.catalog = catalog
        self.idx = {p.sku: p for p in catalog}
        self.goal = goal
        self.max_turns = max_turns
        self.state: EnvState | None = None

    def reset(self) -> str:
        self.state = EnvState(budget_remaining=self.goal.budget)
        return self._context()

    def step(self, action_text: str) -> tuple[str, bool, dict]:
        """Apply one action. Returns (context, done, info). info records what
        happened (valid + note) — the substrate for later per-turn credit."""
        s = self.state
        if s is None:
            raise RuntimeError("call reset() first")
        if s.done:
            return self._context(), True, {"action": "NOOP", "valid": False,
                                           "note": "episode already done"}
        a = parse_action(action_text)
        s.turn += 1
        info = {"action": a.kind, "arg": a.arg, "valid": True, "note": ""}

        if a.kind == "ADD_TO_CART":
            p = self.idx.get(a.arg.upper())
            if p is None:
                info.update(valid=False, note="unknown SKU")
            elif p.price > s.budget_remaining:
                info.update(valid=False, note="over budget")
            else:
                s.cart.append(p.sku)
                s.budget_remaining = round(s.budget_remaining - p.price, 2)
                info["note"] = f"added {p.sku} (${p.price:.0f})"
        elif a.kind == "REMOVE":
            sku = a.arg.upper()
            if sku in s.cart:
                s.cart.remove(sku)
                s.budget_remaining = round(s.budget_remaining + self.idx[sku].price, 2)
                info["note"] = f"removed {sku}"
            else:
                info.update(valid=False, note="not in cart")
        elif a.kind == "APPLY_FILTER":
            crit = _parse_criteria(a.arg)
            if crit:
                s.filters.update(crit)
                info["note"] = f"filters now {s.filters}"
            else:
                info.update(valid=False, note="unparseable criteria")
        elif a.kind == "CHECKOUT":
            s.checked_out = True
            s.done = True
            info["note"] = "checked out"
        else:
            info.update(valid=False, note="unrecognized action")

        if s.turn >= self.max_turns:
            s.done = True
            if not s.checked_out:
                info["note"] += " | max_turns reached (no checkout)"
        return self._context(), s.done, info

    def candidates(self, limit: int = 6) -> list[Product]:
        """Catalog products matching the CURRENT filters (what the shopper sees)."""
        matches = [p for p in self.catalog if satisfies(p, self.state.filters)]
        return matches[:limit]

    def _context(self) -> str:
        s = self.state
        g = self.goal
        lines = [
            f"GOAL: buy {g.target_items} laptop(s), budget ${g.budget:.0f}, "
            f"constraints {g.constraints or '(none)'}",
            f"Turn {s.turn}/{self.max_turns} | budget left ${s.budget_remaining:.0f} | "
            f"cart {s.cart or '(empty)'} | filters {s.filters or '(none)'}",
            "Candidates (matching your filters):",
        ]
        for p in self.candidates():
            lines.append(f"- {p.sku}: ${p.price:.0f} | {p.ram_gb}GB | {p.weight_lbs}lbs "
                         f"| {p.battery_hrs}hrs | {p.brand}")
        lines.append("Action: ADD_TO_CART[SKU] | APPLY_FILTER[key=val,...] "
                     "| REMOVE[SKU] | CHECKOUT")
        return "\n".join(lines)
