"""Training alerting rules — catch the failure modes before they waste a run.

Declarative thresholds + pure checkers over the metrics the trainer already logs.
Two entry points:
  - check_step(step_metrics)  -> per-step alerts (KL blow-up, entropy collapse,
                                 grad-norm spike, non-finite grad/kl/loss)
  - check_run(result)         -> run-level alerts (stability failures, reward
                                 stall/regression, KL blow-up anywhere)

CLI (exit code = severity, for monitoring/CI):
    python -m shoprl.observability.alerts --result results/compare_grpo.json
    python -m shoprl.observability.alerts --metrics runs/<exp>/metrics.jsonl
  exit 0 = clean, 1 = warnings only, 2 = at least one CRITICAL.

The failure modes these guard (all seen for real this project):
  KL blow-up      -> policy diverging from the reference (reward-hacking risk).
  entropy collapse-> near-deterministic/degenerate generation (the train()-mode
                     rollout bug produced entropy ~4.4; a *collapse* is the low end).
  non-finite      -> NaN/inf grad or loss = training diverged.
  grad spike      -> unclipped instability.
  reward stall/regression -> no learning / over-optimization (the n=64 sweep
                     showed flat held-out despite high KL).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field


class Level:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_RANK = {Level.INFO: 0, Level.WARNING: 1, Level.CRITICAL: 2}


@dataclass
class Alert:
    level: str
    rule: str
    message: str
    step: int | None = None
    value: float | None = None


@dataclass
class Thresholds:
    kl_warn: float = 0.5           # KL above this = drifting
    kl_crit: float = 1.0           # KL above this = blow-up
    entropy_collapse: float = 0.05  # entropy below this = degenerate
    grad_norm_spike: float = 50.0   # pre-clip grad norm above this = unstable
    reward_regression: float = 0.02  # after < before - this = regressed
    reward_stall: float = 0.01       # |after - before| < this = no learning


def check_step(m: dict, th: Thresholds | None = None) -> list[Alert]:
    th = th or Thresholds()
    step = m.get("step")
    out: list[Alert] = []
    # non-finite grad/kl/loss -> divergence
    for key in ("grad_norm", "kl", "loss"):
        v = m.get(key)
        if v is not None and not math.isfinite(v):
            out.append(Alert(Level.CRITICAL, "nonfinite", f"{key} is non-finite ({v})", step, v))
    kl, ent, gn = m.get("kl"), m.get("entropy"), m.get("grad_norm")
    if kl is not None and math.isfinite(kl):
        if kl >= th.kl_crit:
            out.append(Alert(Level.CRITICAL, "kl_blowup", f"KL {kl:.3f} ≥ {th.kl_crit} (policy diverging from reference)", step, kl))
        elif kl >= th.kl_warn:
            out.append(Alert(Level.WARNING, "kl_high", f"KL {kl:.3f} ≥ {th.kl_warn}", step, kl))
    if ent is not None and math.isfinite(ent) and ent < th.entropy_collapse:
        out.append(Alert(Level.WARNING, "entropy_collapse", f"entropy {ent:.3f} < {th.entropy_collapse} (degenerate generation)", step, ent))
    if gn is not None and math.isfinite(gn) and gn >= th.grad_norm_spike:
        out.append(Alert(Level.WARNING, "grad_spike", f"grad_norm {gn:.1f} ≥ {th.grad_norm_spike}", step, gn))
    return out


def check_run(result: dict, th: Thresholds | None = None) -> list[Alert]:
    """result = a shoprl.rl.run JSON (has step_metrics + held_out_before/after +
    stability_failures) OR {"step_metrics": [...]}."""
    th = th or Thresholds()
    out: list[Alert] = []
    steps = result.get("step_metrics", [])
    for m in steps:
        out.extend(check_step(m, th))
    sf = result.get("stability_failures", 0)
    if sf:
        out.append(Alert(Level.CRITICAL, "stability_failures", f"{sf} step(s) skipped on non-finite loss"))
    before = (result.get("held_out_before") or {}).get("reward_mean")
    after = (result.get("held_out_after") or {}).get("reward_mean")
    if before is not None and after is not None:
        if after < before - th.reward_regression:
            out.append(Alert(Level.WARNING, "reward_regression", f"held-out reward regressed {before:.3f}→{after:.3f}"))
        elif abs(after - before) < th.reward_stall:
            out.append(Alert(Level.INFO, "reward_stall", f"held-out reward stalled {before:.3f}→{after:.3f} (≈no gain)"))
    return out


def summarize(alerts: list[Alert]) -> dict:
    """Group alerts by rule -> count, worst level, first step, extreme value —
    an incident-response summary rather than a wall of per-step lines."""
    by_rule: dict[str, list[Alert]] = defaultdict(list)
    for a in alerts:
        by_rule[a.rule].append(a)
    summary = {}
    for rule, items in by_rule.items():
        worst = max(items, key=lambda a: _RANK[a.level])
        summary[rule] = {
            "level": worst.level, "count": len(items),
            "first_step": next((a.step for a in items if a.step is not None), None),
            "example": worst.message,
        }
    return summary


def max_level(alerts: list[Alert]) -> str | None:
    return max((a.level for a in alerts), key=lambda l: _RANK[l], default=None)


def _load(args) -> dict:
    if args.result:
        return json.load(open(args.result))
    metrics = [json.loads(l) for l in open(args.metrics) if l.strip()]
    return {"step_metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.observability.alerts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--result", help="a shoprl.rl.run result JSON")
    g.add_argument("--metrics", help="a metrics.jsonl")
    args = ap.parse_args()

    alerts = check_run(_load(args))
    summ = summarize(alerts)
    if not summ:
        print("✓ no alerts — training looks healthy")
        raise SystemExit(0)
    print(f"{len(alerts)} alert(s) across {len(summ)} rule(s):")
    for rule, s in sorted(summ.items(), key=lambda kv: -_RANK[kv[1]['level']]):
        loc = f" (first @step {s['first_step']})" if s["first_step"] is not None else ""
        print(f"  [{s['level'].upper():8}] {rule} ×{s['count']}{loc}: {s['example']}")
    top = max_level(alerts)
    raise SystemExit(2 if top == Level.CRITICAL else 1)


if __name__ == "__main__":
    main()
