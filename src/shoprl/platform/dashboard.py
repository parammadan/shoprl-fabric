"""Pillar 6: the platform dashboard.

Renders a single self-contained HTML file from the platform's REAL persisted
state — job states, trajectory rewards + lineage, the checkpoint registry (with
a live integrity re-check), and recovery events. It reads only what was written
to disk (SQLite stores + jsonl); it never imports the trainer or any RL
internals, so it is a true consumer of the platform, decoupled from it.

If a real RL training metrics.jsonl (from the actual policy-gradient trainer)
is supplied, its REAL kl / entropy / reward are shown alongside the operational
view — never fabricated. When absent, the dashboard says so rather than
inventing a KL curve (the pipeline does no gradient update and thus has no KL).

    python -m shoprl.platform.dashboard --root runs/pipeline \
        [--metrics runs/<exp>/metrics.jsonl] --out runs/pipeline/dashboard.html
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from shoprl.platform.checkpoints import CheckpointCorrupt, CheckpointRegistry
from shoprl.platform.store import JobStore
from shoprl.platform.traj_store import TrajectoryStore


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def collect(root: str | Path, training_metrics: str | None = None) -> dict:
    """Gather everything the dashboard shows, reading persisted state only."""
    root = Path(root)
    store = JobStore(str(root / "jobs.db"))
    traj = TrajectoryStore(str(root / "trajectories.db"))
    registry = CheckpointRegistry(str(root / "checkpoints"))
    try:
        checkpoints = []
        for m in registry.list():
            try:
                registry.verify(m.ckpt_id)
                integrity = "OK"
            except CheckpointCorrupt:
                integrity = "CORRUPT"
            checkpoints.append({
                "ckpt_id": m.ckpt_id, "step": m.step,
                "reward_mean": m.metadata.get("reward_mean"),
                "n_files": len(m.files), "integrity": integrity})
        data = {
            "job_counts": store.counts(),
            "reward_stats": traj.reward_stats(),
            "reward_by_policy": traj.reward_by_policy(),
            "checkpoints": checkpoints,
            "recovery_events": _read_jsonl(root / "recovery_events.jsonl"),
            "pipeline_metrics": _read_jsonl(root / "pipeline_metrics.jsonl"),
            "training_metrics": _read_jsonl(training_metrics) if training_metrics else [],
        }
    finally:
        store.close()
        traj.close()
    return data


def _bar(frac: float, w: int = 160) -> str:
    frac = max(0.0, min(1.0, frac))
    return (f'<span class="bar"><span class="fill" style="width:{frac*w:.0f}px">'
            f'</span></span>')


def render(data: dict, out_path: str | Path) -> Path:
    e = html.escape

    # job state cards
    counts = data["job_counts"]
    state_order = ["pending", "running", "succeeded", "retrying", "failed",
                   "dead_letter", "cancelled"]
    cards = "".join(
        f'<div class="card"><div class="num">{counts.get(s,0)}</div>'
        f'<div class="lbl {s}">{e(s.replace("_"," "))}</div></div>'
        for s in state_order if counts.get(s, 0) or s in ("succeeded", "dead_letter"))

    # reward by policy (per training step) — REAL rule-based reward
    rbp = data["reward_by_policy"]
    rmax = max((m for _, m, _ in rbp), default=1.0) or 1.0
    reward_rows = "".join(
        f"<tr><td>{e(pid)}</td><td>{m:.3f}</td>"
        f"<td>{_bar(m/rmax)}</td><td>{n}</td></tr>"
        for pid, m, n in rbp) or '<tr><td colspan="4">no trajectories</td></tr>'

    # checkpoints + integrity
    ck_rows = ""
    for c in data["checkpoints"]:
        rm = "—" if c["reward_mean"] is None else f"{c['reward_mean']:.3f}"
        cls = "ok" if c["integrity"] == "OK" else "bad"
        ck_rows += (f"<tr><td>{e(c['ckpt_id'])}</td><td>{c['step']}</td>"
                    f"<td>{rm}</td><td>{c['n_files']}</td>"
                    f"<td class=\"{cls}\">{e(c['integrity'])}</td></tr>")
    ck_rows = ck_rows or '<tr><td colspan="5">none</td></tr>'

    # recovery events
    ev_rows = ""
    for ev in data["recovery_events"]:
        mb = (f"{ev.get('microbatch_before')}→{ev.get('microbatch_after')}"
              if ev.get("microbatch_after") is not None else "—")
        sim = "SIMULATED" if ev.get("simulated") else "real"
        ev_rows += (f"<tr><td class=\"cls\">{e(ev.get('failure_class',''))}</td>"
                    f"<td>{e(ev.get('action',''))}</td><td>{e(mb)}</td>"
                    f"<td>{e(str(ev.get('restored_ckpt') or '—'))}</td>"
                    f"<td>{e(ev.get('resulting_state',''))}</td>"
                    f"<td class=\"sim\">{sim}</td></tr>")
    ev_rows = ev_rows or '<tr><td colspan="6">no recovery events</td></tr>'

    # optional REAL training metrics (KL/entropy) — never fabricated
    tm = data["training_metrics"]
    if tm:
        def col(k):
            return [r[k] for r in tm if k in r and r[k] is not None]
        kl, ent, rew = col("kl"), col("entropy"), col("reward_mean")
        kl_block = (
            '<table><tr><th>metric</th><th>last</th><th>max</th></tr>'
            + (f'<tr><td>KL</td><td>{kl[-1]:.4f}</td><td>{max(kl):.4f}</td></tr>' if kl else '')
            + (f'<tr><td>entropy</td><td>{ent[-1]:.3f}</td><td>{max(ent):.3f}</td></tr>' if ent else '')
            + (f'<tr><td>reward</td><td>{rew[-1]:+.3f}</td><td>{max(rew):+.3f}</td></tr>' if rew else '')
            + '</table>')
        kl_note = f"Real RL-trainer metrics over {len(tm)} steps."
    else:
        kl_block = ""
        kl_note = ("No training-metrics file supplied. KL / entropy come from "
                   "the RL trainer (metrics.jsonl); this pipeline does no "
                   "gradient update, so it produces reward + operational state "
                   "only — not fabricated KL.")

    rs = data["reward_stats"]
    rs_line = (f"{rs['count']} trajectories · reward min {rs['min']:.3f} / "
               f"mean {rs['mean']:.3f} / max {rs['max']:.3f}" if rs.get("count")
               else "no trajectories")

    doc = _TEMPLATE.format(
        cards=cards, reward_rows=reward_rows, ck_rows=ck_rows, ev_rows=ev_rows,
        kl_block=kl_block, kl_note=e(kl_note), rs_line=e(rs_line))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ShopRL Fabric — platform dashboard</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;
   background:#0f1216;color:#e6e9ef}}
 header{{padding:18px 24px;border-bottom:1px solid #232a33}}
 h1{{font-size:18px;margin:0}} .sub{{color:#8b95a5;font-size:12px;margin-top:4px}}
 main{{padding:20px 24px;max-width:1000px}}
 section{{margin-bottom:26px}} h2{{font-size:13px;text-transform:uppercase;
   letter-spacing:.06em;color:#8b95a5;border-bottom:1px solid #232a33;
   padding-bottom:6px}}
 .cards{{display:flex;gap:10px;flex-wrap:wrap}}
 .card{{background:#161b22;border:1px solid #232a33;border-radius:8px;
   padding:12px 16px;min-width:96px}}
 .num{{font-size:24px;font-weight:600}} .lbl{{font-size:11px;color:#8b95a5;
   text-transform:uppercase;letter-spacing:.04em;margin-top:2px}}
 .lbl.succeeded{{color:#4cc38a}} .lbl.dead_letter,.lbl.failed{{color:#e5484d}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #1c232b}}
 th{{color:#8b95a5;font-weight:500}}
 .bar{{display:inline-block;width:160px;height:9px;background:#1c232b;
   border-radius:5px;vertical-align:middle}}
 .fill{{display:inline-block;height:9px;background:#4c8bf5;border-radius:5px}}
 .ok{{color:#4cc38a}} .bad{{color:#e5484d;font-weight:600}}
 .cls{{color:#e2a03f;font-weight:600}} .sim{{color:#8b95a5;font-size:11px}}
 .note{{color:#8b95a5;font-size:12px;margin-top:8px}}
</style></head><body>
<header><h1>ShopRL Fabric — platform dashboard</h1>
<div class="sub">Real persisted state: jobs · trajectories · checkpoints · recovery.
A read-only consumer — no trainer internals imported.</div></header>
<main>
 <section><h2>Job states</h2><div class="cards">{cards}</div></section>
 <section><h2>Reward per policy version (real rule-based reward)</h2>
  <div class="note">{rs_line}</div>
  <table><tr><th>policy</th><th>mean reward</th><th></th><th>n</th></tr>
  {reward_rows}</table></section>
 <section><h2>Checkpoint registry (live integrity re-check)</h2>
  <table><tr><th>checkpoint</th><th>step</th><th>reward_mean</th>
  <th>files</th><th>integrity</th></tr>{ck_rows}</table></section>
 <section><h2>Failure recovery events</h2>
  <table><tr><th>class</th><th>action</th><th>microbatch</th><th>restored</th>
  <th>result</th><th>trigger</th></tr>{ev_rows}</table></section>
 <section><h2>RL training metrics (KL / entropy)</h2>{kl_block}
  <div class="note">{kl_note}</div></section>
</main></body></html>"""


def build(root: str | Path, out_path: str | None = None,
          training_metrics: str | None = None) -> Path:
    data = collect(root, training_metrics)
    out = out_path or str(Path(root) / "dashboard.html")
    return render(data, out)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Render the ShopRL platform dashboard")
    ap.add_argument("--root", required=True, help="pipeline run dir")
    ap.add_argument("--metrics", help="optional real RL trainer metrics.jsonl")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = build(args.root, args.out, args.metrics)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
