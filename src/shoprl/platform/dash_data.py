"""Read-only data access + DEV-mode fault injection for the live dashboard.

Everything here reads the platform's PERSISTED state (SQLite stores, the
checkpoint registry, jsonl event/metric files) or produces new REAL state by
driving the platform's own APIs. It imports no trainer internals and fabricates
no metric — if something isn't recorded, callers surface "absent" rather than
invent it.

The `sim_*` functions are the confirmation-gated DEV fault controls. They are
real operations on the real stores (that's the point of fault injection) but the
*fault* is simulated and labelled: killing a worker = expiring a lease and
reaping it; OOM = raising SimulatedOOM through the recovery controller;
duplicate = deriving a child trajectory. Never wire these into a production path.
"""
from __future__ import annotations

import json
from pathlib import Path

from shoprl.platform.checkpoints import CheckpointCorrupt, CheckpointRegistry
from shoprl.platform.failures import RecoveryController, SimulatedOOM
from shoprl.platform.store import JobStore
from shoprl.platform.traj_store import TrajectoryStore


def paths(root: str | Path) -> dict:
    root = Path(root)
    return {
        "root": str(root),
        "jobs_db": str(root / "jobs.db"),
        "traj_db": str(root / "trajectories.db"),
        "ckpt_root": str(root / "checkpoints"),
        "events_path": str(root / "recovery_events.jsonl"),
    }


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


# --- reads -----------------------------------------------------------------
def snapshot(root: str | Path, training_metrics: str | None = None) -> dict:
    """The operational snapshot, read from persisted state only: job states,
    reward-per-policy, checkpoints (with a LIVE sha256 integrity re-check),
    recovery events, pipeline metrics, and optional real RL training metrics.
    Never imports trainer internals; absent metrics stay absent."""
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
        return {
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


def comparisons(results_dir: str | Path) -> list[dict]:
    """Real RLOO/GRPO/PPO comparison results written by `shoprl.rl.run`
    (results/*.json). Empty list if none exist — the UI then shows 'absent'
    instead of inventing a KL comparison."""
    d = Path(results_dir)
    if not d.exists():
        return []
    import json
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            r = json.loads(f.read_text())
            if "algorithm" in r and "final_kl" in r:
                r["_source"] = f.name
                out.append(r)
        except (ValueError, OSError):
            continue
    return out


def trajectories(root: str | Path, limit: int = 500) -> list:
    p = paths(root)
    if not Path(p["traj_db"]).exists():
        return []
    ts = TrajectoryStore(p["traj_db"])
    try:
        return ts.recent(limit)
    finally:
        ts.close()


def trajectory_detail(root: str | Path, traj_id: str) -> dict:
    """The single-trajectory ('#824') view: prompt, response, reward
    components, total reward, policy version, advantage, and lineage. KL is
    reported as absent because it is a per-step training metric, not a
    per-trajectory value."""
    p = paths(root)
    ts = TrajectoryStore(p["traj_db"])
    try:
        t = ts.get(traj_id)
        ancestry = [a.id for a in ts.ancestry(traj_id)]
    finally:
        ts.close()
    meta = t.meta or {}
    return {
        "id": t.id,
        "kind": t.kind,
        "prompt": t.prompt,
        "response": "\n".join(s.action for s in t.steps),
        "total_reward": t.reward,
        "reward_components": meta.get("reward_components"),   # real or None
        "advantage": meta.get("advantage"),                   # real group-relative or None
        "group_mean": meta.get("group_mean"),
        "group_std": meta.get("group_std"),
        "kl": meta.get("kl"),                                 # absent by design (None)
        "policy_id": t.lineage.policy_id,
        "job_id": t.lineage.job_id,
        "prompt_id": t.lineage.prompt_id,
        "seed": t.lineage.seed,
        "parent_id": t.lineage.parent_id,
        "ancestry": ancestry,
        "num_steps": t.num_steps,
    }


# --- DEV-mode fault injection (SIMULATION) ---------------------------------
def sim_kill_worker(root: str | Path) -> dict:
    """SIMULATION: enqueue+claim a job (as if a worker picked it up), then let
    its lease expire and reap it — exactly what happens when a worker process
    dies mid-job. The reaper requeues it. Real recovery logic; simulated death."""
    p = paths(root)
    s = JobStore(p["jobs_db"])
    try:
        job = s.create("sim-rollout", {"note": "SIMULATION: killed-worker demo"})
        s.claim(kinds=["sim-rollout"], lease_seconds=1.0, now=0.0)  # worker claims
        reaped = s.reap_expired(now=10_000.0)                        # lease expired -> reap
        after = s.get(job.id)
        return {"ok": True, "job_id": job.id, "reaped": len(reaped),
                "resulting_state": after.state.value, "attempts": after.attempts,
                "label": "SIMULATION"}
    finally:
        s.close()


def sim_oom(root: str | Path) -> dict:
    """SIMULATION: raise SimulatedOOM through the recovery controller on a
    freshly claimed optimize job. Produces a REAL RecoveryEvent + batch shrink
    (+ checkpoint restore if one exists) + requeue."""
    p = paths(root)
    s = JobStore(p["jobs_db"])
    reg = CheckpointRegistry(p["ckpt_root"])
    try:
        job = s.create("optimize", {"microbatch_size": 8, "grad_accum_steps": 1,
                                    "note": "SIMULATION: OOM demo"})
        s.claim(kinds=["optimize"], now=0.0)
        ctl = RecoveryController(s, registry=reg, events_path=p["events_path"])
        ev = ctl.handle(s.get(job.id), SimulatedOOM("cuda oom (SIMULATION)"))
        return {"ok": True, "job_id": job.id, "action": ev.action,
                "microbatch": f"{ev.microbatch_before}->{ev.microbatch_after}",
                "restored_ckpt": ev.restored_ckpt,
                "resulting_state": ev.resulting_state,
                "simulated": ev.simulated, "label": "SIMULATION"}
    finally:
        s.close()


def sim_corrupt_checkpoint(root: str | Path) -> dict:
    """DEV: flip a byte in the latest checkpoint's data file so the registry's
    sha256 verify() detects corruption. The corruption is real (and the detection
    is real); we deliberately tamper to demonstrate integrity checking. The
    Checkpoints tab's live re-check will show it as CORRUPT."""
    p = paths(root)
    reg = CheckpointRegistry(p["ckpt_root"])
    latest = reg.latest()
    if latest is None:
        return {"ok": False, "error": "no checkpoint to corrupt"}
    victim = next((e for e in latest.files if not e.path.endswith("manifest.json")),
                  latest.files[0])
    fpath = Path(p["ckpt_root"]) / latest.ckpt_id / victim.path
    data = bytearray(fpath.read_bytes())
    data[0] ^= 0xFF                                   # flip one byte
    fpath.write_bytes(bytes(data))
    try:
        reg.verify(latest.ckpt_id)
        status = "OK (unexpected!)"
    except CheckpointCorrupt:
        status = "CORRUPT"
    return {"ok": True, "ckpt_id": latest.ckpt_id, "file": victim.path,
            "integrity": status, "label": "DEV/SIMULATION"}


def sim_duplicate_trajectory(root: str | Path, traj_id: str | None = None) -> dict:
    """SIMULATION: duplicate a trajectory (redelivery). Uses derive() so the
    copy is lineage-linked to its parent — demonstrating the provenance graph
    and the idempotency story (a real dedup would reject a same-id redelivery)."""
    p = paths(root)
    ts = TrajectoryStore(p["traj_db"])
    try:
        if traj_id is None:
            recent = ts.recent(1)
            if not recent:
                return {"ok": False, "error": "no trajectories to duplicate"}
            parent = recent[0]
        else:
            parent = ts.get(traj_id)
        child = parent.derive()                 # new id, parent_id -> parent
        ts.put(child)
        return {"ok": True, "parent_id": parent.id, "duplicate_id": child.id,
                "label": "SIMULATION"}
    finally:
        ts.close()
