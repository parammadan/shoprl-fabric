"""ShopRL Fabric — LIVE platform dashboard (Streamlit).

A running app (not static HTML) over the platform's PERSISTED state: the SQLite
job store, trajectory store, checkpoint registry, recovery-event log, and the RL
trainer's metrics/comparison files. It reads real data on every refresh and
never imports trainer internals or fabricates a metric — anything not recorded
is shown as absent.

Run:
    .venv/bin/streamlit run src/shoprl/platform/streamlit_app.py
    # then set the "Run root" in the sidebar to a pipeline run dir, e.g.
    # runs/pipeline (produce one with: python -m shoprl.platform.pipeline
    # --root runs/pipeline)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from shoprl.platform import dash_data

st.set_page_config(page_title="ShopRL Fabric — live platform dashboard",
                   layout="wide")

# Job states grouped by pillar-stage, in lifecycle order.
STAGE_ORDER = ["pending", "running", "retrying", "succeeded",
               "failed", "dead_letter", "cancelled"]
GOOD, BAD = {"succeeded"}, {"failed", "dead_letter"}


# --- sidebar: config + honest scope + DEV fault controls -------------------
def sidebar() -> dict:
    st.sidebar.title("ShopRL Fabric")
    st.sidebar.caption(
        "Live, read-only view of real persisted platform state. Single machine; "
        "distributed behaviour is local processes, documented as a scale-out "
        "design — not a multi-node claim.")

    root = st.sidebar.text_input("Run root", value="runs/pipeline")
    results_dir = st.sidebar.text_input("RL results dir (comparison JSONs)",
                                        value="results")
    metrics = st.sidebar.text_input(
        "RL trainer metrics.jsonl (optional, for KL/entropy)", value="")
    live = st.sidebar.toggle("Live auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh every (s)", 2, 30, 3, disabled=not live)

    st.sidebar.divider()
    with st.sidebar.expander("⚠️ DEV MODE — fault injection (SIMULATION)"):
        st.caption("Every control below is a SIMULATION acting on the real "
                   "stores. Confirm each before running.")
        _fault_controls(root)

    return {"root": root, "results_dir": results_dir,
            "metrics": metrics or None, "live": live, "interval": interval}


def _fault_controls(root: str) -> None:
    # kill worker
    if st.checkbox("Confirm: simulate a killed worker", key="c_kill"):
        if st.button("💥 Kill worker (SIMULATION)", key="b_kill"):
            r = dash_data.sim_kill_worker(root)
            st.success(f"SIMULATION: worker died, job reaped -> "
                       f"{r['resulting_state']} (attempts {r['attempts']}).")
    # OOM
    if st.checkbox("Confirm: simulate an OOM", key="c_oom"):
        if st.button("🧨 Trigger OOM (SIMULATION)", key="b_oom"):
            r = dash_data.sim_oom(root)
            st.success(f"SIMULATION: OOM handled -> {r['action']}, microbatch "
                       f"{r['microbatch']}, restore {r['restored_ckpt']}.")
    # duplicate trajectory
    if st.checkbox("Confirm: duplicate a trajectory", key="c_dup"):
        if st.button("🧬 Duplicate latest trajectory (SIMULATION)", key="b_dup"):
            r = dash_data.sim_duplicate_trajectory(root)
            if r.get("ok"):
                st.success(f"SIMULATION: duplicated {r['parent_id'][:8]} -> "
                           f"{r['duplicate_id'][:8]} (lineage-linked).")
            else:
                st.warning(r.get("error", "nothing to duplicate"))


# --- panels ----------------------------------------------------------------
def jobs_panel(root: str) -> None:
    counts = dash_data.snapshot(root)["job_counts"]
    total = sum(counts.values()) or 0
    st.subheader("Job states")
    st.caption(f"{total} jobs · live from {Path(root) / 'jobs.db'}")
    cols = st.columns(len(STAGE_ORDER))
    for col, s in zip(cols, STAGE_ORDER):
        n = counts.get(s, 0)
        col.metric(s.replace("_", " ").title(), n,
                   delta="ok" if s in GOOD and n else
                         ("!" if s in BAD and n else None),
                   delta_color="normal" if s in GOOD else "inverse")
    if total:
        df = pd.DataFrame({"state": STAGE_ORDER,
                           "jobs": [counts.get(s, 0) for s in STAGE_ORDER]})
        st.bar_chart(df.set_index("state"), horizontal=True, height=240)


def reward_kl_panel(cfg: dict) -> None:
    snap = dash_data.snapshot(cfg["root"], cfg["metrics"])
    st.subheader("Reward per policy version (real rule-based reward)")
    rbp = snap["reward_by_policy"]
    if rbp:
        df = pd.DataFrame(rbp, columns=["policy", "mean_reward", "n"])
        st.bar_chart(df.set_index("policy")["mean_reward"], height=240)
        st.caption(f"{snap['reward_stats'].get('count', 0)} trajectories scored.")
    else:
        st.info("No trajectories yet. Run the pipeline to populate.")

    st.divider()
    st.subheader("RLOO / GRPO / PPO — KL stability comparison (real runs)")
    comps = dash_data.comparisons(cfg["results_dir"])
    if comps:
        tbl = pd.DataFrame([{
            "algorithm": c["algorithm"], "final_kl": c.get("final_kl"),
            "max_kl": c.get("max_kl"), "reward_gain": c.get("reward_gain"),
            "stability_failures": c.get("stability_failures"),
            "source": c.get("_source")} for c in comps])
        st.dataframe(tbl, use_container_width=True, hide_index=True)
        st.bar_chart(tbl.set_index("algorithm")[["final_kl", "max_kl"]], height=260)
        # KL trajectories over training steps, if present
        curves = {c["algorithm"]: c.get("kl_trajectory") for c in comps
                  if c.get("kl_trajectory")}
        if curves:
            maxlen = max(len(v) for v in curves.values())
            cdf = pd.DataFrame({k: pd.Series(v) for k, v in curves.items()},
                               index=range(maxlen))
            st.line_chart(cdf, height=260)
            st.caption("Per-step KL trajectory — measured, not fabricated.")
    else:
        st.warning(
            f"No comparison runs found in `{cfg['results_dir']}/*.json`. "
            "Produce them with e.g. `python -m shoprl.rl.run "
            "--config configs/compare_rloo.yaml --out results/rloo.json` "
            "(needs a GPU). Nothing is invented here.")

    tm = snap["training_metrics"]
    st.divider()
    st.subheader("RL trainer KL / entropy (metrics.jsonl)")
    if tm:
        keys = [k for k in ("kl", "entropy", "reward_mean") if any(k in r for r in tm)]
        mdf = pd.DataFrame([{k: r.get(k) for k in keys} for r in tm])
        st.line_chart(mdf, height=260)
    else:
        st.info("No metrics.jsonl supplied. KL/entropy come from the RL trainer; "
                "the pipeline does no gradient update, so it has none — shown as "
                "absent, not invented.")


def checkpoints_panel(root: str) -> None:
    st.subheader("Checkpoint registry (integrity verified on load)")
    cks = dash_data.snapshot(root)["checkpoints"]
    if not cks:
        st.info("No checkpoints yet.")
        return
    df = pd.DataFrame(cks)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"integrity": st.column_config.TextColumn("integrity")})
    bad = [c["ckpt_id"] for c in cks if c["integrity"] != "OK"]
    if bad:
        st.error(f"CORRUPT checkpoints detected: {', '.join(bad)}")
    else:
        st.success("All checkpoints pass sha256 verification.")


def recovery_panel(root: str) -> None:
    st.subheader("Failure recovery events")
    st.caption("OOM events are triggered by SimulatedOOM and labelled — a laptop "
               "cannot produce a real CUDA OOM. Recovery logic is real.")
    evs = dash_data.snapshot(root)["recovery_events"]
    if not evs:
        st.info("No recovery events yet. Trigger one from DEV MODE in the sidebar.")
        return
    rows = [{
        "class": e.get("failure_class"), "action": e.get("action"),
        "microbatch": (f"{e.get('microbatch_before')}->{e.get('microbatch_after')}"
                       if e.get("microbatch_after") is not None else "—"),
        "restored": e.get("restored_ckpt") or "—",
        "result": e.get("resulting_state"),
        "trigger": "SIMULATED" if e.get("simulated") else "real",
        "gpu_mem_gb": e.get("gpu_mem_gb"),
    } for e in evs]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def trajectory_panel(root: str) -> None:
    st.subheader("Trajectory explorer")
    trajs = dash_data.trajectories(root, limit=500)
    if not trajs:
        st.info("No trajectories yet. Run the pipeline.")
        return
    labels = {f"{t.id[:8]} · {t.lineage.policy_id} · r={t.reward:.3f}"
              if t.reward is not None else f"{t.id[:8]} · {t.lineage.policy_id}": t.id
              for t in trajs}
    pick = st.selectbox(f"Select a trajectory ({len(trajs)} recent)",
                        list(labels.keys()))
    d = dash_data.trajectory_detail(root, labels[pick])

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Trajectory** `{d['id']}`  ·  policy `{d['policy_id']}`")
        st.markdown("**Prompt**")
        st.code(d["prompt"] or "—", language=None)
        st.markdown("**Response**")
        st.code(d["response"] or "—", language=None)
    with right:
        st.metric("Total reward",
                  "—" if d["total_reward"] is None else f"{d['total_reward']:.3f}")
        st.metric("Advantage (group-relative)",
                  "N/A" if d["advantage"] is None else f"{d['advantage']:+.3f}")
        st.metric("KL", "N/A")
        st.caption("KL is a per-step training metric, not stored per trajectory — "
                   "shown as N/A rather than invented.")

    st.markdown("**Reward components** (real, from `compute_reward`)")
    rc = d["reward_components"]
    if rc:
        comp = {k: v for k, v in rc.items() if isinstance(v, (int, float))}
        st.bar_chart(pd.Series(comp), height=220)
    else:
        st.info("Reward components not recorded for this trajectory.")

    st.markdown("**Lineage / provenance**")
    st.json({"policy_id": d["policy_id"], "job_id": d["job_id"],
             "prompt_id": d["prompt_id"], "seed": d["seed"],
             "parent_id": d["parent_id"], "ancestry (root→this)": d["ancestry"],
             "group_mean": d["group_mean"], "group_std": d["group_std"]})


# --- main ------------------------------------------------------------------
def main() -> None:
    cfg = sidebar()
    root = cfg["root"]
    st.title("ShopRL Fabric — live platform dashboard")

    if not Path(dash_data.paths(root)["jobs_db"]).exists():
        st.warning(
            f"No pipeline run found at `{root}`. Create one:\n\n"
            "```\npython -m shoprl.platform.pipeline --root " + root + "\n```")
        return

    # Live-refreshing job states (fragment reruns itself on `interval`).
    @st.fragment(run_every=cfg["interval"] if cfg["live"] else None)
    def _live_jobs():
        jobs_panel(root)
        st.caption("🟢 live" if cfg["live"] else "⏸ paused — toggle live in sidebar")
    _live_jobs()

    st.divider()
    tabs = st.tabs(["Reward & KL", "Checkpoints", "Recovery events",
                    "Trajectory explorer"])
    with tabs[0]:
        reward_kl_panel(cfg)
    with tabs[1]:
        checkpoints_panel(root)
    with tabs[2]:
        recovery_panel(root)
    with tabs[3]:
        trajectory_panel(root)


main()
