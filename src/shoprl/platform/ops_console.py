"""ShopRL Fabric — live operations console (Streamlit, API-driven).

Unlike the earlier dashboard (which read the stores directly), this console
talks ONLY to the platform HTTP API via ApiClient — so it depends on the API
contract, not on Python internals, and could run on a different machine from the
platform. It shows jobs, the scheduler, policy versions + staleness, checkpoints
(with integrity), experiments + run comparison, RL metrics (reward / KL /
entropy / grad-norm) with active alerts, recovery events, and a trajectory
explorer. Dev-only controls (pause / resume / cancel / kill-worker / replay) are
confirmation-gated and labelled SIMULATION where they inject a fault.

Run:
    # option A — against a running server:
    .venv/bin/uvicorn shoprl.platform.api:app          # SHOPRL_ROOT=runs/pipeline
    .venv/bin/streamlit run src/shoprl/platform/ops_console.py
    # option B — embedded (in-process API, no server): default; set Run root in
    # the sidebar to a pipeline run dir.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from shoprl.platform.api_client import ApiClient

st.set_page_config(page_title="ShopRL Fabric — ops console", layout="wide")
STAGES = ["pending", "running", "paused", "retrying", "succeeded",
          "failed", "dead_letter", "cancelled"]


def get_client() -> ApiClient:
    mode = st.session_state.get("mode", "embedded")
    if mode == "embedded":
        root = st.session_state.get("root", os.environ.get("SHOPRL_ROOT", "runs/pipeline"))
        runs = st.session_state.get("runs", os.environ.get("SHOPRL_RUNS", "runs"))
        return ApiClient.in_process(root, runs)
    return ApiClient(base_url=st.session_state.get("base_url", "http://127.0.0.1:8000"))


def sidebar() -> dict:
    st.sidebar.title("ShopRL Fabric ops")
    st.sidebar.caption("API-driven. The console never touches the stores directly "
                       "— only the platform HTTP API.")
    mode = st.sidebar.radio("API connection", ["embedded", "http"],
                            help="embedded = in-process ASGI (same API, no server)")
    st.session_state["mode"] = mode
    if mode == "embedded":
        default_root = (st.session_state.get("root")
                        or os.environ.get("SHOPRL_ROOT", "runs/pipeline"))
        default_runs = (st.session_state.get("runs")
                        or os.environ.get("SHOPRL_RUNS", "runs"))
        st.session_state["root"] = st.sidebar.text_input("Run root", value=default_root)
        st.session_state["runs"] = st.sidebar.text_input("Runs dir", value=default_runs)
    else:
        st.session_state["base_url"] = st.sidebar.text_input(
            "API base URL", value="http://127.0.0.1:8000")
    live = st.sidebar.toggle("Live auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh every (s)", 2, 30, 4, disabled=not live)
    st.sidebar.divider()
    with st.sidebar.expander("⚠️ DEV MODE — controls (SIMULATION where noted)"):
        _dev_controls()
    return {"live": live, "interval": interval}


def _dev_controls() -> None:
    api = get_client()
    jid = st.text_input("Job id (pause/resume/cancel)")
    c1, c2, c3 = st.columns(3)
    if c1.button("Pause", disabled=not jid):
        _try(lambda: api.pause(jid), "paused")
    if c2.button("Resume", disabled=not jid):
        _try(lambda: api.resume(jid), "resumed")
    if c3.button("Cancel", disabled=not jid):
        _try(lambda: api.cancel(jid), "cancelled")
    st.divider()
    if st.checkbox("Confirm: kill a worker (SIMULATION)"):
        if st.button("💥 Kill worker"):
            r = _try(api.kill_worker, "worker killed")
            if r:
                st.info(f"SIMULATION: reaped -> {r['resulting_state']}")
    tid = st.text_input("Trajectory id to replay (SIMULATION)")
    if tid and st.button("🧬 Replay trajectory"):
        r = _try(lambda: api.replay(tid), "replayed")
        if r:
            st.info(f"SIMULATION: duplicate {r['duplicate_id'][:8]}")


def _try(fn, ok_msg):
    try:
        r = fn()
        st.success(ok_msg)
        return r
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        return None


# --- panels ----------------------------------------------------------------
def overview_panel(api: ApiClient) -> None:
    ov = api.overview()
    counts = ov["job_counts"]
    st.subheader("Jobs")
    cols = st.columns(len(STAGES))
    for col, s in zip(cols, STAGES):
        col.metric(s.replace("_", " ").title(), counts.get(s, 0))
    rs = ov["reward_stats"]
    if rs.get("count"):
        st.caption(f"{rs['count']} trajectories · reward mean {rs['mean']:.3f} "
                   f"(min {rs['min']:.3f} / max {rs['max']:.3f})")


def recovery_panel(api: ApiClient) -> None:
    st.subheader("Recovery events")
    st.caption("OOM is triggered by SimulatedOOM and labelled — a laptop cannot "
               "produce a real CUDA OOM. Recovery logic is real.")
    evs = api.overview()["recovery_events"]
    if not evs:
        st.info("No recovery events. Trigger one from DEV MODE in the sidebar.")
        return
    st.dataframe(pd.DataFrame([{
        "class": e.get("failure_class"), "action": e.get("action"),
        "microbatch": (f"{e.get('microbatch_before')}->{e.get('microbatch_after')}"
                       if e.get("microbatch_after") is not None else "—"),
        "restored": e.get("restored_ckpt") or "—",
        "result": e.get("resulting_state"),
        "trigger": "SIMULATED" if e.get("simulated") else "real"} for e in evs]),
        hide_index=True, use_container_width=True)


def training_health_panel(api: ApiClient) -> None:
    st.subheader("Training Health")
    st.caption("Run- and training-step-level metrics (KL is never per-trajectory). "
               "Comparison overlays are HISTORICAL committed artifacts; the "
               "single-run section is the most recent persisted run.")

    # --- PPO / GRPO / RLOO comparison overlays (historical) ---
    st.markdown("### PPO / GRPO / RLOO comparison — historical (committed artifacts)")
    comps = api.comparisons()
    if comps:
        tbl = pd.DataFrame([{
            "algorithm": c["algorithm"], "final_kl": c["final_kl"], "max_kl": c["max_kl"],
            "reward_gain": c["reward_gain"], "stability_failures": c["stability_failures"],
            "critical_alerts": c["alerts"]["critical"], "source": c["source"]}
            for c in comps])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        kl = {c["algorithm"]: c["final_kl"] for c in comps}
        if all(k in kl for k in ("rloo", "grpo", "ppo")):
            st.caption(f"Measured final KL — RLOO {kl['rloo']:.3f} ≪ "
                       f"GRPO {kl['grpo']:.3f} ≪ PPO {kl['ppo']:.2f}")
        ppo = next((c for c in comps if c["algorithm"] == "ppo"), None)
        if ppo and ppo["alerts"]["critical"]:
            st.error(f"PPO critical KL alerts: {ppo['alerts']['critical']} "
                     f"({ppo['alerts']['by_rule'].get('kl_blowup', 0)} kl_blowup) "
                     "— from persisted step_metrics")
        for metric, label in [("kl", "KL vs reference"), ("entropy", "Entropy"),
                              ("clip_frac", "Clip fraction"), ("grad_norm", "Grad norm"),
                              ("reward_mean", "Reward mean")]:
            series = {c["algorithm"]: pd.Series([m.get(metric) for m in c["step_metrics"]])
                      for c in comps if any(m.get(metric) is not None for m in c["step_metrics"])}
            if series:
                st.markdown(f"**{label}** (per training step)")
                st.line_chart(pd.DataFrame(series), height=220)
    else:
        st.info("No comparison artifacts found (comparisons/*.json). Nothing invented.")

    st.divider()
    # --- single-run health (most recent persisted metrics.jsonl) ---
    st.markdown("### Single run — most recent persisted metrics")
    runs = api.metrics_runs()
    if not runs:
        st.info("No run metrics.jsonl found under the runs dir.")
        return
    sel = st.selectbox("Run (from metrics.jsonl)", runs, key="th_run")
    m = api.run_metrics(sel)
    rows = m["metrics"] if m else []
    if rows:
        df = pd.DataFrame(rows)
        if "reward_mean" in df:
            band = pd.DataFrame({"reward_mean": df["reward_mean"]})
            if "reward_std" in df:
                band["reward_+std"] = df["reward_mean"] + df["reward_std"]
                band["reward_-std"] = df["reward_mean"] - df["reward_std"]
            st.markdown("**Reward mean ± std**")
            st.line_chart(band, height=220)
        singles = [(k, lbl) for k, lbl in [("kl", "KL vs reference"), ("entropy", "Entropy"),
                   ("clip_frac", "Clip fraction"), ("grad_norm", "Grad norm")] if k in df]
        for k, lbl in singles:
            st.markdown(f"**{lbl}**")
            st.line_chart(df[[k]], height=200)
        comp_keys = [k for k in ("reward_budget", "reward_groundedness", "reward_coverage",
                     "reward_quality_format", "reward_quality_comparison") if k in df]
        if comp_keys:
            st.markdown("**Per-component rewards**")
            st.line_chart(df[comp_keys], height=220)
    al = api.run_alerts(sel)
    st.markdown("**Alerts (active + historical for this run)**")
    if al and al["n_alerts"]:
        st.error(f"{al['n_alerts']} alert(s) · max level {al['max_level']}")
        st.dataframe(pd.DataFrame(al["alerts"]), hide_index=True, use_container_width=True)
    elif al is not None:
        st.success("No alerts for this run.")


def scheduler_panel(api: ApiClient) -> None:
    st.subheader("Scheduler")
    s = api.scheduler()
    c1, c2, c3 = st.columns(3)
    c1.metric("GPU", f"{s['gpu']['used']}/{s['gpu']['slots']}",
              delta=f"{s['gpu']['queued']} queued" or None)
    c2.metric("CPU workers", f"{s['cpu']['used']}/{s['cpu']['slots']}",
              delta=f"{s['cpu']['queued']} queued")
    c3.metric("Running / max", f"{s['running_total']}/{s['max_concurrent']}")
    if s["at_capacity"]:
        st.warning("At capacity — new jobs will queue.")


def jobs_panel(api: ApiClient) -> None:
    st.subheader("Jobs")
    jobs = api.jobs()
    if not jobs:
        st.info("No jobs.")
        return
    st.dataframe(pd.DataFrame([{
        "id": j["id"][:12], "kind": j["kind"], "state": j["state"],
        "resource": j["resource"], "priority": j["priority"],
        "attempts": j["attempts"]} for j in jobs]),
        hide_index=True, use_container_width=True)


def policies_panel(api: ApiClient) -> None:
    st.subheader("Policy versions")
    pols = api.policies()
    if not pols:
        st.info("No policies published.")
        return
    st.dataframe(pd.DataFrame([{"version": p["version"], "fingerprint": p["fingerprint"]}
                              for p in pols]), hide_index=True, use_container_width=True)
    stale = api.policy_staleness()
    if stale and stale.get("n"):
        st.metric("Rollout staleness (max behind latest)", stale.get("max_staleness"))
        st.caption(f"current v{stale['current_version']} · "
                   f"{stale.get('on_policy_count', 0)} on-policy / "
                   f"{stale.get('stale_count', 0)} stale of {stale['n']} trajectories")


def experiments_panel(api: ApiClient) -> None:
    st.subheader("Experiments")
    runs = api.runs()
    if not runs:
        st.info("No registered runs.")
        return
    st.dataframe(pd.DataFrame([{
        "run_id": r["run_id"], "algorithm": r["algorithm"], "status": r["status"],
        "config_hash": r["config_hash"], "policy_version": r.get("policy_version"),
        "final_kl": (r.get("eval_result") or {}).get("final_kl")} for r in runs]),
        hide_index=True, use_container_width=True)
    ids = [r["run_id"] for r in runs]
    picked = st.multiselect("Compare runs", ids, default=ids[:min(3, len(ids))])
    if len(picked) >= 2:
        cmp = api.compare_runs(picked)
        st.caption("comparable ✅" if cmp["comparable"] else
                   "⚠️ NOT comparable — dataset/reward versions differ")
        st.dataframe(pd.DataFrame(cmp["rows"]), hide_index=True, use_container_width=True)
    # RL metrics + alerts for a selected run
    sel = st.selectbox("Run metrics", ids)
    m = api.run_metrics(sel)
    if m and m["metrics"]:
        keys = [k for k in ("reward_mean", "kl", "entropy", "grad_norm")
                if any(k in row for row in m["metrics"])]
        if keys:
            st.line_chart(pd.DataFrame([{k: row.get(k) for k in keys}
                                        for row in m["metrics"]]), height=260)
    else:
        st.info(f"No metrics.jsonl for run '{sel}'.")
    al = api.run_alerts(sel)
    if al and al["n_alerts"]:
        st.error(f"{al['n_alerts']} active alert(s) · max level {al['max_level']}")
        st.dataframe(pd.DataFrame(al["alerts"]), hide_index=True, use_container_width=True)
    elif al is not None:
        st.success("No active alerts.")


def checkpoints_panel(api: ApiClient) -> None:
    st.subheader("Checkpoints (integrity verified on load)")
    cks = api.checkpoints()
    if not cks:
        st.info("No checkpoints.")
        return
    st.dataframe(pd.DataFrame(cks), hide_index=True, use_container_width=True)
    bad = [c["ckpt_id"] for c in cks if c["integrity"] != "OK"]
    (st.error if bad else st.success)(
        f"CORRUPT: {bad}" if bad else "All checkpoints verify OK.")


def trajectory_panel(api: ApiClient) -> None:
    st.subheader("Trajectory explorer")
    trajs = api.trajectories(limit=300)
    if not trajs:
        st.info("No trajectories.")
        return
    labels = {f"{t['id'][:8]} · {t['policy_id']} · "
              f"r={t['reward']:.3f}" if t["reward"] is not None
              else f"{t['id'][:8]} · {t['policy_id']}": t["id"] for t in trajs}
    pick = st.selectbox(f"Select ({len(trajs)})", list(labels.keys()))
    d = api.trajectory(labels[pick])
    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**Prompt**"); st.code(d["prompt"] or "—")
        st.markdown(f"**Response**"); st.code(d["response"] or "—")
    with right:
        st.metric("Total reward", "—" if d["total_reward"] is None else f"{d['total_reward']:.3f}")
        st.metric("Advantage", "N/A" if d["advantage"] is None else f"{d['advantage']:+.3f}")
        st.metric("KL", "N/A")
        st.caption("KL is a per-step training metric, not per-trajectory.")
    if d.get("reward_components"):
        comp = {k: v for k, v in d["reward_components"].items() if isinstance(v, (int, float))}
        st.bar_chart(pd.Series(comp), height=200)
    st.json({"policy_id": d["policy_id"], "job_id": d["job_id"],
             "prompt_id": d["prompt_id"], "parent_id": d["parent_id"],
             "ancestry": d["ancestry"]})


def main() -> None:
    cfg = sidebar()
    st.title("ShopRL Fabric — operations console")
    api = get_client()
    try:
        h = api.health()
        st.caption(f"🟢 API OK · root `{h['root']}`")
    except Exception as e:
        st.error(f"API unreachable: {e}")
        return

    tabs = st.tabs(["Overview", "Jobs", "Training Health", "Experiments",
                    "Policies", "Checkpoints", "Trajectories", "Recovery"])
    with tabs[0]:                                     # live overview + scheduler
        @st.fragment(run_every=cfg["interval"] if cfg["live"] else None)
        def _live():
            overview_panel(api)
            scheduler_panel(api)
            st.caption("🟢 live" if cfg["live"] else "⏸ paused — toggle in sidebar")
        _live()
    with tabs[1]:
        jobs_panel(api)
    with tabs[2]:
        training_health_panel(api)
    with tabs[3]:
        experiments_panel(api)
    with tabs[4]:
        policies_panel(api)
    with tabs[5]:
        checkpoints_panel(api)
    with tabs[6]:
        trajectory_panel(api)
    with tabs[7]:
        recovery_panel(api)


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _has_streamlit_context():
    main()
