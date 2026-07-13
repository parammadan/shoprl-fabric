# Live dashboard — run guide

A running Streamlit app over the platform's **real persisted state** (SQLite job
store, trajectory store, checkpoint registry, recovery-event log, RL metrics). It
reads real data and refreshes live. It imports **no trainer internals** and
**fabricates no metric** — anything not recorded is shown as absent.

## 1. Install

```bash
.venv/bin/pip install -e ".[dashboard]"      # or: pip install "streamlit>=1.36"
```

## 2. Produce some real state to look at

The dashboard reads a *run directory*. Create one by running the end-to-end
pipeline (no GPU or model download needed):

```bash
.venv/bin/python -m shoprl.platform.pipeline --root runs/pipeline
```

This writes `runs/pipeline/{jobs.db, trajectories.db, checkpoints/,
recovery_events.jsonl, pipeline_metrics.jsonl}`.

## 3. Launch

```bash
.venv/bin/streamlit run src/shoprl/platform/streamlit_app.py
```

In the sidebar set **Run root** to your run dir (default `runs/pipeline`).
Optional sidebar inputs:
- **RL results dir** — a folder of `*.json` written by `shoprl.rl.run`
  (the RLOO/GRPO/PPO comparison). Absent → the KL-comparison panel says so.
- **RL trainer metrics.jsonl** — a real training run's metrics for the
  KL/entropy curves. Absent → shown as absent, never invented.
- **Live auto-refresh** + interval — the job-states panel re-reads on a timer.

## 4. What each panel shows

| Panel | Source (all real, on disk) |
|---|---|
| **Job states** (live) | `jobs.db` counts by lifecycle stage, auto-refreshing |
| **Reward & KL** | reward per policy version from `trajectories.db`; RLOO/GRPO/PPO `final_kl`/`max_kl` + per-step KL curves from `results/*.json`; trainer KL/entropy from a `metrics.jsonl` if supplied |
| **Checkpoints** | `checkpoints/` registry, **sha256 integrity re-verified on load** (OK / CORRUPT) |
| **Recovery events** | `recovery_events.jsonl`; OOMs labelled **SIMULATED** |
| **Trajectory explorer** | pick a trajectory → prompt, response, **real** reward components, total reward, **real group-relative advantage**, policy version, and lineage/ancestry. KL is shown **N/A** (a per-step training metric, not per-trajectory) rather than invented. |

## 5. DEV MODE — fault injection (SIMULATION)

The sidebar expander holds confirmation-gated fault controls. Each is a real
operation on the real stores, but the *fault* is simulated and labelled:

- **Kill worker** — claims a job then expires its lease and reaps it (what
  happens when a worker process dies mid-job); the reaper requeues it.
- **Trigger OOM** — raises `SimulatedOOM` through the recovery controller,
  producing a real `RecoveryEvent` (batch shrink + checkpoint restore + requeue).
- **Duplicate trajectory** — `derive()`s a lineage-linked copy of the latest
  trajectory (redelivery / provenance demo).

Refresh a panel (or wait for auto-refresh) to see the injected effect.

## Honesty note

Real: the whole read path, integrity checks, reward components, group-relative
advantage, and recovery logic. Simulated + labelled: worker death, OOM trigger
(a laptop has no CUDA), and the stub token generation upstream. KL / advantage
that aren't persisted are shown absent, not fabricated.
