# Dashboards — run guide

Two surfaces, distinct jobs (the earlier direct-reading Streamlit dashboard and
static HTML platform dashboard were consolidated into the API-driven ops console):

1. **Operations console** (`platform/ops_console.py`) — a live Streamlit app that
   operates the platform **only through the HTTP API**: jobs, scheduler, policy
   versions + staleness, checkpoints (integrity), experiments + comparison, RL
   metrics (reward/KL/entropy/grad-norm) with active alerts, and a trajectory
   explorer. Dev controls: pause/resume/cancel (real) + kill-worker/replay
   (SIMULATION).
2. **RL-metrics dashboard** (`observability/dashboard.py`) — a self-contained HTML
   view of a training run's `metrics.jsonl` (the 6-panel KL/entropy/reward
   curves). Different purpose: per-run RL curves, not live ops.

## Install

```bash
.venv/bin/pip install -e ".[dashboard,api]"
```

## Produce some real state

```bash
.venv/bin/python -m shoprl.platform.pipeline --root runs/pipeline
```

## Run the ops console

**Embedded (no server, default):** the console drives the same FastAPI app
in-process. Set the run root in the sidebar (default `runs/pipeline`).

```bash
.venv/bin/streamlit run src/shoprl/platform/ops_console.py
```

**Against a running API server:** switch the sidebar to `http` mode.

```bash
SHOPRL_ROOT=runs/pipeline SHOPRL_RUNS=runs .venv/bin/uvicorn shoprl.platform.api:app
.venv/bin/streamlit run src/shoprl/platform/ops_console.py   # sidebar: http, http://127.0.0.1:8000
```

## RL-metrics HTML (per run)

```bash
.venv/bin/python -m shoprl.observability.dashboard \
    --metrics runs/<exp>/metrics.jsonl --out runs/<exp>/dashboard.html
```

## Honesty note

The console reads only via the API; it imports no trainer internals and shows
absent metrics as absent (never invents a curve). Pause/resume/cancel are real;
kill-worker and replay are labelled simulations.
