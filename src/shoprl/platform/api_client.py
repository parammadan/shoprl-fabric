"""HTTP client for the platform API.

The ops console talks to the platform ONLY through this client — never by
importing the stores. That's the point of the API boundary: the UI depends on
the HTTP contract, not on Python internals.

Two modes:
  - ApiClient(base_url)         -> real HTTP to a running `uvicorn` server.
  - ApiClient.in_process(root)  -> the SAME FastAPI app over an in-process ASGI
                                   transport (no network hop). Still goes through
                                   the real API layer; convenient for a
                                   single-machine demo and for tests.
"""
from __future__ import annotations

from pathlib import Path

import httpx


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000",
                 client: httpx.Client | None = None, timeout: float = 10.0):
        self._c = client or httpx.Client(base_url=base_url, timeout=timeout)

    @classmethod
    def in_process(cls, root: str | Path, runs_dir: str | Path = "runs") -> "ApiClient":
        # The SAME FastAPI app over Starlette's TestClient — a sync httpx client
        # that drives the ASGI app in-process (no network, no server). Still goes
        # through the real API layer.
        from fastapi.testclient import TestClient
        from shoprl.platform.api import create_app
        return cls(client=TestClient(create_app(root, runs_dir)))

    # --- low level -------------------------------------------------------
    def _get(self, path: str, optional: bool = False, **params):
        r = self._c.get(path, params={k: v for k, v in params.items() if v is not None})
        if optional and r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: dict | None = None):
        r = self._c.post(path, json=json)
        r.raise_for_status()
        return r.json()

    # --- typed helpers ---------------------------------------------------
    def health(self) -> dict:
        return self._get("/health")

    def overview(self) -> dict:
        return self._get("/overview")

    def scheduler(self) -> dict:
        return self._get("/scheduler")

    def jobs(self, state: str | None = None, resource: str | None = None) -> list:
        return self._get("/jobs", state=state, resource=resource)

    def job(self, job_id: str) -> dict:
        return self._get(f"/jobs/{job_id}")

    def pause(self, job_id: str) -> dict:
        return self._post(f"/jobs/{job_id}/pause")

    def resume(self, job_id: str) -> dict:
        return self._post(f"/jobs/{job_id}/resume")

    def cancel(self, job_id: str) -> dict:
        return self._post(f"/jobs/{job_id}/cancel")

    def checkpoints(self) -> list:
        return self._get("/checkpoints")

    def trajectories(self, limit: int = 200) -> list:
        return self._get("/trajectories", limit=limit)

    def trajectory(self, traj_id: str) -> dict:
        return self._get(f"/trajectories/{traj_id}")

    def runs(self) -> list:
        return self._get("/runs")

    def run(self, run_id: str) -> dict:
        return self._get(f"/runs/{run_id}")

    def run_metrics(self, run_id: str) -> dict | None:
        return self._get(f"/runs/{run_id}/metrics", optional=True)

    def run_alerts(self, run_id: str) -> dict | None:
        return self._get(f"/runs/{run_id}/alerts", optional=True)

    def metrics_runs(self) -> list:
        return self._get("/metrics-runs")

    def comparisons(self) -> list:
        return self._get("/comparisons")

    def compare_runs(self, ids: list[str]) -> dict:
        return self._get("/runs/compare", ids=",".join(ids))

    def policies(self) -> list:
        return self._get("/policies")

    def policy_latest(self) -> dict | None:
        return self._get("/policies/latest", optional=True)

    def policy_staleness(self) -> dict | None:
        return self._get("/policies/staleness", optional=True)

    def artifacts(self, type: str | None = None, run_id: str | None = None) -> list:
        return self._get("/artifacts", type=type, run_id=run_id)

    # --- dev-mode (SIMULATION) -------------------------------------------
    def kill_worker(self) -> dict:
        return self._post("/dev/kill-worker")

    def replay(self, traj_id: str) -> dict:
        return self._post(f"/dev/replay/{traj_id}")

    def corrupt_checkpoint(self) -> dict:
        return self._post("/dev/corrupt-checkpoint")

    def sim_oom(self) -> dict:
        return self._post("/dev/oom")
