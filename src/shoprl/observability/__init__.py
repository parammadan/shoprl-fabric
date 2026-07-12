"""Observability: read training metrics logs, render a dashboard to watch runs."""
from shoprl.observability.alerts import Alert, Thresholds, check_run, check_step, summarize
from shoprl.observability.dashboard import load_metrics, render_dashboard

__all__ = [
    "load_metrics", "render_dashboard",
    "Alert", "Thresholds", "check_step", "check_run", "summarize",
]
