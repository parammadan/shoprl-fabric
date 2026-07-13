"""GPU telemetry — real measurements when CUDA is present, honest absence otherwise.

Collects device name, memory (allocated / reserved / peak), and utilization when
running on a CUDA GPU. On an M1 / CPU it returns {"available": False} — it never
fabricates a number. Utilization needs NVML (pynvml or `nvidia-smi`); if neither
is present it is reported as None (available but unmeasured), not invented.
"""
from __future__ import annotations


def _nvml_utilization(index: int = 0) -> float | None:
    """GPU utilization %, via pynvml then nvidia-smi. None if neither works."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        pynvml.nvmlShutdown()
        return float(util)
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def gpu_telemetry() -> dict:
    """Snapshot of GPU state. {'available': False} on CPU/MPS (never faked)."""
    try:
        import torch
    except Exception:
        return {"available": False, "reason": "torch not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}
    i = torch.cuda.current_device()
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(i),
        "mem_allocated_gb": round(torch.cuda.memory_allocated(i) / 1e9, 3),
        "mem_reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 3),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 3),
        "utilization_pct": _nvml_utilization(i),   # None if NVML unavailable
    }
