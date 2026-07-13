#!/usr/bin/env bash
# Gated, CUDA-matched vLLM install for the rollout benchmark.
#
# Why this is gated: the previous attempt ran `pip install vllm`, which pulled a
# vLLM built for a NEWER CUDA/torch than the AMI had (CUDA 13 vs the AMI's 12.8)
# and broke torch with `libnvrtc.so.13 not found`. The fix is (1) install vLLM
# into a SEPARATE virtualenv so it can never clobber the training env's torch,
# and (2) DRY-RUN first and ABORT if the resolver wants to change torch.
#
# Usage:
#   scripts/install_vllm_gpu.sh <vllm_venv_dir> [vllm_version]
# Then run the rollout-only benchmark with that venv's python (engine: vllm).
#
# This script does NOT provision a GPU and makes no cost. Run it ON the GPU box.
set -euo pipefail

VENV="${1:-.venv-vllm}"
VLLM_VERSION="${2:-}"   # e.g. 0.9.2 ; leave empty to let pip pick, still gated

echo "== Detecting the environment's torch / CUDA =="
BASE_PY="$(command -v python3 || true)"
[ -n "$BASE_PY" ] || { echo "no python3 on PATH"; exit 1; }
"$BASE_PY" - <<'PY'
import torch
print("torch     :", torch.__version__)
print("cuda      :", torch.version.cuda)
print("cuda avail:", torch.cuda.is_available())
PY
TORCH_BEFORE="$("$BASE_PY" -c 'import torch; print(torch.__version__)')"
CUDA_BEFORE="$("$BASE_PY" -c 'import torch; print(torch.version.cuda)')"

echo "== Creating an ISOLATED venv for vLLM: $VENV =="
"$BASE_PY" -m venv "$VENV" --system-site-packages
# --system-site-packages so it reuses the AMI's torch; the dry-run gate below
# ensures vLLM does not try to REPLACE that torch.

PKG="vllm"
[ -n "$VLLM_VERSION" ] && PKG="vllm==$VLLM_VERSION"

echo "== DRY RUN: does '$PKG' want to change torch? =="
DRY="$("$VENV/bin/pip" install --dry-run "$PKG" 2>&1 || true)"
echo "$DRY" | grep -Ei 'torch|nvidia-|cuda' || true
if echo "$DRY" | grep -Eiq 'Would install torch-[0-9]'; then
  echo "ABORT: the resolver wants to install/replace torch. Pin a vLLM version"
  echo "       built for torch $TORCH_BEFORE / CUDA $CUDA_BEFORE and retry, or"
  echo "       build vLLM from source against the installed torch."
  exit 2
fi

echo "== Installing vLLM (isolated) =="
"$VENV/bin/pip" install "$PKG"

echo "== Verifying torch is UNCHANGED and CUDA still loads =="
"$VENV/bin/python" - <<PY
import torch
tb, cb = "$TORCH_BEFORE", "$CUDA_BEFORE"
assert torch.__version__ == tb, f"torch changed: {tb} -> {torch.__version__}"
assert str(torch.version.cuda) == cb, f"cuda changed: {cb} -> {torch.version.cuda}"
assert torch.cuda.is_available(), "CUDA no longer available"
import vllm; print("vllm", vllm.__version__, "OK; torch", torch.__version__, "cuda", torch.version.cuda)
PY

echo "== OK. Run the benchmark, e.g.:"
echo "   $VENV/bin/python -m shoprl.bench.harness --config configs/bench_vllm.yaml \\"
echo "       --steps 8 --rollout-only --out results/bench_vllm.json"
