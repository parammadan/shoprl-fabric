# vLLM vs HF rollout benchmark — run guide (gated)

The one performance measurement still open. Rollout is 82–84% of RL wall-clock
here (measured), so vLLM's throughput vs HF `generate` is the headline efficiency
lever — but the last attempt broke torch (`libnvrtc.so.13`: a vLLM built for
CUDA 13 clobbered the AMI's CUDA 12.8). This plan avoids that and produces real,
comparable numbers.

## Metrics captured (both engines, identical settings)

`shoprl.bench.harness` now reports, all measured:

- **tokens/sec**, **requests/sec**, **rollout latency (ms/request)**
- **iteration time (s/step)**, **peak GPU memory (GB)**
- **padding waste** (default vs length-bucketed)
- **TTFT (ms)** — from vLLM's per-request metrics; HF batch-generate reports
  `None` (shown as absent, never fabricated)

## The gate (no cost until you approve)

1. **Isolated venv + dry-run.** `scripts/install_vllm_gpu.sh` creates a separate
   venv (so vLLM can't replace the training env's torch), **dry-runs** the
   install, and **aborts if the resolver wants to change torch**, then verifies
   torch/CUDA are unchanged and `import vllm` works. Nothing installs if the
   versions would conflict.
2. **Rollout-only leg.** The vLLM leg runs `--rollout-only` (skips the LoRA
   optimize), so it measures pure generation throughput and doesn't need the
   training stack in the vLLM venv — the two engines are compared on the *same*
   task/model/settings, generation side by side.

## Run (on the GPU box)

```bash
# HF leg (training env)
python -m shoprl.bench.harness --config configs/bench_hf.yaml \
    --steps 8 --rollout-only --out results/bench_hf.json

# vLLM leg (isolated, CUDA-matched)
scripts/install_vllm_gpu.sh .venv-vllm            # gated install
.venv-vllm/bin/python -m shoprl.bench.harness --config configs/bench_vllm.yaml \
    --steps 8 --rollout-only --out results/bench_vllm.json
```

Both configs use identical model / batch / max_new_tokens / dtype so the delta
is attributable to the engine.

## After the run

- Record the two `results/bench_*.json` as **BENCHMARK artifacts** in the
  artifact registry (lineage to the run), and add the real numbers to
  `EFFICIENCY.md` (docs). **Do not fabricate improvements** — report whatever is
  measured, including a null result.
- **Teardown:** terminate the instance and verify (`0` running instances /
  volumes, spot request closed) as with every cloud run on this project.

## Status

Tooling is ready and tested on CPU (metric derivations unit-tested). The GPU run
is **pending explicit go-ahead + a provider** (AWS spot A10G / Kaggle T4 / Modal)
— it is the only step that spends money, so it is not run automatically.
