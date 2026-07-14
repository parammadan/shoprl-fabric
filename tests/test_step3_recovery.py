"""Step 3 recovery-scenario tests (laptop-verifiable): OOM recovery shrinks the
batch + restores a checkpoint + resumes; corrupt checkpoint is detected. The
REAL CUDA OOM path is exercised on the GPU (configs/demo_gpu_oom.yaml); here we
inject the failure deterministically to test the recovery LOGIC without a GPU."""
import json
import pathlib
import tempfile

import pytest

from shoprl.config import Config, RolloutConfig, TrainingConfig
from shoprl.platform import CheckpointRegistry, dash_data
from shoprl.platform.checkpoints import CheckpointCorrupt
from shoprl.platform.control import run_with_oom_recovery
from shoprl.platform.failures import SimulatedOOM


def _cfg(prompts=4, num_samples=8):
    return Config(rollout=RolloutConfig(engine="stub", num_samples=num_samples),
                  training=TrainingConfig(prompts_per_step=prompts, catalog_size=120))


def _fake_out():
    d = tempfile.mkdtemp(prefix="ck-")
    (pathlib.Path(d) / "adapter.safetensors").write_bytes(b"\x00w")
    return {"checkpoint_dir": d,
            "result": {"algorithm": "grpo", "reward_gain": 0.05, "final_kl": 0.02,
                       "train_time_s": 1.0},
            "samples": [{"prompt": "p", "response": "ADD X", "reward": 0.5,
                         "components": {"total": 0.5}, "prompt_id": "P1"}]}


def _seed_checkpoint(root):
    src = pathlib.Path(root) / "src"; src.mkdir(parents=True, exist_ok=True)
    (src / "adapter.safetensors").write_bytes(b"\x00seed")
    return CheckpointRegistry(pathlib.Path(root) / "checkpoints").save(src, step=5)


def test_oom_recovery_shrinks_restores_resumes(tmp_path):
    _seed_checkpoint(tmp_path)                         # a prior good checkpoint to restore
    calls = []

    def runner(config, n, ns, resume_from=None):
        batch = config.training.prompts_per_step * config.rollout.num_samples
        calls.append((batch, resume_from))
        if config.training.prompts_per_step > 1:       # "oversized" -> real-OOM analog
            raise SimulatedOOM(f"cuda oom at batch {batch}")
        return _fake_out()

    ref = run_with_oom_recovery(_cfg(prompts=4, num_samples=8), 8, 2, str(tmp_path),
                                runner=runner)
    assert ref["run_id"]                               # eventually succeeded
    batches = [b for b, _ in calls]
    assert batches[0] > batches[-1]                    # batch shrank: 32 -> 16 -> 8
    assert calls[-1][1] is not None                    # resumed FROM a restored checkpoint
    events = [json.loads(l) for l in open(tmp_path / "recovery_events.jsonl")]
    assert events and all(e["failure_class"] == "oom" for e in events)
    assert events[0]["restored_ckpt"] and events[0]["microbatch_before"] > events[0]["microbatch_after"]


def test_oom_recovery_dead_letters_when_cannot_shrink(tmp_path):
    def always_oom(config, n, ns, resume_from=None):
        raise SimulatedOOM("cuda oom")
    with pytest.raises(SimulatedOOM):
        run_with_oom_recovery(_cfg(prompts=1, num_samples=2), 8, 2, str(tmp_path),
                              runner=always_oom, max_oom_retries=3)
    events = [json.loads(l) for l in open(tmp_path / "recovery_events.jsonl")]
    assert events[-1]["resulting_state"] == "dead_letter"   # couldn't fit at min batch


def test_non_oom_error_is_not_swallowed(tmp_path):
    def boom(config, n, ns, resume_from=None):
        raise ValueError("bad config")                 # PERMANENT, not OOM
    with pytest.raises(ValueError):
        run_with_oom_recovery(_cfg(), 8, 2, str(tmp_path), runner=boom)


def test_corrupt_checkpoint_detected(tmp_path):
    _seed_checkpoint(tmp_path)
    r = dash_data.sim_corrupt_checkpoint(tmp_path)
    assert r["ok"] and r["integrity"] == "CORRUPT"
    # the live snapshot the dashboard reads now flags it CORRUPT
    cks = dash_data.snapshot(tmp_path)["checkpoints"]
    assert any(c["integrity"] == "CORRUPT" for c in cks)


def test_corrupt_checkpoint_none_present(tmp_path):
    assert dash_data.sim_corrupt_checkpoint(tmp_path)["ok"] is False
