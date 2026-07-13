"""End-to-end test: all six pillars driven by the pipeline on real data."""
import pytest

from shoprl.platform.checkpoints import CheckpointRegistry
from shoprl.platform.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipe")
    cfg = PipelineConfig(steps=2, prompts_per_step=2, num_samples=2,
                         n_workers=2, oom_at_step=1)
    summary = run_pipeline(root, cfg)
    return root, summary


def test_all_work_succeeds_nothing_dead_lettered(run):
    _, s = run
    # 4 rollout jobs (2 steps x 2 prompts) + 2 optimize = 6 succeeded
    assert s["job_counts"].get("succeeded") == 6
    assert s["job_counts"].get("dead_letter", 0) == 0
    assert s["job_counts"].get("pending", 0) == 0


def test_trajectories_persisted_with_expected_count(run):
    _, s = run
    assert s["trajectories"] == 2 * 2 * 2          # steps x prompts x samples


def test_two_checkpoints_written_and_verify_clean(run):
    root, s = run
    assert len(s["checkpoints"]) == 2
    reg = CheckpointRegistry(root / "checkpoints")
    for ckpt_id in s["checkpoints"]:
        reg.verify(ckpt_id)                        # no raise = integrity intact


def test_oom_was_recovered_not_fatal(run):
    _, s = run
    assert s["recovery_events"] == 1
    assert s["per_step"][1]["recovered_oom"] is True
    assert s["per_step"][0]["recovered_oom"] is False


def test_rewards_are_real_numbers(run):
    _, s = run
    for row in s["per_step"]:
        assert 0.0 <= row["reward_mean"] <= 1.0    # rule-based reward is in [0,1]
        assert row["n_trajectories"] == 4
