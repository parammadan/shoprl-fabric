"""Pillar 6 test: the dashboard renders REAL persisted state, no fabrication."""
import json

import pytest

from shoprl.platform import dashboard
from shoprl.platform.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("dash")
    run_pipeline(r, PipelineConfig(steps=2, prompts_per_step=2, num_samples=2,
                                   n_workers=2, oom_at_step=1))
    return r


def test_collect_reads_real_state(root):
    d = dashboard.collect(root)
    assert d["job_counts"]["succeeded"] == 6
    assert d["reward_stats"]["count"] == 8
    assert len(d["checkpoints"]) == 2
    assert all(c["integrity"] == "OK" for c in d["checkpoints"])
    assert len(d["recovery_events"]) == 1


def test_render_writes_self_contained_html(root):
    out = dashboard.build(root)
    assert out.exists()
    text = out.read_text()
    assert "platform dashboard" in text
    assert "SIMULATED" in text                     # the recovery event is labelled
    assert "no gradient update" in text            # honest KL note, not a fake curve


def test_corruption_shows_as_corrupt(root):
    # flip a byte in one checkpoint file; the dashboard's live re-check flags it
    ck = dashboard.collect(root)["checkpoints"][0]["ckpt_id"]
    victim = next((root / "checkpoints" / ck).glob("state.json"))
    victim.write_text(victim.read_text() + " ")    # mutate contents
    d = dashboard.collect(root)
    statuses = {c["ckpt_id"]: c["integrity"] for c in d["checkpoints"]}
    assert statuses[ck] == "CORRUPT"


def test_real_training_metrics_shown_when_supplied(root, tmp_path):
    # a real trainer metrics.jsonl (kl/entropy) is shown alongside; values used verbatim
    mpath = tmp_path / "metrics.jsonl"
    mpath.write_text("\n".join(json.dumps(r) for r in [
        {"step": 0, "kl": 0.01, "entropy": 3.1, "reward_mean": 0.5},
        {"step": 1, "kl": 0.58, "entropy": 2.0, "reward_mean": 0.6}]))
    d = dashboard.collect(root, training_metrics=str(mpath))
    assert len(d["training_metrics"]) == 2
    out = dashboard.render(d, tmp_path / "d.html")
    text = out.read_text()
    assert "0.5800" in text                        # real max KL, not invented
