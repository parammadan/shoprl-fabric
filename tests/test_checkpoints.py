"""Pillar 4 tests: atomic write, corruption detection, resume-equivalence."""
import json

import pytest

from shoprl.platform import CheckpointCorrupt, CheckpointRegistry
from shoprl.platform.checkpoints import (MANIFEST_NAME, CheckpointNotFound,
                                         _STAGING)


def _src(tmp_path, **files):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    for name, content in files.items():
        (d / name).write_bytes(content if isinstance(content, bytes)
                               else content.encode())
    return d


# --- save + manifest -------------------------------------------------------
def test_save_produces_ready_manifest_with_hashes(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    src = _src(tmp_path, **{"adapter.bin": b"\x00\x01weights",
                            "train_state.json": '{"step": 5}'})
    m = reg.save(src, step=5, policy_id="step-005")
    assert m.status == "READY" and m.step == 5 and m.policy_id == "step-005"
    assert {e.path for e in m.files} == {"adapter.bin", "train_state.json"}
    assert all(len(e.sha256) == 64 and e.size > 0 for e in m.files)


def test_latest_and_list_ordered(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    reg.save(_src(tmp_path, a="1"), step=1)
    m2 = reg.save(_src(tmp_path, a="2"), step=2)
    assert len(reg.list()) == 2
    assert reg.latest().ckpt_id == m2.ckpt_id


def test_get_missing_raises(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    with pytest.raises(CheckpointNotFound):
        reg.get("nope")


# --- corruption detection --------------------------------------------------
def test_verify_passes_on_clean_checkpoint(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    m = reg.save(_src(tmp_path, w=b"clean-bytes"), step=1)
    reg.verify(m.ckpt_id)                       # no raise
    assert reg.resolve(m.ckpt_id).exists()


def test_verify_detects_flipped_byte(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    m = reg.save(_src(tmp_path, w=b"original-content"), step=1)
    victim = reg.root / m.ckpt_id / "w"
    victim.write_bytes(b"corrupted-content!!")    # bit-rot / tamper after write
    with pytest.raises(CheckpointCorrupt):
        reg.verify(m.ckpt_id)


def test_verify_detects_missing_file(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    m = reg.save(_src(tmp_path, w=b"bytes", extra=b"more"), step=1)
    (reg.root / m.ckpt_id / "extra").unlink()     # truncated/lost file
    with pytest.raises(CheckpointCorrupt):
        reg.verify(m.ckpt_id)


# --- atomicity -------------------------------------------------------------
def test_crash_before_commit_leaves_no_ready_checkpoint(tmp_path):
    # Simulate a crash *during* a write: a half-written dir exists only under
    # .staging/, never as a resolvable checkpoint. The registry must not see it.
    reg = CheckpointRegistry(tmp_path / "ckpts")
    reg.save(_src(tmp_path, a="good"), step=1)            # one real checkpoint
    # hand-craft an orphaned staging dir (what a crash would leave behind)
    orphan = reg.root / _STAGING / "step-000009-deadbeef"
    orphan.mkdir(parents=True)
    (orphan / "adapter.bin").write_bytes(b"half")
    # no manifest, never renamed into place:
    assert len(reg.list()) == 1                           # orphan not listed
    assert reg.latest().step == 1
    assert reg.sweep_staging() == 1                       # cleanable
    assert not orphan.exists()


def test_no_partial_dir_in_root_during_normal_save(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    m = reg.save(_src(tmp_path, a="x"), step=3)
    # the committed dir contains a manifest (it only appears post-rename)
    assert (reg.root / m.ckpt_id / MANIFEST_NAME).exists()
    # staging is empty after a successful commit
    assert list((reg.root / _STAGING).iterdir()) == []


# --- resume-equivalence ----------------------------------------------------
def test_resume_equivalence_state_roundtrip(tmp_path):
    # Saving then resuming reproduces the exact state that was checkpointed.
    reg = CheckpointRegistry(tmp_path / "ckpts")
    state = {"step": 7, "optimizer": {"lr": 1e-5, "beta": [0.9, 0.999]},
             "rng": [1, 2, 3, 4]}
    m = reg.save_state(state, step=7, policy_id="step-007")
    resumed = reg.load_state(m.ckpt_id)
    assert resumed == state                       # byte-for-byte equivalent


def test_resume_equivalence_survives_new_registry_instance(tmp_path):
    root = tmp_path / "ckpts"
    m = CheckpointRegistry(root).save_state({"step": 2, "v": "abc"}, step=2)
    # fresh process/instance pointing at the same root
    reg2 = CheckpointRegistry(root)
    assert reg2.load_state(m.ckpt_id) == {"step": 2, "v": "abc"}
