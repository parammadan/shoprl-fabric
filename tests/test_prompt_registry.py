"""Prompt registry tests: versioned, hashed, reproducible prompt datasets."""
import pytest

from shoprl.platform import PromptRegistry
from shoprl.platform.prompt_registry import (PromptDatasetCorrupt,
                                             PromptDatasetNotFound,
                                             dataset_version_id)


def test_materialize_persists_versioned_dataset(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    meta = reg.materialize(catalog_size=120, n=16, seed=0)
    assert meta.dataset_version == dataset_version_id(120, 16, 0)
    assert meta.n_prompts == 16 and meta.seed == 0
    assert len(meta.hash) == 16 and meta.prompt_version == meta.hash


def test_load_returns_exact_prompts_with_answers(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    meta = reg.materialize(catalog_size=120, n=16, seed=0)
    prompts = reg.load(meta.dataset_version)
    assert len(prompts) == 16
    assert all(p.prompt_id and p.prompt for p in prompts)
    assert any(p.answer_skus for p in prompts)        # ground truth preserved


def test_same_params_are_reproducible_and_idempotent(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    m1 = reg.materialize(catalog_size=120, n=16, seed=0)
    m2 = reg.materialize(catalog_size=120, n=16, seed=0)   # re-materialize
    assert m1.hash == m2.hash                          # deterministic content
    assert len(reg.list()) == 1                        # not duplicated


def test_different_seed_gives_different_hash_and_version(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    a = reg.materialize(catalog_size=120, n=16, seed=0)
    b = reg.materialize(catalog_size=120, n=16, seed=1)
    assert a.dataset_version != b.dataset_version
    assert a.hash != b.hash
    assert len(reg.list()) == 2


def test_load_verifies_hash_and_detects_tampering(tmp_path):
    reg = PromptRegistry(tmp_path / "prompts")
    meta = reg.materialize(catalog_size=120, n=8, seed=0)
    victim = reg.root / meta.dataset_version / "prompts.jsonl"
    lines = victim.read_text().splitlines()
    lines[0] = lines[0].replace("P-0001", "P-9999", 1)   # valid JSON, changed value
    victim.write_text("\n".join(lines) + "\n")
    with pytest.raises(PromptDatasetCorrupt):
        reg.load(meta.dataset_version)


def test_missing_dataset_raises(tmp_path):
    with pytest.raises(PromptDatasetNotFound):
        PromptRegistry(tmp_path / "prompts").get("nope")


def test_survives_restart(tmp_path):
    root = tmp_path / "prompts"
    dv = PromptRegistry(root).materialize(catalog_size=120, n=8, seed=0).dataset_version
    assert len(PromptRegistry(root).load(dv)) == 8    # fresh instance reads it
