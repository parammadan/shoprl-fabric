"""Prompt registry — versioned, hashed, reproducible prompt datasets.

Prompts were regenerated from a seed on every run. That's reproducible in
principle, but it means a run's evaluation set only exists as "trust that seed N
still produces the same thing" — if the generator changes, silently so does the
data, and two runs you thought were comparable no longer are. The prompt
registry materialises a generated prompt set to disk as an immutable, hashed
artifact so a run references a *stored* dataset, and a content hash makes any
drift detectable.

It reuses the existing generators (`shoprl.data.generate_catalog` /
`generate_prompts`) — it does not reimplement prompt generation, only persists
and versions the result.

Tracks: dataset_version (params id), prompt_version (content hash of the exact
prompts), seed, hash, metadata. `materialize` is idempotent on identical params
and asserts the content hash is stable; `load` verifies the hash on read.

Scope: real, single-machine, local files. Not simulated.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from shoprl.data import generate_catalog, generate_prompts
from shoprl.data.prompts import PromptExample

MANIFEST = "manifest.json"
PROMPTS = "prompts.jsonl"
_STAGING = ".staging"


class PromptDatasetNotFound(KeyError):
    pass


class PromptDatasetCorrupt(Exception):
    pass


class PromptDatasetMeta(BaseModel):
    dataset_version: str                  # params identity, e.g. cat300-n64-seed0
    prompt_version: str                   # content hash of the exact prompts
    seed: int
    n_prompts: int
    catalog_size: int
    hash: str                             # == prompt_version (kept for clarity)
    created_at: float = Field(default_factory=lambda: __import__("time").time())
    metadata: dict = Field(default_factory=dict)


def _hash_prompts(prompts: list[PromptExample]) -> str:
    h = hashlib.sha256()
    for p in prompts:                     # order is deterministic from the generator
        h.update(p.model_dump_json().encode())
    return h.hexdigest()[:16]


def dataset_version_id(catalog_size: int, n: int, seed: int) -> str:
    return f"cat{catalog_size}-n{n}-seed{seed}"


class PromptRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / _STAGING).mkdir(parents=True, exist_ok=True)

    # --- materialize -----------------------------------------------------
    def materialize(self, *, catalog_size: int = 300, n: int = 64, seed: int = 0,
                    metadata: dict | None = None) -> PromptDatasetMeta:
        """Generate (deterministically) and persist a prompt dataset. Idempotent:
        if a dataset with these params already exists, verify its content hash is
        unchanged and return it (drift would raise)."""
        dv = dataset_version_id(catalog_size, n, seed)
        catalog = generate_catalog(n=catalog_size, seed=seed)
        prompts = generate_prompts(catalog, n=n, seed=seed)
        phash = _hash_prompts(prompts)

        existing = self.root / dv / MANIFEST
        if existing.exists():
            meta = self.get(dv)
            if meta.hash != phash:
                raise PromptDatasetCorrupt(
                    f"{dv} exists with hash {meta.hash} but regeneration now "
                    f"hashes {phash} — the prompt generator drifted")
            return meta

        meta = PromptDatasetMeta(
            dataset_version=dv, prompt_version=phash, seed=seed, n_prompts=len(prompts),
            catalog_size=catalog_size, hash=phash, metadata=metadata or {})
        staging = self.root / _STAGING / f"{dv}-{uuid.uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with open(staging / PROMPTS, "w") as f:
            for p in prompts:
                f.write(p.model_dump_json() + "\n")
        with open(staging / MANIFEST, "w") as f:
            f.write(meta.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(staging, self.root / dv)           # atomic commit
        return meta

    # --- read ------------------------------------------------------------
    def get(self, dataset_version: str) -> PromptDatasetMeta:
        mpath = self.root / dataset_version / MANIFEST
        if not mpath.exists():
            raise PromptDatasetNotFound(dataset_version)
        return PromptDatasetMeta.model_validate_json(mpath.read_text())

    def load(self, dataset_version: str, verify: bool = True) -> list[PromptExample]:
        """Load the exact stored prompts. Verifies the content hash by default —
        a tampered/corrupt dataset raises instead of silently training on drift."""
        meta = self.get(dataset_version)
        ppath = self.root / dataset_version / PROMPTS
        prompts = [PromptExample.model_validate_json(l)
                   for l in ppath.read_text().splitlines() if l.strip()]
        if verify and _hash_prompts(prompts) != meta.hash:
            raise PromptDatasetCorrupt(f"{dataset_version}: content hash mismatch")
        return prompts

    def list(self) -> list[PromptDatasetMeta]:
        out = []
        for child in self.root.iterdir():
            if child.name == _STAGING or not child.is_dir():
                continue
            if (child / MANIFEST).exists():
                out.append(self.get(child.name))
        return sorted(out, key=lambda m: m.created_at)
