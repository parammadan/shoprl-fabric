"""Checkpoint registry with atomic write + integrity verification.

The problem this solves: the legacy `save_checkpoint` writes the adapter and
train_state.json *directly into the final path*. If the process dies (or a spot
GPU is reclaimed) mid-write, you're left with a directory that LOOKS like a
checkpoint but is truncated — and you'll happily resume from garbage.

The registry makes a checkpoint's appearance atomic and its contents verifiable:

    stage -> copy files (hashing each) -> write manifest -> fsync
          -> os.replace(staging -> final)   # the single commit point
          -> READY

Because the final directory only appears via one atomic rename *after* the
manifest is written, there is no observable half-written checkpoint: a crash at
any earlier point leaves only a `.staging/` dir (garbage, swept later), never a
resolvable one. The on-disk manifest (per-file sha256 + sizes) IS the registry —
`verify()` re-hashes and compares, so bit-rot or truncation is detected before a
resume. There is no separate DB index to drift out of sync with the files.

Scope: real, single-machine, local filesystem. `os.replace` is atomic only
within one filesystem (true here). Corruption in tests is injected deliberately
(we flip a byte); the detection logic is real.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_NAME = "manifest.json"
_STAGING = ".staging"


class CheckpointNotFound(KeyError):
    pass


class CheckpointCorrupt(Exception):
    """A file's contents don't match the manifest (truncation / bit-rot), or a
    manifested file is missing."""


class FileEntry(BaseModel):
    path: str          # relative to the checkpoint dir
    sha256: str
    size: int


class Manifest(BaseModel):
    ckpt_id: str
    step: int
    policy_id: str | None = None
    created_at: float = Field(default_factory=time.time)
    status: str = "READY"          # only READY manifests are ever written to disk
    files: list[FileEntry]
    metadata: dict = Field(default_factory=dict)


def _sha256(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _fsync_dir(path: Path) -> None:
    # Best effort: make the rename/durability ordering real. Not all platforms
    # allow fsync on a directory fd; ignore failures honestly.
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


class CheckpointRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / _STAGING).mkdir(parents=True, exist_ok=True)

    # --- write -----------------------------------------------------------
    def save(self, source_dir: str | Path, *, step: int,
             policy_id: str | None = None, metadata: dict | None = None) -> Manifest:
        """Atomically register the contents of `source_dir` as a checkpoint."""
        source_dir = Path(source_dir)
        ckpt_id = f"step-{step:06d}-{uuid.uuid4().hex[:8]}"
        staging = self.root / _STAGING / ckpt_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        # copy every file, hashing as we go
        entries: list[FileEntry] = []
        for src in sorted(source_dir.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(source_dir).as_posix()
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            digest, size = _sha256(dst)
            entries.append(FileEntry(path=rel, sha256=digest, size=size))
            with open(dst, "rb") as f:                 # durability before commit
                os.fsync(f.fileno())

        manifest = Manifest(ckpt_id=ckpt_id, step=step, policy_id=policy_id,
                            files=entries, metadata=metadata or {})
        with open(staging / MANIFEST_NAME, "w") as f:
            f.write(manifest.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(staging)

        final = self.root / ckpt_id
        os.replace(staging, final)                     # <-- atomic commit point
        _fsync_dir(self.root)
        return manifest

    def save_state(self, state: dict, *, step: int, policy_id: str | None = None,
                   metadata: dict | None = None) -> Manifest:
        """Convenience: checkpoint an arbitrary JSON-serialisable state dict
        (stand-in for a trainer's train_state / small adapter payload)."""
        tmp = self.root / _STAGING / f"_state-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            with open(tmp / "state.json", "w") as f:
                json.dump(state, f, sort_keys=True)
            return self.save(tmp, step=step, policy_id=policy_id, metadata=metadata)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- read / verify ---------------------------------------------------
    def get(self, ckpt_id: str) -> Manifest:
        mpath = self.root / ckpt_id / MANIFEST_NAME
        if not mpath.exists():
            raise CheckpointNotFound(ckpt_id)
        return Manifest.model_validate_json(mpath.read_text())

    def verify(self, ckpt_id: str) -> Manifest:
        """Re-hash every manifested file and compare. Raises CheckpointCorrupt
        on any mismatch or missing file. This is what guards a resume."""
        manifest = self.get(ckpt_id)
        base = self.root / ckpt_id
        for entry in manifest.files:
            fpath = base / entry.path
            if not fpath.exists():
                raise CheckpointCorrupt(f"{ckpt_id}: missing file {entry.path}")
            digest, size = _sha256(fpath)
            if digest != entry.sha256:
                raise CheckpointCorrupt(
                    f"{ckpt_id}: checksum mismatch for {entry.path}")
            if size != entry.size:
                raise CheckpointCorrupt(f"{ckpt_id}: size mismatch for {entry.path}")
        return manifest

    def resolve(self, ckpt_id: str) -> Path:
        """Verify integrity, then return the on-disk path to resume from."""
        self.verify(ckpt_id)
        return self.root / ckpt_id

    def load_state(self, ckpt_id: str) -> dict:
        """Verify + read back a state saved via save_state()."""
        path = self.resolve(ckpt_id)
        return json.loads((path / "state.json").read_text())

    # --- list ------------------------------------------------------------
    def list(self) -> list[Manifest]:
        out = []
        for child in self.root.iterdir():
            if child.name == _STAGING or not child.is_dir():
                continue
            if (child / MANIFEST_NAME).exists():
                out.append(self.get(child.name))
        return sorted(out, key=lambda m: m.created_at)

    def latest(self) -> Manifest | None:
        items = self.list()
        return items[-1] if items else None

    def sweep_staging(self) -> int:
        """Remove orphaned staging dirs left by crashed writes. Returns count."""
        staging = self.root / _STAGING
        n = 0
        for child in staging.iterdir():
            shutil.rmtree(child, ignore_errors=True)
            n += 1
        return n

    def prune(self, keep_last_n: int, protect: list[str] | None = None) -> list[str]:
        """Retention: keep the most recent `keep_last_n` checkpoints, delete the
        rest — except any id in `protect` (e.g. a run's best_checkpoint), which
        is never removed. Checkpoints are large; without this the dir grows
        unbounded. Returns the removed ckpt_ids."""
        if keep_last_n < 1:
            raise ValueError("keep_last_n must be >= 1")
        protect = set(protect or [])
        items = self.list()                          # oldest-first (by created_at)
        keep_recent = {m.ckpt_id for m in items[-keep_last_n:]}
        removed = []
        for m in items:
            if m.ckpt_id in keep_recent or m.ckpt_id in protect:
                continue
            shutil.rmtree(self.root / m.ckpt_id, ignore_errors=True)
            removed.append(m.ckpt_id)
        return removed
