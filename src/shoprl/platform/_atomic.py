"""Atomic directory-ingest primitive — shared by the checkpoint and policy
registries so the crash-safe write is implemented ONCE.

    stage under .staging/  ->  copy files (sha256 + size each)  ->  write manifest
      ->  fsync files + dir  ->  os.replace(staging -> final)   [the commit point]

A crash before the rename leaves only a .staging/ orphan, never a resolvable
half-written artifact. `os.replace` is atomic within one filesystem.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

STAGING = ".staging"


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def fsync_dir(path: Path) -> None:
    """Best effort — not all platforms allow fsync on a directory fd."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_ingest(root: Path, source_dir: Path, dest_name: str, *,
                  manifest_name: str,
                  manifest_bytes: Callable[[list[dict]], str],
                  overwrite: bool = False) -> tuple[list[dict], Path]:
    """Copy source_dir into root/dest_name atomically. `manifest_bytes(entries)`
    returns the manifest JSON to write (entries = [{path, sha256, size}]).
    Returns (entries, final_path). With overwrite=False a pre-existing dest_name
    raises FileExistsError (fail-closed against a version-collision race)."""
    root = Path(root)
    (root / STAGING).mkdir(parents=True, exist_ok=True)
    final = root / dest_name
    if final.exists() and not overwrite:
        raise FileExistsError(f"{dest_name} already exists in {root}")

    staging = root / STAGING / f"{dest_name}-{uuid.uuid4().hex[:8]}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    entries: list[dict] = []
    for src in sorted(Path(source_dir).rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(source_dir).as_posix()
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        digest, size = sha256_file(dst)
        entries.append({"path": rel, "sha256": digest, "size": size})
        with open(dst, "rb") as f:
            os.fsync(f.fileno())

    with open(staging / manifest_name, "w") as f:
        f.write(manifest_bytes(entries))
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(staging)
    os.replace(staging, final)                    # atomic commit point
    fsync_dir(root)
    return entries, final
