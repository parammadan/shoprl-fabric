"""Policy registry + weight synchronization — the on-policy lifecycle.

The correctness gap this closes: the trainer updates the policy every step, but
rollout workers are separate processes that generate with whatever adapter they
last loaded. If a worker keeps generating with an old adapter, its trajectories
are *off-policy* relative to the current policy — and nothing would notice.

The lifecycle:

    trainer trains a step
      -> PolicyRegistry.publish(adapter)        # atomic, versioned (v1, v2, ...)
        -> worker.refresh() loads the latest     # weight sync
          -> trajectory tagged with policy_version
            -> staleness = trainer_version - trajectory_version   # measured

`policy_version` reuses the trajectory's existing `lineage.policy_id` (as
"v{n}"), so staleness is computable from data already persisted (Pillar 3).

Atomic publish reuses the checkpoint-registry pattern (stage -> hash -> manifest
-> os.replace), so a worker never loads a half-written policy.

Honesty: on this project the "adapter" a test publishes is a small state dict,
not GB of weights, and a lagging worker is *simulated* by pinning it to an old
version (labelled). The versioning, atomic publish, refresh, and staleness
measurement are all real. Real multi-node weight broadcast is out of scope.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from shoprl.platform.checkpoints import _fsync_dir, _sha256

MANIFEST = "manifest.json"
_STAGING = ".staging"


class PolicyCorrupt(Exception):
    pass


class PolicyNotFound(KeyError):
    pass


class PolicyVersion(BaseModel):
    version: int
    fingerprint: str                      # sha256 over the file hashes
    created_at: float = Field(default_factory=time.time)
    metadata: dict = Field(default_factory=dict)
    files: list[dict] = Field(default_factory=list)   # [{path, sha256, size}]
    path: str | None = None               # on-disk dir (set on read, not stored)


class PolicyRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / _STAGING).mkdir(parents=True, exist_ok=True)

    # --- publish (trainer side) -----------------------------------------
    def _next_version(self) -> int:
        latest = self.latest()
        return 1 if latest is None else latest.version + 1

    def publish(self, source_dir: str | Path, metadata: dict | None = None) -> PolicyVersion:
        """Atomically register the next policy version from a directory of
        adapter files."""
        source_dir = Path(source_dir)
        version = self._next_version()
        staging = self.root / _STAGING / f"v{version}-{uuid.uuid4().hex[:8]}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        entries = []
        for src in sorted(source_dir.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(source_dir).as_posix()
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            digest, size = _sha256(dst)
            entries.append({"path": rel, "sha256": digest, "size": size})
            with open(dst, "rb") as f:
                os.fsync(f.fileno())
        fingerprint = _fingerprint(entries)
        pv = PolicyVersion(version=version, fingerprint=fingerprint,
                           metadata=metadata or {}, files=entries)
        with open(staging / MANIFEST, "w") as f:
            f.write(pv.model_dump_json(exclude={"path"}, indent=2))
            f.flush()
            os.fsync(f.fileno())
        _fsync_dir(staging)
        final = self.root / f"v{version}"
        os.replace(staging, final)                    # atomic commit
        _fsync_dir(self.root)
        pv.path = str(final)
        return pv

    def publish_state(self, state: dict, metadata: dict | None = None) -> PolicyVersion:
        tmp = self.root / _STAGING / f"_state-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        try:
            with open(tmp / "policy_state.json", "w") as f:
                json.dump(state, f, sort_keys=True)
            return self.publish(tmp, metadata)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # --- read (worker side) ---------------------------------------------
    def _version_dirs(self) -> list[int]:
        out = []
        for child in self.root.iterdir():
            if child.name.startswith("v") and (child / MANIFEST).exists():
                try:
                    out.append(int(child.name[1:]))
                except ValueError:
                    continue
        return sorted(out)

    def get(self, version: int) -> PolicyVersion:
        mpath = self.root / f"v{version}" / MANIFEST
        if not mpath.exists():
            raise PolicyNotFound(version)
        pv = PolicyVersion.model_validate_json(mpath.read_text())
        pv.path = str(self.root / f"v{version}")
        return pv

    def latest(self) -> PolicyVersion | None:
        vs = self._version_dirs()
        return self.get(vs[-1]) if vs else None

    def list(self) -> list[PolicyVersion]:
        return [self.get(v) for v in self._version_dirs()]

    def verify(self, version: int) -> PolicyVersion:
        pv = self.get(version)
        base = self.root / f"v{version}"
        for e in pv.files:
            fp = base / e["path"]
            if not fp.exists():
                raise PolicyCorrupt(f"v{version}: missing {e['path']}")
            digest, size = _sha256(fp)
            if digest != e["sha256"] or size != e["size"]:
                raise PolicyCorrupt(f"v{version}: checksum mismatch {e['path']}")
        return pv

    def load_state(self, version: int) -> dict:
        pv = self.verify(version)
        return json.loads((Path(pv.path) / "policy_state.json").read_text())


def _fingerprint(entries: list[dict]) -> str:
    import hashlib
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["path"]):
        h.update(e["sha256"].encode())
    return h.hexdigest()[:16]


# --- worker-side weight sync ----------------------------------------------
class PolicyClient:
    """A rollout worker's handle to the current policy. It refreshes to the
    latest published version before generating and tags trajectories with the
    version it actually used."""

    def __init__(self, registry: PolicyRegistry):
        self.registry = registry
        self._version: int | None = None
        self._pinned: int | None = None

    def pin(self, version: int) -> None:
        """SIMULATION: pin the worker to an old version so it stops refreshing —
        used to demonstrate stale-rollout detection. Not a production path."""
        self._pinned = version

    def refresh(self) -> int | None:
        if self._pinned is not None:
            self._version = self._pinned            # SIMULATION: lagging worker
        else:
            latest = self.registry.latest()
            self._version = latest.version if latest else None
        return self._version

    @property
    def version(self) -> int | None:
        return self._version

    def policy_id(self) -> str:
        """The string used on trajectory lineage.policy_id."""
        return f"v{self._version}"


# --- staleness measurement -------------------------------------------------
def staleness(current_version: int, trajectory_version: int) -> int:
    """How many policy updates behind the trajectory's generating policy was."""
    return current_version - trajectory_version


def parse_policy_version(policy_id: str) -> int | None:
    if isinstance(policy_id, str) and policy_id.startswith("v") and policy_id[1:].isdigit():
        return int(policy_id[1:])
    return None


def staleness_report(traj_store, current_version: int, limit: int = 500) -> dict:
    """Staleness distribution over recent trajectories, parsing their tagged
    policy version from lineage.policy_id. Trajectories not tagged 'v{n}' are
    ignored (reported as unversioned)."""
    versions, unversioned = [], 0
    for t in traj_store.recent(limit):
        v = parse_policy_version(t.lineage.policy_id)
        if v is None:
            unversioned += 1
        else:
            versions.append(current_version - v)
    if not versions:
        return {"n": 0, "unversioned": unversioned}
    return {"n": len(versions), "unversioned": unversioned,
            "mean_staleness": sum(versions) / len(versions),
            "max_staleness": max(versions),
            "stale_count": sum(1 for s in versions if s > 0),
            "on_policy_count": sum(1 for s in versions if s == 0)}
