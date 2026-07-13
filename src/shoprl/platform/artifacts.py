"""Artifact registry — one metadata + lineage index over everything produced.

Checkpoints, prompt datasets, policies, benchmark outputs, and eval reports each
already live in their own store. What was missing is a single place to ask "what
artifacts did run R produce, and what was each derived from?" The artifact
registry is that index: it records a light, uniform record per artifact
(type, a ref into its own store, the producing run, parents, uri, hash,
metadata) and the parent edges between them — a cross-artifact lineage DAG.

It does NOT re-store the artifacts. A checkpoint's bytes stay in the
CheckpointRegistry; here we keep only a reference (ckpt_id) plus provenance. So
this unifies without duplicating, and the specialized registries remain the
source of truth for their own content.

Example lineage: prompt_dataset --\
                                   >-- checkpoint --< eval_report
                        policy  --/

Scope: real, single-machine, one SQLite file. Not simulated.
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    CHECKPOINT = "checkpoint"
    PROMPT_DATASET = "prompt_dataset"
    POLICY = "policy"
    BENCHMARK = "benchmark"
    EVAL_REPORT = "eval_report"


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: ArtifactType
    ref: str                                   # id/version in its own registry
    run_id: str | None = None                  # producing experiment run
    parents: list[str] = Field(default_factory=list)   # parent artifact_ids
    uri: str | None = None                     # on-disk location, if any
    hash: str | None = None                    # fingerprint, if available
    metadata: dict = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class ArtifactNotFound(KeyError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    ref         TEXT NOT NULL,
    run_id      TEXT,
    uri         TEXT,
    hash        TEXT,
    created_at  REAL NOT NULL,
    data        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifact_parents (
    child_id  TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    PRIMARY KEY (child_id, parent_id)
);
CREATE INDEX IF NOT EXISTS idx_art_type ON artifacts(type);
CREATE INDEX IF NOT EXISTS idx_art_run  ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_art_ref  ON artifacts(type, ref);
CREATE INDEX IF NOT EXISTS idx_art_parent ON artifact_parents(parent_id);
"""


class ArtifactRegistry:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def register(self, type: ArtifactType, ref: str, *, run_id: str | None = None,
                 parents: list[str] | None = None, uri: str | None = None,
                 hash: str | None = None, metadata: dict | None = None) -> Artifact:
        art = Artifact(type=ArtifactType(type), ref=ref, run_id=run_id,
                       parents=parents or [], uri=uri, hash=hash,
                       metadata=metadata or {})
        self.conn.execute(
            "INSERT INTO artifacts (artifact_id, type, ref, run_id, uri, hash, "
            "created_at, data) VALUES (?,?,?,?,?,?,?,?)",
            (art.artifact_id, art.type.value, art.ref, art.run_id, art.uri,
             art.hash, art.created_at, art.model_dump_json()))
        for p in art.parents:
            self.conn.execute(
                "INSERT OR IGNORE INTO artifact_parents (child_id, parent_id) "
                "VALUES (?,?)", (art.artifact_id, p))
        self.conn.commit()
        return art

    def get(self, artifact_id: str) -> Artifact:
        row = self.conn.execute(
            "SELECT data FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if row is None:
            raise ArtifactNotFound(artifact_id)
        return Artifact.model_validate_json(row["data"])

    def list(self, type: ArtifactType | None = None,
             run_id: str | None = None) -> list[Artifact]:
        q, args, clauses = "SELECT data FROM artifacts", [], []
        if type:
            clauses.append("type=?"); args.append(ArtifactType(type).value)
        if run_id:
            clauses.append("run_id=?"); args.append(run_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at"
        return [Artifact.model_validate_json(r["data"])
                for r in self.conn.execute(q, args).fetchall()]

    def by_ref(self, type: ArtifactType, ref: str) -> Artifact | None:
        row = self.conn.execute(
            "SELECT data FROM artifacts WHERE type=? AND ref=? "
            "ORDER BY created_at DESC LIMIT 1",
            (ArtifactType(type).value, ref)).fetchone()
        return Artifact.model_validate_json(row["data"]) if row else None

    def children(self, artifact_id: str) -> list[Artifact]:
        rows = self.conn.execute(
            "SELECT child_id FROM artifact_parents WHERE parent_id=?",
            (artifact_id,)).fetchall()
        return [self.get(r["child_id"]) for r in rows]

    def ancestors(self, artifact_id: str) -> list[Artifact]:
        """All transitive parents (DAG, deduped, cycle-guarded)."""
        seen: set[str] = set()
        out: list[Artifact] = []
        stack = list(self.get(artifact_id).parents)
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                p = self.get(pid)
            except ArtifactNotFound:
                continue
            out.append(p)
            stack.extend(p.parents)
        return out

    def lineage(self, artifact_id: str) -> dict:
        art = self.get(artifact_id)
        return {"artifact": art.model_dump(),
                "ancestors": [a.model_dump() for a in self.ancestors(artifact_id)],
                "children": [c.model_dump() for c in self.children(artifact_id)]}


# --- thin registrars pulling from the specialized registries ---------------
def register_checkpoint(areg: ArtifactRegistry, manifest, *, run_id=None,
                        parents=None) -> Artifact:
    """Index a CheckpointRegistry Manifest (bytes stay in that registry)."""
    return areg.register(ArtifactType.CHECKPOINT, manifest.ckpt_id, run_id=run_id,
                         parents=parents, metadata={"step": manifest.step,
                                                    "policy_id": manifest.policy_id,
                                                    **manifest.metadata})


def register_prompt_dataset(areg: ArtifactRegistry, meta, *, run_id=None) -> Artifact:
    return areg.register(ArtifactType.PROMPT_DATASET, meta.dataset_version,
                         run_id=run_id, hash=meta.hash,
                         metadata={"seed": meta.seed, "n_prompts": meta.n_prompts,
                                   "prompt_version": meta.prompt_version})


def register_policy(areg: ArtifactRegistry, policy_version, *, run_id=None,
                    parents=None) -> Artifact:
    return areg.register(ArtifactType.POLICY, f"v{policy_version.version}",
                         run_id=run_id, parents=parents,
                         hash=policy_version.fingerprint,
                         metadata=policy_version.metadata)


def register_benchmark(areg: ArtifactRegistry, name: str, *, uri=None,
                       run_id=None, parents=None, metadata=None) -> Artifact:
    return areg.register(ArtifactType.BENCHMARK, name, run_id=run_id, uri=uri,
                         parents=parents, metadata=metadata)


def register_eval_report(areg: ArtifactRegistry, name: str, *, uri=None,
                         run_id=None, parents=None, metadata=None) -> Artifact:
    return areg.register(ArtifactType.EVAL_REPORT, name, run_id=run_id, uri=uri,
                         parents=parents, metadata=metadata)
