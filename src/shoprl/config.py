"""Config system: one YAML file fully describes an experiment.

Everything downstream (rollout, reward, learner, checkpointing) reads from the
Config object loaded here. Pydantic gives us validation + clear errors when a
YAML is malformed, and typed access everywhere else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str = "shoprl-dev"
    seed: int = 0


class ModelConfig(BaseModel):
    name: str = "Qwen/Qwen3-0.6B"
    # "auto" picks bf16/fp16 where sane; resolved by the engine, not here.
    dtype: str = "auto"
    # "auto" -> mps if available else cpu. The engine resolves the concrete
    # device so config stays portable between M1 and cloud GPU.
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class RolloutConfig(BaseModel):
    # Which RolloutEngine implementation to use. "stub" needs no ML deps and is
    # for tests/plumbing; "hf" is the M1 path; "vllm" is the cloud drop-in.
    engine: Literal["stub", "hf", "vllm"] = "hf"
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    # Group size: how many completions we sample per prompt. GRPO/RLOO compute
    # advantages *within* this group, so this is an RL hyperparameter, not just
    # a generation knob. Keeping it here makes that coupling explicit.
    num_samples: int = Field(default=4, ge=1)


class Config(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate an experiment config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
