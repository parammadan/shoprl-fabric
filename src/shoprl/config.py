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


class TrainingConfig(BaseModel):
    # Kept tiny by default so the whole loop runs on an 8GB M1. Scale on GPU.
    steps: int = Field(default=3, ge=1)
    prompts_per_step: int = Field(default=1, ge=1)  # groups per optimizer step
    lr: float = 1e-5
    clip_eps: float = 0.2       # PPO trust-region width
    beta: float = 0.04          # KL penalty coefficient
    max_grad_norm: float = 1.0
    # LoRA: only these adapter params get gradients + optimizer state.
    lora_r: int = 8
    lora_alpha: int = 16
    # Task/data knobs for building rollout prompts.
    catalog_size: int = 300
    shortlist: int = 6
    ckpt_dir: str = "checkpoints"


class RewardConfig(BaseModel):
    # Composite weights (must be paired with the reward functions). Defaults
    # mirror shoprl.reward.composite so behavior is unchanged unless overridden.
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "budget": 0.25,
            "groundedness": 0.25,
            "coverage": 0.25,
            "quality_format": 0.15,
            "quality_comparison": 0.10,
        }
    )
    hallucination_penalty: float = 0.50


class Config(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    # Which RL algorithm the unified entry point dispatches to.
    algorithm: Literal["grpo", "rloo", "ppo"] = "grpo"
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate an experiment config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)
