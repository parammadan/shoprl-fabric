"""RolloutEngine interface + the data types that cross it.

Design note: RL post-training samples a *group* of completions per prompt
(GRPO/RLOO compute advantages within that group). So the engine's unit of work
is a prompt -> RolloutGroup, and a batch is list[str] -> list[RolloutGroup].
Keeping this shape in the interface means the learner never has to reshape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Completion:
    """One sampled continuation of a prompt."""

    prompt: str
    text: str
    # Token ids for the completion (and prompt) — the learner needs these to
    # recompute log-probs under the current policy during the update. The stub
    # engine leaves them empty; real engines (HF, vLLM) fill them.
    prompt_token_ids: list[int] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)


@dataclass
class RolloutGroup:
    """All completions sampled for a single prompt (the GRPO group)."""

    prompt: str
    completions: list[Completion]

    def __len__(self) -> int:
        return len(self.completions)


class RolloutEngine(ABC):
    """Samples completions from a policy.

    Implementations: StubRolloutEngine (no deps, tests), HFRolloutEngine
    (transformers `generate`, M1/MPS), and later a vLLM engine for cloud GPU.
    All are interchangeable behind this method.
    """

    @abstractmethod
    def generate(
        self, prompts: list[str], num_samples: int, seed: int | None = None
    ) -> list[RolloutGroup]:
        """Return one RolloutGroup per prompt, each with `num_samples` completions.

        `seed` lets a caller vary sampling across calls (e.g. per training step)
        or fix it for reproducibility. None = use the engine's own default.
        """
        raise NotImplementedError
