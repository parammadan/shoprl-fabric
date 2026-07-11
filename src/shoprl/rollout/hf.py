"""HuggingFace `generate` rollout engine — the M1/MPS development path.

This is deliberately the *reference* implementation, not the fast one. It runs
anywhere (CPU, Apple MPS, CUDA) so the entire platform is debuggable on a
laptop with a tiny model. On cloud GPU we swap in a vLLM engine behind the same
RolloutEngine interface; nothing else changes.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shoprl.config import Config
from shoprl.rollout.base import Completion, RolloutEngine, RolloutGroup


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    # bf16 on CUDA (fast, stable); float32 elsewhere. float32 on MPS/CPU avoids
    # the fp16 numerical surprises that would corrupt a correctness-first setup.
    return torch.bfloat16 if device == "cuda" else torch.float32


class HFRolloutEngine(RolloutEngine):
    def __init__(self, config: Config, model=None, tokenizer=None):
        """If `model`/`tokenizer` are given, sample from THAT model (e.g. the
        trainer's live policy) instead of loading a fresh one — so the same
        weights we update are the ones we roll out from."""
        self.config = config
        self.device = _resolve_device(config.model.device)
        self.dtype = _resolve_dtype(config.model.dtype, self.device)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model.name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                config.model.name, dtype=self.dtype
            ).to(self.device)
            model.eval()  # only force eval when we own the model
        self.model = model

    def _format(self, prompt: str) -> str:
        """Wrap a raw prompt as a chat turn if the tokenizer has a template.

        Qwen3 is a chat model; feeding raw text gives incoherent output. We also
        disable "thinking" mode so completions are the recommendation itself,
        not a long <think> trace — appropriate for our short-answer task.
        """
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Qwen3-specific; ignored elsewhere
                )
            except TypeError:
                # Non-Qwen template without enable_thinking kwarg.
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return prompt

    @torch.no_grad()
    def generate(
        self, prompts: list[str], num_samples: int, seed: int | None = None
    ) -> list[RolloutGroup]:
        # Seed for reproducibility; callers vary it per step to get fresh samples.
        torch.manual_seed(self.config.experiment.seed if seed is None else seed)
        rc = self.config.rollout

        groups: list[RolloutGroup] = []
        # Per-prompt loop: simple, low peak memory on M1, and grouping is
        # trivial. Batching across prompts is a later optimization.
        for prompt in prompts:
            text = self._format(prompt)
            enc = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_len = enc["input_ids"].shape[1]

            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=rc.temperature,
                top_p=rc.top_p,
                max_new_tokens=rc.max_new_tokens,
                num_return_sequences=num_samples,
                use_cache=True,  # KV cache for fast rollout (training sets config off)
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_ids = enc["input_ids"][0].tolist()
            comps: list[Completion] = []
            for seq in out:
                completion_ids = seq[input_len:].tolist()
                text_out = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True
                ).strip()
                comps.append(
                    Completion(
                        prompt=prompt,
                        text=text_out,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=completion_ids,
                    )
                )
            groups.append(RolloutGroup(prompt=prompt, completions=comps))
        return groups
