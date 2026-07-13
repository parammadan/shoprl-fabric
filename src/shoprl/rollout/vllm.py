"""vLLM rollout engine — the cloud-GPU drop-in behind the RolloutEngine interface.

Rollout dominates RL wall-clock (measured repeatedly this project); vLLM's paged
attention + continuous batching generates far faster than HF `generate`. Same
interface as HFRolloutEngine, so the learner/reward/loss code is unchanged — only
`config.rollout.engine: vllm` differs.

GPU-only: `vllm` is imported lazily inside __init__, so this module imports fine
on an M1 (for tests/typing); instantiating it without a CUDA GPU + vllm installed
raises. Not runnable on M1 by design.

Caveat (documented, not hidden): vLLM holds its OWN copy of the weights. For
on-policy training where the policy changes every step, you must sync the updated
(LoRA) weights into the vLLM engine each step (weight reload / collective) — a
further piece. As-is this engine is a fast drop-in for the rollout benchmark,
baseline/after eval, and fixed-policy generation; wiring live weight-sync into
the training loop is the follow-up.
"""
from __future__ import annotations

from shoprl.config import Config
from shoprl.rollout.base import Completion, RolloutEngine, RolloutGroup


class VLLMRolloutEngine(RolloutEngine):
    def __init__(self, config: Config, model=None, tokenizer=None):
        from transformers import AutoTokenizer  # light
        from vllm import LLM  # lazy, GPU-only

        self.config = config
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.model.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = config.model.dtype if config.model.dtype != "auto" else "auto"
        # Modest memory fraction so vLLM can COEXIST with the HF LoRA trainer on
        # one GPU during the benchmark (0.6B is tiny; ~half of 24GB is ample).
        self.llm = model or LLM(model=config.model.name, dtype=dtype,
                                gpu_memory_utilization=0.5)
        self.last_ttft_ms: float | None = None   # set per generate() for the bench

    def _format(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            msgs = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate(self, prompts, num_samples, seed=None):
        from vllm import SamplingParams

        rc = self.config.rollout
        sp = SamplingParams(n=num_samples, temperature=rc.temperature, top_p=rc.top_p,
                            max_tokens=rc.max_new_tokens, seed=seed)
        texts = [self._format(p) for p in prompts]
        outs = self.llm.generate(texts, sp)  # continuous-batched across all prompts

        # TTFT: mean (first_token - arrival) across requests, when vLLM reports
        # metrics. Best-effort — never fabricated; stays None if unavailable.
        try:
            ttfts = [o.metrics.first_token_time - o.metrics.arrival_time
                     for o in outs if getattr(o, "metrics", None)
                     and o.metrics.first_token_time and o.metrics.arrival_time]
            self.last_ttft_ms = round(1000 * sum(ttfts) / len(ttfts), 1) if ttfts else None
        except Exception:
            self.last_ttft_ms = None

        groups: list[RolloutGroup] = []
        for prompt, o in zip(prompts, outs):
            prompt_ids = list(o.prompt_token_ids)
            comps = [
                Completion(prompt=prompt, text=c.text,
                           prompt_token_ids=prompt_ids,
                           completion_token_ids=list(c.token_ids))
                for c in o.outputs
            ]
            groups.append(RolloutGroup(prompt=prompt, completions=comps))
        return groups
