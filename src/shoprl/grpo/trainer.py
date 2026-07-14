"""Live GRPO training loop: rollout -> reward -> advantages -> KL -> loss -> step.

    python -m shoprl.grpo.trainer --config configs/train_dev.yaml

Small by design so the whole thing runs on an 8GB M1 with Qwen3-0.6B + LoRA.
Each step logs reward / KL / entropy / grad-norm so the dynamics are visible.

Data flow (one step):
  1. pick prompts -> build grounded task prompts (retrieve->shortlist)
  2. HFRolloutEngine.generate -> a group of completions per prompt
  3. compute_reward per completion (verifiable, vs catalog)
  4. group_advantages within each prompt's group -> A_i per completion
  5. one forward through the POLICY (LoRA on)  -> logp_policy (grad) + entropy
     one forward through the REFERENCE (LoRA off, no grad) -> logp_ref
     logp_old = logp_policy.detach()   (single update per batch -> ratio 1)
  6. grpo_loss -> backward -> clip grad norm -> optimizer.step()
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from shoprl.config import Config, load_config
from shoprl.data import generate_catalog, generate_prompts
from shoprl.data.catalog import catalog_index
from shoprl.grpo.advantages import batch_group_advantages
from shoprl.grpo.logprobs import build_batch, mean_entropy, token_logprobs
from shoprl.grpo.loss import grpo_loss
from shoprl.reward import RewardContext, compute_reward
from shoprl.rollout.base import Completion
from shoprl.rollout.hf import HFRolloutEngine, _resolve_device, _resolve_dtype
from shoprl.task import build_shortlist, build_task_prompt


def build_policy(config: Config, resume_from: str | None = None):
    """Load base model, wrap with LoRA. Only adapter params train; the frozen
    base doubles as the reference (LoRA starts as a no-op, so base == initial
    policy). If `resume_from` is a checkpoint dir, LOAD that adapter and continue
    from it (used by OOM recovery: restart from the last good checkpoint with a
    smaller batch). Returns (model, tokenizer, device)."""
    device = _resolve_device(config.model.device)
    dtype = _resolve_dtype(config.model.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(config.model.name, dtype=dtype)
    if resume_from:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, resume_from, is_trainable=True).to(device)
    else:
        lora = LoraConfig(
            r=config.training.lora_r,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=0.0,  # keep train/eval identical -> logp is well-defined
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora).to(device)

    # Gradient checkpointing: recompute layer activations during backward instead
    # of storing them. The attention activations scale with T^2 across all layers
    # and were the real OOM cause on the 16GB T4 (our task prompt is long); this
    # trades ~30% compute for a large memory cut. use_cache must be off for it;
    # the rollout engine re-enables the KV cache per generate() call.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()  # so grads flow with the frozen base
    return model, tokenizer, device


def _policy_logprobs(model, input_ids, attention_mask):
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    return token_logprobs(logits, input_ids), logits


def train_step(config, model, tokenizer, engine, catalog, idx, examples, device, step):
    tr = config.training
    # 1-2. pick prompts, build grounded task prompts, roll out.
    picks = [examples[(step * tr.prompts_per_step + j) % len(examples)]
             for j in range(tr.prompts_per_step)]
    task_prompts, contexts = [], []
    for ex in picks:
        sl = build_shortlist(ex, catalog, k=tr.shortlist, seed=config.experiment.seed)
        task_prompts.append(build_task_prompt(ex, idx, sl))
        contexts.append(RewardContext(catalog=idx, constraints=ex.constraints))

    # Rollout in EVAL mode with checkpointing OFF. Gradient checkpointing is a
    # training-forward optimization; if left on during generate() it force-nulls
    # the KV cache every decode step and corrupts autoregressive decoding ->
    # near-random tokens -> flat rewards -> zero gradient. (This was THE bug: the
    # model was generating in train() mode, so rollouts were garbage while eval,
    # which uses .eval(), scored fine.)
    model.gradient_checkpointing_disable()
    model.eval()
    # Fresh samples each step (vary the seed).
    groups = engine.generate(task_prompts, config.rollout.num_samples,
                             seed=config.experiment.seed + step + 1)

    # 3-4. reward each completion; group-relative advantages within each group.
    completions: list[Completion] = []
    rewards_per_group: list[list[float]] = []
    breakdowns = []
    for ctx, group in zip(contexts, groups):
        grp_rewards = []
        for comp in group.completions:
            r = compute_reward(comp.text, ctx,
                               weights=config.rewards.weights,
                               hallucination_penalty=config.rewards.hallucination_penalty)
            grp_rewards.append(r.total)
            breakdowns.append(r)
            completions.append(comp)
        rewards_per_group.append(grp_rewards)
    advantages_nested = batch_group_advantages(rewards_per_group)
    advantages = torch.tensor([a for grp in advantages_nested for a in grp],
                              dtype=torch.float32, device=device)
    flat_rewards = [r for grp in rewards_per_group for r in grp]

    # 5. batch + forward passes. Back to train mode + checkpointing ON so the
    # policy backward is memory-safe on the GPU.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    input_ids, attn, cmask = build_batch(completions, tokenizer.pad_token_id, device)
    mask_shift = cmask[:, 1:]

    # Reference forward FIRST, under no_grad, and free its logits before the
    # policy forward. This keeps the two [B,T,vocab] logits tensors from
    # coexisting -> roughly halves the logits memory peak (critical on a 16GB T4).
    with torch.no_grad():
        with model.disable_adapter():  # LoRA off -> frozen reference
            logp_ref, ref_logits = _policy_logprobs(model, input_ids, attn)
        del ref_logits

    model.train()
    logp_policy, logits = _policy_logprobs(model, input_ids, attn)
    entropy = mean_entropy(logits, mask_shift)
    logp_old = logp_policy.detach()  # single step/batch -> ratio == 1

    # 6. loss -> backward -> clip -> step.
    loss, stats = grpo_loss(
        logp_policy, logp_old, logp_ref, advantages, mask_shift,
        clip_eps=tr.clip_eps, beta=tr.beta,
    )
    optimizer = train_step.optimizer
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), tr.max_grad_norm
    )
    optimizer.step()

    comp_means = {
        f"reward_{k}": statistics.mean(getattr(b, k) for b in breakdowns)
        for k in ("budget", "groundedness", "coverage",
                  "quality_format", "quality_comparison")
    }
    return {
        "reward_mean": statistics.mean(flat_rewards),
        "reward_std": statistics.pstdev(flat_rewards) if len(flat_rewards) > 1 else 0.0,
        "loss": stats.loss,
        "kl": stats.mean_kl,
        "entropy": entropy,
        "grad_norm": float(grad_norm),
        "clip_frac": stats.clip_fraction,
        "ratio": stats.mean_ratio,
        "hallucination_rate": sum(b.hallucinated for b in breakdowns) / len(breakdowns),
        **comp_means,
    }


def save_checkpoint(model, config, step) -> str:
    path = os.path.join(config.training.ckpt_dir, f"step-{step}")
    model.save_pretrained(path)  # LoRA adapter only (small)
    with open(os.path.join(path, "train_state.json"), "w") as f:
        json.dump({"step": step, "model": config.model.name}, f)
    return path


def run_training(config: Config) -> str:
    """Run the full GRPO loop from a Config. Returns the checkpoint path."""
    tr = config.training

    print(f"[trainer] model={config.model.name} steps={tr.steps} "
          f"group={config.rollout.num_samples} prompts/step={tr.prompts_per_step}")

    catalog = generate_catalog(n=tr.catalog_size, seed=config.experiment.seed)
    idx = catalog_index(catalog)
    examples = generate_prompts(catalog, n=64, seed=config.experiment.seed)

    model, tokenizer, device = build_policy(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[trainer] device={device} LoRA trainable={trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    engine = HFRolloutEngine(config, model=model, tokenizer=tokenizer)
    train_step.optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=tr.lr
    )

    # Structured metrics log (append per step so a run can be watched live and
    # survives interruption). This is what the observability dashboard reads.
    run_dir = os.path.join("runs", config.experiment.name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()  # fresh file per run
    print(f"[trainer] metrics -> {metrics_path}")

    for step in range(tr.steps):
        m = train_step(config, model, tokenizer, engine, catalog, idx, examples,
                       device, step)
        m["step"] = step
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")
        print(
            f"step {step:>2} | reward {m['reward_mean']:+.3f}±{m['reward_std']:.3f} "
            f"| loss {m['loss']:+.4f} | kl {m['kl']:.4f} | entropy {m['entropy']:.3f} "
            f"| grad_norm {m['grad_norm']:.3f} | clipfrac {m['clip_frac']:.2f} "
            f"| ratio {m['ratio']:.3f}"
        )
        # Periodic checkpoint = spot-interruption safety on cloud GPU.
        if (step + 1) % tr.save_every == 0 and (step + 1) < tr.steps:
            save_checkpoint(model, config, step + 1)

    path = save_checkpoint(model, config, tr.steps)
    print(f"[trainer] saved checkpoint -> {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.grpo.trainer")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
