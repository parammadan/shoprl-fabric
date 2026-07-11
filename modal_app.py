"""Run the GRPO training lifecycle on a Modal serverless GPU.

    modal run modal_app.py            # baseline -> train -> baseline -> push

One command spins up a T4, does before/after eval on the held-out split, and
pushes the LoRA checkpoint + eval JSONs to HF Hub. Reuses the T4-tuned config
(fp32: T4 has no bf16). Prereq: `modal secret create huggingface HF_TOKEN=hf_...`.

See docs repo MODAL_RUN.md for the full walkthrough.
"""
import modal

REPO = "https://github.com/parammadan/shoprl-fabric.git"
CONFIG = "configs/grpo_qwen_06b_kaggle.yaml"   # T4-tuned (fp32); works on Modal T4
CKPT = "checkpoints/step-150"                   # matches config steps=150
HF_REPO = "parammadan/shoprl-fabric-qwen06b-grpo"

# Deps first (CUDA torch on Modal's GPU hosts), then clone + install the package
# without deps so nothing downgrades torch. Bump CACHE_BUST to force a re-clone
# after pushing new commits.
CACHE_BUST = "2026-07-11e"
image = (
    modal.Image.debian_slim(python_version="3.12")  # package requires >=3.12
    .apt_install("git")
    .pip_install("torch", "transformers", "peft", "accelerate",
                 "pydantic>=2.6", "pyyaml", "huggingface_hub")
    .run_commands(
        f"echo {CACHE_BUST} && git clone {REPO} /root/shoprl",
        "pip install -e /root/shoprl --no-deps",
    )
)

app = modal.App("shoprl-grpo", image=image)


@app.function(gpu="T4", timeout=60 * 60 * 4,
              secrets=[modal.Secret.from_name("huggingface")])
def train(smoke: bool = False):
    import subprocess

    def run(*args):
        subprocess.run(list(args), cwd="/root/shoprl", check=True)

    import torch
    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    # smoke = cheap end-to-end validation (3 steps, tiny eval, isolated HF repo).
    cfg = "configs/smoke_gpu.yaml" if smoke else CONFIG
    ckpt = "checkpoints/step-3" if smoke else CKPT
    hf_repo = f"{HF_REPO}-smoke" if smoke else HF_REPO
    n = "4" if smoke else "32"

    # before (untrained) -> train -> after (trained), same held-out split
    run("python", "-m", "shoprl.eval.baseline", "--config", cfg,
        "--n-prompts", n, "--num-samples", "4", "--out", "outputs/before.json")
    run("python", "-m", "shoprl.train", "--config", cfg)
    run("python", "-m", "shoprl.eval.baseline", "--config", cfg,
        "--adapter", ckpt, "--n-prompts", n, "--num-samples", "4",
        "--out", "outputs/after.json")

    # push LoRA checkpoint + eval provenance to HF Hub (HF_TOKEN from the secret)
    from huggingface_hub import HfApi, create_repo
    create_repo(hf_repo, exist_ok=True, repo_type="model")
    api = HfApi()
    api.upload_folder(folder_path=f"/root/shoprl/{ckpt}", repo_id=hf_repo,
                      commit_message="GRPO smoke" if smoke else "GRPO run 1 (Modal T4)")
    api.upload_folder(folder_path="/root/shoprl/outputs", repo_id=hf_repo,
                      path_in_repo="eval")
    print(f"pushed adapter + eval -> https://huggingface.co/{hf_repo}")


@app.local_entrypoint()
def main(smoke: bool = False):
    train.remote(smoke=smoke)
