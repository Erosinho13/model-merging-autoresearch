"""
Run the baseline source on Modal; see README.md for setup.

Run with:

python -m modal run modal_app.py --method knots_ties

Optional GPU and dataset checks are in modal_diagnostics.py.
"""

import json
import os
from pathlib import Path

import modal
import sys
import subprocess
from configs.nlp_benchmark import ADAPTERS

ROOT = Path(__file__).resolve().parent
app = modal.App("lora-merging-research")
cache = modal.Volume.from_name("lora-merging-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("lora-merging-results", create_if_missing=True)

# See requirements-modal.txt and README.md for runtime and setup details.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.2.2", "torchvision==0.17.2")
    .pip_install_from_requirements(str(ROOT / "requirements-modal.txt"))
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/workspace",
          "LORA_REPOSITORY_ROOT": "/workspace",
          "TOKENIZERS_PARALLELISM": "false"})
)
# Explicit source allowlist: never upload credentials, Git metadata or local data.
for directory in ("configs", "dataset", "eval_scripts", "private_eval"):
    image = image.add_local_dir(
        ROOT / directory, f"/workspace/{directory}",
        ignore=["**/__pycache__/**", "**/.DS_Store"],
    )
for filename in (
    "utils.py", "ft_handlers.py", "masking_ops.py", "merging_functions.py",
    "task_merger.py", "baseline_configs.py", "merging_types.py",
    # modal_runner imports this module during remote function hydration.
    "modal_app.py", "requirements-modal.txt",
):
    image = image.add_local_file(ROOT / filename, f"/workspace/{filename}")


@app.function(image=image, cpu=2, memory=8192, timeout=3600,
              max_containers=1, scaledown_window=2, volumes={"/cache": cache},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def download_weights():
    """Cache weights and validate saved classification heads on CPU."""
    # Defer worker dependencies until Modal executes this function remotely.
    import torch
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/workspace")
    repos = [*ADAPTERS, "hoffman-lab/KnOTS-Llama3_8B_lora_R16_pretrained_model",
             "meta-llama/Meta-Llama-3-8B"]
    report = {}
    try:
        for repo in repos:
            print(f"Caching {repo} on CPU", flush=True)
            folder = Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"]))
            report[repo] = folder.name
            if repo in ADAPTERS:
                if (folder / "adapter_model.safetensors").exists():
                    from safetensors.torch import load_file
                    state = load_file(str(folder / "adapter_model.safetensors"))
                else:
                    state = torch.load(folder / "adapter_model.bin", map_location="cpu", weights_only=True)
                heads = [v for k, v in state.items() if k.endswith("score.weight")]
                if len(heads) != 1 or heads[0].shape != (3, 4096):
                    raise RuntimeError(f"{repo}: saved classifier unavailable; heads.pt required")
                del state, heads
        Path("/cache/checkpoints-ready.json").write_text(json.dumps(report, indent=2))
        return {"revisions": report, "uses_gpu": False, "saved_heads": "verified"}
    finally:
        cache.commit()


@app.function(image=image, cpu=1, memory=2048, timeout=60,
              max_containers=1, scaledown_window=2, volumes={"/cache": cache})
def check_checkpoints_ready():
    """Verify offline checkpoint files on CPU before requesting a GPU."""
    # Keep cache checks independent of local Hugging Face dependencies.
    from huggingface_hub import snapshot_download, hf_hub_download
    if not Path("/cache/checkpoints-ready.json").exists():
        raise RuntimeError("Checkpoints are not ready. Run: python -m modal run modal_app.py --download")
    revisions = json.loads(Path("/cache/checkpoints-ready.json").read_text())
    for repo, revision in revisions.items():
        folder = Path(snapshot_download(repo, local_files_only=True))
        if folder.name != revision:
            raise RuntimeError(f"Cached revision changed for {repo}; rerun --download")
        is_base = repo == "meta-llama/Meta-Llama-3-8B"
        config_name = "config.json" if is_base else "adapter_config.json"
        hf_hub_download(repo, config_name, local_files_only=True)
        if is_base:
            indices = list(folder.glob("*model*.index.json"))
            if indices:
                for index in indices:
                    for filename in set(json.loads(index.read_text())["weight_map"].values()):
                        if not (folder / filename).is_file():
                            raise RuntimeError(f"Missing cached weight shard: {repo}/{filename}")
            elif not any((folder / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
                raise RuntimeError(f"Missing cached model weights: {repo}")
        elif not any((folder / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise RuntimeError(f"Missing cached adapter weights: {repo}")
    return revisions


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=196608,
              timeout=3600, max_containers=1, scaledown_window=2,
              volumes={"/cache": cache, "/results": results},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def evaluate_fixed(method: str, split: str = "test", scale: float = 0.0, top_k: float = 0.0,
                   dare_pruning_coeffs: float = -1.0):
    """Evaluate one fixed baseline with cached weights; no online downloads."""
    # Remote-only evaluator imports stay out of local app initialization.
    from eval_scripts.nli_fixed_test import EVALUATIONS, fixed_config
    if method not in EVALUATIONS:
        raise ValueError("Unsupported baseline")
    if not Path("/cache/checkpoints-ready.json").exists():
        raise RuntimeError("Run --download first to cache weights and verify heads")
    os.chdir("/workspace")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    fixed_config(method, scale=scale if scale else None, top_k=top_k if top_k else None,
                 dare_pruning_coeffs=None if dare_pruning_coeffs == -1 else dare_pruning_coeffs)
    output_dir = "/results/fixed-test" if split == "test" and scale == 0 and top_k == 0 and dare_pruning_coeffs == -1 else f"/results/hp/{method}/{split}/scale-{scale}_topk-{top_k}"
    if dare_pruning_coeffs != -1:
        output_dir += f"_p-{dare_pruning_coeffs}"
    try:
        command = [sys.executable, "-m", "eval_scripts.nli_fixed_test",
                   "--method", method, "--split", split, "--output-dir", output_dir]
        if scale:
            command += ["--scale", str(scale)]
        if top_k:
            command += ["--top-k", str(top_k)]
        if dare_pruning_coeffs != -1:
            command += ["--dare-pruning-coeffs", str(dare_pruning_coeffs)]
        subprocess.run(command, check=True)
        return json.loads(Path(output_dir, f"{method}.json").read_text())
    finally:
        results.commit()


@app.local_entrypoint()
def main(download: bool = False, method: str = "", split: str = "test",
         scale: float = 0.0, top_k: float = 0.0, dare_pruning_coeffs: float = -1.0):
    if method:
        from eval_scripts.nli_fixed_test import fixed_config
        fixed_config(method, scale=scale if scale else None, top_k=top_k if top_k else None,
                     dare_pruning_coeffs=None if dare_pruning_coeffs == -1 else dare_pruning_coeffs)
        if split not in {"val", "test"}:
            raise ValueError("split must be val or test")
        check_checkpoints_ready.remote()
        print(json.dumps(evaluate_fixed.remote(method, split, scale, top_k, dare_pruning_coeffs), indent=2))
    elif download:
        print(json.dumps(download_weights.remote(), indent=2))
    else:
        raise ValueError("Choose --download or --method. Optional checks: modal_diagnostics.py")
