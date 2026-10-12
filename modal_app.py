"""
Run the KnOTS source on Modal; see MODAL.md for setup.

Run with:

python -m modal run modal_app.py

or

python -m modal run modal_app.py --module eval_scripts.nli_pertask_linearsearch
"""

import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
app = modal.App("lora-merging-research")
cache = modal.Volume.from_name("lora-merging-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("lora-merging-results", create_if_missing=True)

# See requirements-modal.txt and MODAL.md for reproduction constraints.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.2.2", "torchvision==0.17.2")
    .pip_install_from_requirements(str(ROOT / "requirements-modal.txt"))
    .env({"HF_HOME": "/cache/huggingface", "PYTHONPATH": "/workspace",
          "TOKENIZERS_PARALLELISM": "false"})
)
# Explicit source allowlist: never upload credentials, Git metadata or local data.
for directory in ("configs", "dataset", "models", "eval_scripts", "training_scripts"):
    image = image.add_local_dir(
        ROOT / directory, f"/workspace/{directory}",
        ignore=["**/__pycache__/**", "**/.DS_Store"],
    )
for filename in (
    "utils.py", "ft_handlers.py", "masking_ops.py", "merging_functions.py",
    "task_merger.py", "singular_vals_trainer.py",
):
    image = image.add_local_file(ROOT / filename, f"/workspace/{filename}")


@app.function(image=image, gpu="L4", timeout=300, max_containers=1,
              scaledown_window=2)
def gpu_check():
    """Verify CUDA, source upload and merging imports without downloading models."""
    import sys
    import torch

    sys.path.insert(0, "/workspace")
    import utils
    import task_merger
    from merging_functions import tv_merging

    if not torch.cuda.is_available():
        raise RuntimeError("Modal worker has no CUDA device")
    vectors = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
    merged, _, _ = tv_merging(list(vectors), merging_type="sum")
    torch.testing.assert_close(merged, torch.tensor([4., 6.], device="cuda"))
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "repository_imports": "ok",
            "merging_check": "ok"}


@app.function(
    image=image, gpu="A100-80GB", cpu=8, memory=131072,
    timeout=3600, max_containers=1, scaledown_window=2,
    volumes={"/cache": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def run_module(module: str):
    """Run an existing script with persistent HF cache and CSV output."""
    import subprocess
    import sys

    if module not in {"eval_scripts.nli_pertask_linearsearch"}:
        raise ValueError("Only the NLP evaluation module is currently supported")
    os.environ["HUGGINGFACE_TOKEN"] = os.environ["HF_TOKEN"]
    os.chdir("/workspace")
    Path("/results/csvs_new").mkdir(parents=True, exist_ok=True)
    if not Path("csvs_new").exists():
        Path("csvs_new").symlink_to("/results/csvs_new", target_is_directory=True)
    try:
        subprocess.run([sys.executable, "-m", module], check=True)
    finally:
        cache.commit()
        results.commit()
    return {"results_volume": "lora-merging-results"}


@app.function(
    image=image, cpu=2, memory=8192, timeout=900, max_containers=1,
    scaledown_window=2, volumes={"/cache": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def prepare_data():
    """CPU-only dataset checks, tokenizer access and adapter metadata inspection."""
    import importlib.metadata
    import sys
    import datasets
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    sys.path.insert(0, "/workspace")
    from configs.nlp_benchmark import TASKS, ADAPTERS

    report = {"uses_gpu": False, "datasets": {}, "adapters": {},
              "versions": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "peft", "datasets", "huggingface-hub")}}
    try:
        for task, repo, name in (
            ("snli", "stanfordnlp/snli", None),
            ("mnli", "SetFit/mnli", None),
            ("scitail", "allenai/scitail", "tsv_format"),
        ):
            print(f"Checking {task} on CPU", flush=True)
            data = datasets.load_dataset(repo, name)
            # MNLI evaluation partitions its validation split, as in dataset/mnli.py.
            splits = ("train", "validation") if task == "mnli" else ("train", "validation", "test")
            for split in splits:
                if split not in data or not len(data[split]):
                    raise RuntimeError(f"{task}: missing or empty {split}")
            report["datasets"][task] = {s: len(d) for s, d in data.items()}
        AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        report["tokenizer_access"] = "ok"
        api = HfApi()
        for task, repo in zip(TASKS, ADAPTERS):
            info = api.model_info(repo)
            path = hf_hub_download(repo, "adapter_config.json", revision=info.sha)
            cfg = json.loads(Path(path).read_text())
            report["adapters"][task] = {"revision": info.sha,
                "rank": cfg.get("r"), "base_model": cfg.get("base_model_name_or_path"),
                "modules_to_save": cfg.get("modules_to_save"),
                "files": [s.rfilename for s in info.siblings]}
        report["model_weights_downloaded"] = False
        report["classification_heads"] = "not verified yet"
        Path("/results/preparation.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        cache.commit()
        results.commit()


@app.function(image=image, cpu=2, memory=8192, timeout=3600,
              max_containers=1, scaledown_window=2, volumes={"/cache": cache},
              secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])])
def download_weights():
    """Cache weights and validate saved classification heads on CPU."""
    import sys
    import torch
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/workspace")
    from configs.nlp_benchmark import ADAPTERS
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
def evaluate_fixed(method: str, split: str = "test", scale: float = 0.0, top_k: float = 0.0):
    """Evaluate one fixed baseline with cached weights; no online downloads."""
    import subprocess
    import sys
    if method not in {"average", "task_arithmetic", "ties", "dare_ties", "base", "snli", "mnli", "scitail"}:
        raise ValueError("Unsupported baseline")
    if not Path("/cache/checkpoints-ready.json").exists():
        raise RuntimeError("Run --download first to cache weights and verify heads")
    os.chdir("/workspace")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    output_dir = "/results/fixed-test" if split == "test" and scale == 0 and top_k == 0 else f"/results/hp/{method}/{split}/scale-{scale}_topk-{top_k}"
    try:
        command = [sys.executable, "-m", "eval_scripts.nli_fixed_test",
                   "--method", method, "--split", split, "--output-dir", output_dir]
        if scale:
            command += ["--scale", str(scale)]
        if top_k:
            command += ["--top-k", str(top_k)]
        subprocess.run(command, check=True)
        return json.loads(Path(output_dir, f"{method}.json").read_text())
    finally:
        results.commit()


@app.local_entrypoint()
def main(check: bool = True, module: str = "", prepare: bool = False,
         download: bool = False, method: str = "", split: str = "test",
         scale: float = 0.0, top_k: float = 0.0):
    if method:
        from eval_scripts.nli_fixed_test import fixed_config
        fixed_config(method, scale=scale if scale else None, top_k=top_k if top_k else None)
        if split not in {"val", "test"}:
            raise ValueError("split must be val or test")
        check_checkpoints_ready.remote()
        print(json.dumps(evaluate_fixed.remote(method, split, scale, top_k), indent=2))
    elif download:
        print(json.dumps(download_weights.remote(), indent=2))
    elif module:
        raise RuntimeError("Evaluation is disabled until task heads, three-task configuration "
                           "and GPU memory requirements are verified. Run --prepare instead.")
    elif prepare:
        print(json.dumps(prepare_data.remote(), indent=2))
    elif check:
        print(json.dumps(gpu_check.remote(), indent=2))
