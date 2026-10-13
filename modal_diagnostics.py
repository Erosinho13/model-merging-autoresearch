"""Optional Modal diagnostics; baseline runs use modal_app.py."""
import json
from pathlib import Path

import modal
from modal_app import image, cache, results
import sys
import importlib.metadata
from configs.nlp_benchmark import TASKS, ADAPTERS

app = modal.App("lora-merging-diagnostics")


@app.function(image=image, gpu="L4", timeout=300, max_containers=1,
              scaledown_window=2)
def gpu_check():
    """Verify CUDA, source upload and merging imports without downloading models."""
    # Load GPU dependencies inside the Modal worker, not on app initialization.
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
    image=image, cpu=2, memory=8192, timeout=900, max_containers=1,
    scaledown_window=2, volumes={"/cache": cache, "/results": results},
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
def prepare_data():
    """CPU-only dataset checks, tokenizer access and adapter metadata inspection."""
    # Load dataset dependencies only when this remote diagnostic runs.
    import datasets
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    sys.path.insert(0, "/workspace")

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


@app.local_entrypoint()
def main(prepare: bool = False):
    """Default: GPU smoke check. Use --prepare for CPU dataset/access checks."""
    if prepare:
        print(json.dumps(prepare_data.remote(), indent=2))
    else:
        print(json.dumps(gpu_check.remote(), indent=2))
