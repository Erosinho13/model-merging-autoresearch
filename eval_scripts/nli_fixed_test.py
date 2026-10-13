"""Validation/test evaluation of released NLP merging configurations."""

import argparse
import gc
import importlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from time import monotonic
from copy import deepcopy
from pathlib import Path

from configs.nlp_benchmark import TASKS, ADAPTERS, select_benchmark
from baseline_configs import BASELINES
import dataset.configs as defaults

METHODS = {
    "average": "llama8B_r16_tv",
    "task_arithmetic": "llama8B_r16_tv",
    "ties": "llama8B_r16_ties",
    "dare_ties": "llama8B_r16_dare_ties",
    "knots_ties": "llama8B_r16_knots_ties",
    "knots_dare_ties": "llama8B_r16_knots_dare_ties",
}
DARE_SEEDS = (420, )
EVALUATIONS = (*METHODS, "base", *TASKS)


def progress(message):
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"[{timestamp}] {message}", flush=True)


def fixed_config(method, seed=420, scale=None, top_k=None, dare_pruning_coeffs=None):
    if method not in EVALUATIONS:
        raise ValueError(f"Unknown evaluation: {method}")
    config = select_benchmark(importlib.import_module(f"configs.{METHODS.get(method, 'llama8B_r16_tv')}").config)
    if method in BASELINES:
        config["task_merge_config"] = deepcopy(BASELINES[method])
    # Empty-string cache_dir overrides HF_HOME; None uses the same default cache
    # as snapshot_download in the CPU preparation step.
    config["model"]["cachedir"] = None
    # A blank path means compute the KnOTS alignment in memory, not torch.load("").
    if not config["task_merge_config"].get("ingredients_path"):
        config["task_merge_config"]["ingredients_path"] = None
    if method == "average":
        config["task_merge_config"]["scaling_coeffs"] = [1.0 / len(TASKS)]
    if scale is not None:
        if method not in METHODS or method == "average" or scale <= 0:
            raise ValueError("Positive scale overrides apply only to merge methods other than average")
        config["task_merge_config"]["scaling_coeffs"] = [scale]
    if top_k is not None:
        if method not in {"ties", "dare_ties", "knots_ties", "knots_dare_ties"} or not 1 < top_k <= 100:
            raise ValueError("top-k must be a retention percentage in (1, 100] for TIES/DARE-TIES")
        config["task_merge_config"]["topK"] = top_k
    if dare_pruning_coeffs is not None:
        if not config["task_merge_config"].get("dare") or not 0 <= dare_pruning_coeffs < 1:
            raise ValueError("dare-pruning-coeffs must be in [0, 1) and applies only to DARE-enabled methods")
        config["task_merge_config"]["dare_pruning_coeffs"] = dare_pruning_coeffs
    # Keep the pruning seed explicit; no validation-based seed selection.
    if config["task_merge_config"].get("dare"):
        config["task_merge_config"]["dare_seed"] = seed
    return config


def load_unmerged(selection, raw):
    """Load one backbone and recover the same task heads used for merging."""
    # Defer ML imports: fixed_config is also used before worker offline setup.
    import torch
    from transformers import AutoModelForSequenceClassification
    from peft import PeftModel

    heads = {}
    for task, repo in zip(TASKS, ADAPTERS):
        from huggingface_hub import snapshot_download
        folder = Path(snapshot_download(repo, local_files_only=True))
        if (folder / "adapter_model.safetensors").exists():
            from safetensors.torch import load_file
            state = load_file(str(folder / "adapter_model.safetensors"))
        else:
            state = torch.load(folder / "adapter_model.bin", map_location="cpu", weights_only=True)
        candidates = [v for k, v in state.items() if k.endswith("score.weight")]
        if len(candidates) != 1 or candidates[0].shape != (3, 4096):
            raise RuntimeError(f"Missing or ambiguous saved head for {task}")
        heads[task] = candidates[0].clone()
        del state, candidates
    progress(f"Loading one unmerged backbone: {selection}")
    model = AutoModelForSequenceClassification.from_pretrained(
        raw["model"]["name"], num_labels=3, cache_dir=None, return_dict=True)
    if selection != "base":
        repo = ADAPTERS[TASKS.index(selection)]
        model = PeftModel.from_pretrained(model, repo)
    return model, heads


def run(method, output_dir, seed=420, split="test", scale=None, top_k=None, dare_pruning_coeffs=None):
    started = monotonic()
    progress(f"Starting {method}, seed {seed}; importing ML libraries")
    # Defer ML imports: fixed_config is also used before worker offline setup.
    import torch
    from utils import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Llama evaluation")
    set_seed(seed)
    raw = fixed_config(method, seed, scale, top_k, dare_pruning_coeffs)
    print(json.dumps(raw["task_merge_config"], indent=2), flush=True)
    if method not in METHODS:
        merged, heads = load_unmerged(method, raw)
        progress("Unmerged model loaded; skipping merging")
    else:
        merged, heads = load_merged(method, raw)
    gc.collect()
    merged.config.pad_token_id = 128001
    merged.config.use_cache = False
    merged.config.pretraining_tp = 1
    progress("Moving evaluation model to GPU")
    merged.to("cuda").eval()
    progress("Evaluation model ready on GPU")
    report = {"method": method, "config": METHODS.get(method), "split": split,
              "settings": raw["task_merge_config"] if method in METHODS else None,
              "merged": method in METHODS, "backbone": method,
              "tasks": {}, "head_source": "released task adapters", "seed": seed,
              "dtype": str(next(merged.parameters()).dtype)}
    return evaluate_tasks(merged, heads, raw, report, method, output_dir, seed, started)


def load_merged(method, raw):
    # Defer ML imports: fixed_config is also used before worker offline setup.
    import torch
    from ft_handlers import LoRAHandler
    from private_eval.knots_merger import get_merge_handler
    from private_eval.model_utils import prepare_llama

    progress("Loading base model and three adapters on CPU")
    models = prepare_llama(raw["model"], "cpu")
    progress("Models loaded; extracting task classification heads")
    # Recover the original task heads from the released adapters. Fail rather
    # than silently using a randomly initialized classifier.
    heads = {}
    for task, model in zip(TASKS, models["bases"]):
        matches = [(n, p.detach().cpu().clone()) for n, p in model.named_parameters()
                   if "score.modules_to_save.default.weight" in n]
        if len(matches) != 1 or matches[0][1].shape != (3, 4096):
            raise RuntimeError(f"Cannot recover an unambiguous 3x4096 head for {task}")
        heads[task] = matches[0][1]
        progress(f"Recovered {task} head: {tuple(heads[task].shape)}")
    merge_config = raw["task_merge_config"]
    aligned = merge_config["representation"] == "svd-vector"
    # Accelerate the released float64 SVD on CUDA; source models stay on CPU.
    merge_device = "cuda" if aligned and torch.cuda.is_available() else "cpu"
    merger_class = get_merge_handler(merge_config["representation"])
    merger = merger_class(models["bases"], models["new"], LoRAHandler,
                          device=merge_device, merge_config=merge_config)
    scaling = raw["task_merge_config"]["scaling_coeffs"][0]
    merger.set_scaling_coeffs(float(scaling))
    with torch.no_grad():
        if aligned:
            progress(f"Computing KnOTS SVD alignment on {merge_device}; this may take several minutes")
            merger.transform(merge_config)
            progress("KnOTS alignment complete")
        progress(f"Merging task updates using {method}")
        merged = merger.merge(merge_config)
    progress("Merge complete; releasing source models")
    del merger, models
    if aligned and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged, heads


def evaluate_tasks(merged, heads, raw, report, method, output_dir, seed, started):
    # Defer ML imports: fixed_config is also used before worker offline setup.
    import torch
    from private_eval.model_utils import evaluate_logits
    for dataset in raw["dataset"]:
        task = dataset["name"]
        task_started = monotonic()
        progress(f"Preparing {task} {report['split']} loader (tokenization/cache lookup)")
        # Use original test loaders/preprocessing; avoid training-loader setup.
        data_config = deepcopy(getattr(defaults, task))
        data_config.update(dataset)
        loader = importlib.import_module(f"dataset.{task}").prepare_test_loaders(data_config)[report["split"]]
        progress(f"{task} loader ready: {len(loader.dataset)} examples, {len(loader)} batches")
        parameters = [p for n, p in merged.named_parameters()
                      if "score.modules_to_save.default.weight" in n or n == "score.weight"]
        if len(parameters) != 1:
            raise RuntimeError("Evaluation model classification head is ambiguous")
        with torch.no_grad():
            parameters[0].copy_(heads[task])
        progress(f"Evaluating {task} on GPU with its original classification head")
        accuracy = evaluate_logits(merged, loader, "cuda", dataset["mask_class"])
        progress(f"{task} finished: accuracy={100 * accuracy:.3f}%; "
                 f"preparation + evaluation={monotonic() - task_started:.1f}s")
        report["tasks"][task] = {"accuracy_percent": 100 * accuracy,
                                  "examples": len(loader.dataset)}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, f"{method}_seed_{seed}.json").write_text(json.dumps(report, indent=2))
        progress(f"Saved partial results: {method}_seed_{seed}.json")
        del loader
        gc.collect()
    report["average_accuracy_percent"] = sum(
        r["accuracy_percent"] for r in report["tasks"].values()) / len(TASKS)
    report["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    Path(output_dir, f"{method}_seed_{seed}.json").write_text(json.dumps(report, indent=2))
    progress(f"Finished {method}, seed {seed}: average accuracy="
             f"{report['average_accuracy_percent']:.3f}%; elapsed={monotonic() - started:.1f}s")
    print(json.dumps(report, indent=2), flush=True)
    return report


def summarize(reports):
    """Compute variability across seeds, not across tasks or examples."""
    def stats(values):
        return {"mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None}

    return {"method": reports[0]["method"], "split": reports[0]["split"],
            "seeds": [r["seed"] for r in reports], "std_definition": "sample, ddof=1",
            "runs": reports,
            "tasks": {task: stats([r["tasks"][task]["accuracy_percent"] for r in reports])
                      for task in TASKS},
            "average_accuracy_percent": stats([r["average_accuracy_percent"] for r in reports])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=EVALUATIONS, required=True,
                        help="Merge method, base backbone, or individual task adapter")
    parser.add_argument("--output-dir", default="fixed-test-results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--top-k", type=float)
    parser.add_argument("--dare-pruning-coeffs", type=float,
                        help="DARE drop probability p in [0, 1)")
    parser.add_argument("--seed", type=int, choices=DARE_SEEDS,
                        help="Run one seed only; DARE methods use DARE_SEEDS")
    args = parser.parse_args()
    fixed_config(args.method, scale=args.scale, top_k=args.top_k,
                 dare_pruning_coeffs=args.dare_pruning_coeffs)
    if args.dry_run:
        seeds = [args.seed] if args.seed is not None else (DARE_SEEDS if fixed_config(args.method)["task_merge_config"].get("dare") else [420])
        print(json.dumps([fixed_config(args.method, seed, args.scale, args.top_k, args.dare_pruning_coeffs) for seed in seeds], indent=2))
    elif args.seed is not None:
        run(args.method, args.output_dir, args.seed, args.split, args.scale, args.top_k, args.dare_pruning_coeffs)
    else:
        seeds = DARE_SEEDS if fixed_config(args.method)["task_merge_config"].get("dare") else (420,)
        reports = []
        progress(f"Running {args.method} for seeds {list(seeds)}")
        # Separate processes release model/GPU memory fully between seeds.
        for index, seed in enumerate(seeds, start=1):
            progress(f"Launching run {index}/{len(seeds)}: seed {seed}")
            command = [sys.executable, "-m", "eval_scripts.nli_fixed_test",
                            "--method", args.method, "--output-dir", args.output_dir,
                            "--seed", str(seed), "--split", args.split]
            if args.scale is not None:
                command += ["--scale", str(args.scale)]
            if args.top_k is not None:
                command += ["--top-k", str(args.top_k)]
            if args.dare_pruning_coeffs is not None:
                command += ["--dare-pruning-coeffs", str(args.dare_pruning_coeffs)]
            subprocess.run(command, check=True)
            reports.append(json.loads(Path(args.output_dir, f"{args.method}_seed_{seed}.json").read_text()))
        summary = summarize(reports)
        Path(args.output_dir, f"{args.method}.json").write_text(json.dumps(summary, indent=2))
        progress(f"All runs complete; saved aggregate results: {args.method}.json")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
