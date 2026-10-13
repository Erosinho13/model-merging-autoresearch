"""Modal entry point for candidate validation and operator-only final tests."""
import hashlib
import json
import uuid
from pathlib import Path

import modal
from modal_app import image, cache, results, check_checkpoints_ready, app as baseline_app

ROOT = Path(__file__).resolve().parent
app = modal.App("lora-research-agent")
io_volume = modal.Volume.from_name("lora-candidate-io", create_if_missing=True)
grader_image = image.add_local_dir(ROOT, "/workspace/research_agent",
                                   ignore=["references/**", "**/__pycache__/**"])
worker_image = (modal.Image.debian_slim(python_version="3.10")
                .pip_install("torch==2.2.2", "numpy==1.26.4", "safetensors==0.4.1")
                .add_local_file(ROOT / "worker.py", "/worker.py"))


@app.function(image=grader_image, gpu="A100-40GB", cpu=8, memory=196608,
              timeout=3600, max_containers=1, scaledown_window=2,
              volumes={"/cache": cache, "/results": results, "/candidate-io": io_volume})
def evaluate_candidate(source: str, split: str = "val"):
    import os
    import sys
    from safetensors.torch import load_file, save_file
    if split not in {"val", "test"}:
        raise ValueError("Invalid split")
    if not Path("/cache/checkpoints-ready.json").exists():
        raise RuntimeError("Run baseline --download first")
    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    from research_agent.bridge import evaluate
    experiment = uuid.uuid4().hex
    output = Path("/results/agent", experiment)
    output.mkdir(parents=True)
    (output / "candidate.py").write_text(source)
    metadata = {"experiment_id": experiment, "split": split,
                "sha256": hashlib.sha256(source.encode()).hexdigest()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def execute(code, base, vectors):
        folder = Path("/candidate-io", experiment)
        folder.mkdir(parents=True)
        (folder / "candidate.py").write_text(code)
        tensors = {f"base/{k}": v.contiguous() for k, v in base.items()}
        tensors.update({f"task{i}/{k}": v.detach().contiguous()
                        for i, vector in enumerate(vectors) for k, v in vector.items()})
        save_file(tensors, str(folder / "inputs.safetensors"))
        del tensors
        io_volume.commit()
        sandbox = modal.Sandbox.create(
            "python", "/worker.py", "/data", app=app, image=worker_image,
            cpu=8, memory=65536, timeout=600, block_network=True,
            volumes={"/data": io_volume.with_mount_options(sub_path=f"/{experiment}")})
        try:
            sandbox.wait()
            if sandbox.returncode != 0:
                raise RuntimeError(f"Candidate sandbox exited with code {sandbox.returncode}")
        finally:
            sandbox.terminate(wait=True)
        io_volume.reload()
        weights = folder / "output.safetensors"
        if weights.is_symlink() or not weights.is_file():
            raise ValueError("Missing regular candidate output file")
        # Tensor-only interchange: never deserialize candidate-controlled pickle.
        return load_file(str(weights))

    try:
        report = evaluate(source, str(output), execute, split)
        report.update(metadata)
        (output / "result.json").write_text(json.dumps(report, indent=2))
        return report
    except Exception as error:
        (output / "error.json").write_text(json.dumps({**metadata, "error": str(error)}))
        raise
    finally:
        results.commit()


@app.local_entrypoint()
def main(candidate: str = "research_agent/average.py", split: str = "val"):
    if split not in {"val", "test"}:
        raise ValueError("Invalid split")
    source = Path(candidate).read_text()
    # CPU cache verification uses the existing baseline app.
    with baseline_app.run():
        check_checkpoints_ready.remote()
    print(json.dumps(evaluate_candidate.remote(source, split), indent=2))
