"""Sandbox entry point and shared tensor-only model reconstruction helper."""
import sys
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def load_model(inputs):
    """Reconstruct a model from tensor-only fixed inputs; also used by the grader."""
    inputs = Path(inputs)
    description = json.loads((inputs / "model.json").read_text())
    config = dict(description["config"])
    model_type = config.pop("model_type")
    # Build the architecture without randomly initializing a full 8B model.
    with init_empty_weights():
        model = get_peft_model(
            AutoModelForSequenceClassification.from_config(AutoConfig.for_model(model_type, **config)),
            LoraConfig(**description["adapter_config"]))
    print("[model loader] Loading cached model weights", flush=True)
    state = load_file(str(inputs / "model.safetensors"))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    return model


def main(folder, inputs=None):
    folder = Path(folder)
    inputs = Path(inputs) if inputs is not None else folder
    sys.path.insert(0, str(folder / "repository"))
    print("[candidate sandbox] Loading cached model", flush=True)
    model = load_model(inputs)
    print("[candidate sandbox] Loading task vectors and LoRA factors", flush=True)
    tensors = load_file(str(inputs / "inputs.safetensors"))
    vectors = [{k[len(f"task{i}/"):]: v for k, v in tensors.items()
                if k.startswith(f"task{i}/")} for i in range(3)]
    lora_adapters = [
        {key: {"A": tensors[f"lora{i}/A/{key}"],
               "B": tensors[f"lora{i}/B/{key}"],
               "scaling": float(tensors[f"lora{i}/scaling/{key}"].item())}
         for key in vectors[i]} for i in range(3)]
    scope = {"torch": torch}
    print("[candidate sandbox] Loading candidate code", flush=True)
    exec(compile((folder / "candidate.py").read_text(), "candidate.py", "exec"), scope)
    with torch.no_grad():
        print("[candidate sandbox] Calling merge", flush=True)
        merged = scope["merge"](model, vectors, lora_adapters)
    if not isinstance(merged, torch.nn.Module):
        raise TypeError("merge must return a torch.nn.Module")
    state = merged.state_dict()
    keys = vectors[0].keys()
    # Only permitted backbone weights leave the sandbox; never export task heads.
    print("[candidate sandbox] Exporting merged backbone weights", flush=True)
    save_file({k: state[k].detach().cpu().contiguous().clone() for k in keys},
              str(folder / "output.safetensors"))
    print("[candidate sandbox] Finished", flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    except Exception as error:
        (Path(sys.argv[1]) / "error.json").write_text(json.dumps({"error": str(error)[:2000]}))
        raise
