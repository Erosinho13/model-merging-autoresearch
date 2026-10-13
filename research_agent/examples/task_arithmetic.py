import torch
from peft import PeftModel
from baseline_configs import BASELINES
from merging_types import LoRAFactors


def merge(base_model: PeftModel, task_vectors: list[dict[str, torch.Tensor]],
          lora_adapters: list[dict[str, LoRAFactors]]) -> torch.nn.Module:
    scale = BASELINES['task_arithmetic']['scaling_coeffs'][0]
    state = base_model.state_dict()
    with torch.no_grad():
        for key in task_vectors[0]:
            state[key].add_(scale * sum(vector[key] for vector in task_vectors))
    return base_model
