import torch
from peft import PeftModel
from merging_types import LoRAFactors


def merge(base_model: PeftModel, task_vectors: list[dict[str, torch.Tensor]],
          lora_adapters: list[dict[str, LoRAFactors]]) -> torch.nn.Module:
    state = base_model.state_dict()
    with torch.no_grad():
        for key in task_vectors[0]:
            state[key].add_(sum(vector[key] for vector in task_vectors) / len(task_vectors))
    return base_model
