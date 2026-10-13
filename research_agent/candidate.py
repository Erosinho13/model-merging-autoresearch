import torch
from peft import PeftModel
from merging_types import LoRAFactors


def merge(
    base_model: PeftModel,
    task_vectors: list[dict[str, torch.Tensor]],
    lora_adapters: list[dict[str, LoRAFactors]],
) -> torch.nn.Module:
    pass
