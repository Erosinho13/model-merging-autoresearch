import torch.nn as nn
from collections import defaultdict, OrderedDict
from merging_types import LoRAFactors

"""
True base_model.model.classifier.original_module.dense.weight
True base_model.model.classifier.original_module.dense.bias
True base_model.model.classifier.original_module.out_proj.weight
True base_model.model.classifier.original_module.out_proj.bias

"""


class LoRAHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_lora_factors(self) -> dict[str, LoRAFactors]:
        """Copy original adapter factors, keyed like the dense task updates."""
        factors = {}
        for name, module in self.base_model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            if "default" not in module.lora_A or "default" not in module.lora_B:
                continue
            A = module.lora_A["default"].weight.detach().cpu().clone()
            B = module.lora_B["default"].weight.detach().cpu().clone()
            if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[1]:
                raise ValueError(f"Invalid LoRA factor shapes: {name}")
            scale = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling
            factors[f"{name}.weight"] = {"A": A, "B": B, "scaling": float(scale)}
        if not factors:
            raise ValueError("No default LoRA adapter factors found")
        return OrderedDict(sorted(factors.items()))
        
    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.base_model.state_dict()
        for key, val in sd.items():
            if 'lora_A.default' in key:
                base_name = key.replace('.lora_A.default', '')
                layer2lora_parameters[base_name]['A'] = val
            elif 'lora_B.default' in key:
                base_name = key.replace('.lora_B.default', '')
                layer2lora_parameters[base_name]['B'] = val
        
        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
            task_parameters[name] = (key2val['B'] @ key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))
    
    def get_model(self):
        return self.base_model.get_base_model
    
    
class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model
