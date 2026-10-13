"""Public types for dense task updates and original task LoRA factors."""
from typing import TypedDict

from torch import Tensor


class LoRAFactors(TypedDict):
    A: Tensor
    B: Tensor
    scaling: float
