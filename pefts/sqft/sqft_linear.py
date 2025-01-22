import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from typing import Tuple
from abc import ABC
from types import MethodType
from typing import Optional, Dict, Callable
from torch.utils.checkpoint import checkpoint
from pefts.base.linear import Linear


class SqftLinear(Linear):
    
    def __init__(self, 
        # linear
        in_features: int, 
        out_features: int, 
        bias: bool = False, 
        device=None, 
        dtype=None,
        # lora
        r: int = 16,
        lora_alpha: float = 16,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        self.scaling = lora_alpha / r
        self._reset_lora_parameters()
    
    def fuse_weight(self):
        mask = (self.weight != 0)
        weight = self.weight + mask * (self.lora_B @ self.lora_A)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fuse_weight()
        return F.linear(x, weight, self.bias)
    
    def _reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @torch.no_grad()
    def set_mask(self):
        pass

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A"):
            mask = (self.weight != 0)
            self.weight.data += (self.lora_B @ self.lora_A) * mask
            delattr(self, "lora_A")
            delattr(self, "lora_B")
            def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight, self.bias)
            self.forward = MethodType(forward_patch, self)
