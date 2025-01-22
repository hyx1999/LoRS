import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from types import MethodType
from pefts.base.linear import Linear

from .lors_autograd import LorsFn, Params

class LorsLinear(Linear):
    
    def __init__(self,
        # linear
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        # lors
        lora_alpha: float = None,
        r: int = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_A = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.scaling_factor = lora_alpha / math.sqrt(r)
        self._reset_lora_parameters()

    def _reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = LorsFn.apply(
            x,
            self.weight, 
            self.bias,
            self.lora_A,
            self.lora_B,
            Params(
                scaling_factor=self.scaling_factor, 
                training=self.training
            )
        )
        return y

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            weight = self.weight.data
            dtype = weight.dtype
            weight = weight.float()
            lora_A = self.lora_A.float()
            lora_B = self.lora_B.float()
            mask = (weight != 0)
            weight += (lora_A @ lora_B) * mask * self.scaling_factor
            self.weight.data = weight.to(dtype)
            delattr(self, "lora_A")
            delattr(self, "lora_B")
            def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight, self.bias)
            self.forward = MethodType(forward_patch, self)
