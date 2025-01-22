import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
import math
from peft.utils import transpose
from pefts.base.linear import Linear


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class DiagonalGradckptLinear(nn.Module):
    def __init__(self, in_feature, out_feature, r=16, dtype=None, device=None):
        super(DiagonalGradckptLinear, self).__init__()
        self.r = r
        self.weight = nn.Parameter(torch.zeros(self.r, in_feature, dtype=dtype, device=device))
        self.weight_col = nn.Parameter(torch.zeros(out_feature, 1, dtype=dtype, device=device))

    def _checkpointed_forward(self, x, weight, weight_col, matrix):
        mat1 = torch.repeat_interleave(weight, matrix.shape[0]//self.r, dim=0) # out x in

        return F.linear(x, (mat1 * weight_col) * matrix)

    def forward(self, matrix, x):

        result = checkpoint.checkpoint(self._checkpointed_forward, x, self.weight, self.weight_col, matrix)
        return result

class SppLinearGC(Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs,
    ):
        Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        assert self.bias is None

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = DiagonalGradckptLinear(in_features, out_features, r=16, dtype=kwargs["dtype"], device=kwargs["device"])
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        # nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A.weight_col)

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A"):
            mat1 = torch.repeat_interleave(self.lora_A.weight, self.weight.shape[0] // self.r, dim=0) # out x in
            self.weight.add_(mat1 * self.lora_A.weight_col * self.weight)
            delattr(self, "lora_A")
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            raise NotImplementedError
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0: # lora_A: in_fit. lora_B: out_fit
                result += self.lora_A(self.weight, self.lora_dropout(x)) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
