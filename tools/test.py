import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple

class SpLoraFn(Function):
        
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor, 
        weight: torch.Tensor, 
        lora_A: torch.Tensor, 
        lora_B: torch.Tensor,
        scaling: float,
        training: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, lora_A, lora_B)
        ctx.scaling = scaling
        output_shape = x.shape[:-1] + (-1,)
        x = x.view(-1, x.shape[-1])
        mask = (weight != 0)
        weight.addmm_(lora_B, lora_A, alpha=scaling).mul_(mask)
        output = x.mm(weight.t()).view(output_shape)
        if not training:
            weight.addmm_(lora_B, lora_A, alpha=-scaling).mul_(mask)
        return output

    @staticmethod
    def backward(ctx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x, weight, lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling

        mask = (weight != 0)
        x_shape = x.shape
        grad_output_shape = grad_output.shape
        x = x.view(-1, x_shape[-1])
        grad_output = grad_output.view(-1, grad_output_shape[-1])
        _grad_weight = grad_output.t().mm(x).mul_(mask)

        grad_x = grad_weight = grad_lora_A = grad_lora_B = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(weight)

        weight.addmm_(lora_B, lora_A, alpha=-scaling).mul_(mask)

        if ctx.needs_input_grad[1]:
            grad_weight = _grad_weight       
        if ctx.needs_input_grad[2]:
            grad_lora_A = _grad_weight.t().mm(lora_B).t()
        if ctx.needs_input_grad[3]:
            grad_lora_B = _grad_weight.mm(lora_A.t())
        if grad_x is not None:
            grad_x = grad_x.view(*x_shape)
        return grad_x, grad_weight, grad_lora_A, grad_lora_B, None, None



class FuseSparseLoraLinear(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        lora_rank: int,
        device=None, 
        dtype=None
    ) -> None:
        super().__init__(in_features, out_features, False, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self) -> None:
        nn.init.normal_(self.lora_A)
        nn.init.normal_(self.lora_B)
        # nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SpLoraFn.apply(
            x, self.weight, self.lora_A, self.lora_B, 1.0, self.training
        )


class FuseSparseLoraLinear_test(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        lora_rank: int,
        device=None, 
        dtype=None
    ) -> None:
        super().__init__(in_features, out_features, False, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        self.mask = nn.Parameter(torch.randint(0, 1, (out_features, in_features)).bool(), requires_grad=False)
        with torch.no_grad():
            self.weight.data *= self.mask
        self.reset_lora_parameters()
    
    
    def reset_lora_parameters(self) -> None:
        nn.init.normal_(self.lora_A)
        nn.init.normal_(self.lora_B)
        # nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dw = self.lora_B.mm(self.lora_A)
        output = F.linear(x, self.mask * (self.weight + dw), None)
        return output

fc1 = FuseSparseLoraLinear(8, 4, 2)
mask = torch.randint(0, 2, (4, 8)).type_as(fc1.weight)
with torch.no_grad():
    fc1.weight.mul_(mask)

fc2 = FuseSparseLoraLinear_test(8, 4, 2)
fc2.load_state_dict(fc1.state_dict(), strict=False)
fc2.mask.copy_(mask)

x = torch.randn(2, 1, 8)
y1 = fc1(x).sum()
y1.backward()
print(f"y1 = {y1}")
print(f"fc1.weight.grad: {fc1.weight.grad}")
print(f"fc1.lora_A.grad: {fc1.lora_A.grad}")
print(f"fc1.lora_B.grad: {fc1.lora_B.grad}")

# y1 = fc1(x).sum()
# y1.backward()
# print(f"y1 = {y1}")
# print(f"fc1.weight.grad: {fc1.weight.grad}")
# print(f"fc1.lora_A.grad: {fc1.lora_A.grad}")
# print(f"fc1.lora_B.grad: {fc1.lora_B.grad}")
# print(f"fc1.selected_weights.grad: {fc1.selected_weights.grad}")

y2 = fc2(x).sum()
y2.backward()
print(f"y2 = {y2}")
print(f"fc2.weight.grad: {fc2.weight.grad}")
print(f"fc2.lora_A.grad: {fc2.lora_A.grad}")
print(f"fc2.lora_B.grad: {fc2.lora_B.grad}")

# y2 = fc2(x).sum()
# y2.backward()
# print(f"y2 = {y2}")
# print(f"fc2.weight.grad: {fc2.weight.grad}")
# print(f"fc2.lora_A.grad: {fc2.lora_A.grad}")
# print(f"fc2.lora_B.grad: {fc2.lora_B.grad}")
# print(f"fc1.selected_weights.grad: {fc2.selected_weights.grad}")
