import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from typing import Tuple
from typing import Optional, Dict, Callable
from collections import namedtuple

Params = namedtuple("Params", ["scaling_factor", "training"])

class LorsFn(Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor, 
        weight: torch.Tensor,
        bias: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        params: Params,
    ) -> torch.Tensor:
        output_shape = x.shape[:-1] + (-1,)
        x_view = x.view(-1, x.shape[-1])
        merged_weight = weight.addmm(lora_A, lora_B, alpha=params.scaling_factor).mul_(weight != 0)
        y = x_view.mm(merged_weight.t()).view(output_shape)
        if bias is not None:
            y.add_(bias)
        ctx.save_for_backward(x, weight, lora_A, lora_B)
        ctx.params = params
        return y

    @staticmethod
    @once_differentiable
    def backward(
        ctx, 
        grad_y: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x, weight, lora_A, lora_B = ctx.saved_tensors
        params: Params = ctx.params
        
        x_shape = x.shape
        grad_output_shape = grad_y.shape
        x = x.view(-1, x_shape[-1])
        grad_y = grad_y.view(-1, grad_output_shape[-1])

        grad_x = grad_bias = grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            merged_weight = weight.addmm(lora_A, lora_B, alpha=params.scaling_factor).mul_(weight != 0)
            grad_x = grad_y.mm(merged_weight).view(*x_shape)
        if ctx.needs_input_grad[1]:
            raise ValueError("Not support computing the gradient of weight.")
        if ctx.needs_input_grad[2]:
            grad_bias = grad_y.sum(dim=0)
        if ctx.needs_input_grad[3]:
            grad_xBt = x @ lora_B.t()
            grad_A = grad_y.t() @ grad_xBt
        if ctx.needs_input_grad[4]:
            grad_yA = grad_y @ lora_A
            grad_B = grad_yA.t() @ x
        return grad_x, None, grad_bias, grad_A, grad_B, None
