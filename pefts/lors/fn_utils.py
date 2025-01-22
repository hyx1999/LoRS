import math
import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import PartialState
from tqdm import tqdm

class LoraInitFn:

    def __init__(self):
        pass
        
    @torch.no_grad()
    def init_lora(self, 
        weight: nn.Parameter,
        grad: torch.Tensor, 
        lora_A: nn.Parameter,
        lora_B: nn.Parameter,
        stable_gamma: float = 1.0,
    ):
        device = weight.device
        grad = grad.to(device).to(torch.float32)
        lora_r = lora_A.shape[1]
        try:
            U, S, V = torch.svd_lowrank(
                grad.float(), q=min(4 * lora_r, min(grad.shape)),
                niter=4
            )
            V = V.T
        except Exception as e:
            raise ValueError("error from torch.svd_lowrank")

        A = U[:, lora_r: 2 * lora_r].zero_()
        B = (V[:lora_r, :] / stable_gamma)

        lora_A.data = A.to(weight.dtype).contiguous()
        lora_B.data = B.to(weight.dtype).contiguous()
