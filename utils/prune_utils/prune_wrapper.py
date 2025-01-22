import math
import torch
import torch.nn as nn
from typing import List

# Define WrappedGPT class
class PruneWrapper:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, modules: List[nn.Linear], modules_name: List[str], 
        parent: nn.Module = None,
        prev_op: nn.Module = None, 
        enable_prune: bool = True, 
        layer_id: int = 0,
    ):
        self.modules = modules
        self.dev = modules[0].weight.device
        self.rows = modules[0].weight.data.shape[0]
        self.columns = modules[0].weight.data.shape[1]

        self.scale_row = torch.zeros((self.columns), device=self.dev)
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        self.parent = parent
        self.prev_op = prev_op
        self.enable_prune = enable_prune
        self.layer_id = layer_id 
        self.modules_name = modules_name
    
    def configurate(self, 
        sparsity_type: str,
        sparsity: float,
        prune_method: str,
        prune_n: int, 
        prune_m: int, 
        **kwargs
    ):
        self.sparsity_type = sparsity_type
        self.sparsity = sparsity
        self.prune_method = prune_method
        self.prune_n = prune_n
        self.prune_m = prune_m

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.modules[0], nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scale_row *= self.nsamples / (self.nsamples+tmp)
        self.H *= self.nsamples / (self.nsamples + tmp)

        inp = inp.type(torch.float32)
        self.scale_row += torch.norm(inp, p=2, dim=1) ** 2  / (self.nsamples + tmp)
        
        inp = math.sqrt(2 / (self.nsamples + tmp)) * inp.float()
        self.H += inp.matmul(inp.t())

        self.nsamples += tmp
    
    def indices_shuffle(self, perm: torch.Tensor):
        return perm.reshape(self.prune_m, -1).transpose(0, 1).reshape(-1)

    @torch.no_grad()
    def prune(self):
        if not self.enable_prune:
            return
        print(f"pruning layer {self.layer_id} name {self.modules_name}")
        
        dtype = self.modules[0].weight.data.dtype
        weight = torch.cat([module.weight.data for module in self.modules], dim=0)
        
        modules_shape_dim0 = [module.weight.shape[0] for module in self.modules]

        # pruning
        weight_prune = weight
        if self.prune_method == "wanda":
            self.wanda_prune(weight_prune, self.prune_n, self.prune_m, self.sparsity)
        elif self.prune_method == "sparsegpt":
            self.sparsegpt_prune(weight_prune, self.prune_n, self.prune_m)
        else:
            raise ValueError
        
        new_weight = weight_prune
        new_weights = torch.split(new_weight, modules_shape_dim0)
        for new_weight, module in zip(new_weights, self.modules):
            module.weight.data = new_weight.to(dtype)  ## set weights to zero
    
    def wanda_prune(self, weight, prune_n, prune_m, sparsity=None):
        scale_row = self.scale_row

        weight_metric = torch.abs(weight) * torch.sqrt(scale_row.reshape((1,-1)))

        weight_mask = (torch.zeros_like(weight_metric) == 1)  ## initialize a mask to be all False
        if prune_n is not None:
            # structured n:m sparsity
            for ii in range(0, weight_metric.shape[1]):
                if ii % prune_m == 0:
                    w_abs = weight_metric[:,ii:(ii+prune_m)].float()
                    weight_mask.scatter_(1,ii+torch.topk(w_abs, prune_n, dim=1, largest=False)[1], True)
        else:
            assert sparsity is not None
            # unstructured
            sort_res = torch.sort(weight_metric, dim=-1, stable=True)
            indices = sort_res[1][:,:int(weight_metric.shape[1]*sparsity)]
            weight_mask.scatter_(1, indices, True)

        weight[weight_mask] = 0
    
    def sparsegpt_prune(
        self, weight, prune_n=0, prune_m=0, sparsity=0.5, blocksize=128, percdamp=.01
    ):
        ori_shape = weight.shape
        ori_dtype = weight.dtype
        W = weight.data.clone()
        W = W.float()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(weight.shape[0], device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                #print(prune_n, prune_m)
                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        weight.data = W.reshape(ori_shape).to(ori_dtype)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
