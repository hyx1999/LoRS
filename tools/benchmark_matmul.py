import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured
from abc import ABC
from torch.sparse import SparseSemiStructuredTensor
import time

class SparseLinearMixin(ABC):
        
    @torch.no_grad()
    def to_sparse(self, random: bool = False):
        if self.bias is None:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features).type_as(self.weight),
                requires_grad=self.weight.requires_grad,
            )
        if random:
            n, m = self.weight.shape
            mask = torch.Tensor([0, 0, 1, 1]).tile((n, m // 4)).to(self.weight.device).bool()
            self.weight = nn.Parameter(
                to_sparse_semi_structured(self.weight.masked_fill(~mask, 0)),
                requires_grad=self.weight.requires_grad
            )
        else:
            self.weight = nn.Parameter(
                to_sparse_semi_structured(self.weight),
                requires_grad=self.weight.requires_grad
            )

    @torch.no_grad()
    def to_dense(self):
        self.weight = nn.Parameter(self.weight.data.to_dense(), requires_grad=self.weight.requires_grad)


class SparseLinear(nn.Linear, SparseLinearMixin):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)


print(torch.version.cuda)

SparseSemiStructuredTensor._FORCE_CUTLASS = True

hidden_size = 4096
intermediate_size = 11008
rank = 256
bs = 1
seqlen = 2048
dev = "cuda:3"
print(f"dev = {dev}")
with torch.inference_mode():
    # qkv
    output_size = hidden_size
    q = nn.Linear(hidden_size, output_size, dtype=torch.float16, device=dev)
    k = nn.Linear(hidden_size, output_size, dtype=torch.float16, device=dev)
    v = nn.Linear(hidden_size, output_size, dtype=torch.float16, device=dev)
    
    q.eval()
    k.eval()
    v.eval()
    
    x = torch.randn((bs * seqlen, hidden_size)).to(dev).to(torch.float16)
    st_time = time.perf_counter()
    for _ in range(100):
        qx = q(x)
        kx = k(x)
        vx = v(x)
        torch.cuda.synchronize(dev)
    ed_time = time.perf_counter()
    dense_time_cost = ed_time - st_time

    q = SparseLinear(hidden_size, output_size, dtype=torch.float16, device=dev)
    k = SparseLinear(hidden_size, output_size, dtype=torch.float16, device=dev)
    v = SparseLinear(hidden_size, output_size, dtype=torch.float16, device=dev)
    lo_proj = nn.Linear(hidden_size, rank, dtype=torch.float16, device=dev)
    lo_q = nn.Linear(rank, output_size, dtype=torch.float16, device=dev)
    lo_k = nn.Linear(rank, output_size, dtype=torch.float16, device=dev)
    lo_v = nn.Linear(rank, output_size, dtype=torch.float16, device=dev)

    q.eval()
    k.eval()
    v.eval()
    lo_proj.eval()
    lo_q.eval()
    lo_k.eval()
    lo_v.eval()
    
    q.to_sparse(random=True)
    k.to_sparse(random=True)
    v.to_sparse(random=True)
    torch.cuda.synchronize(dev)
    
    x = torch.randn((bs * seqlen, hidden_size)).to(dev).to(torch.float16)
    st_time = time.perf_counter()
    for _ in range(100):
        lo_x = lo_proj(x)
        qx = q(x) + lo_q(lo_x)
        kx = k(x) + lo_k(lo_x)
        vx = v(x) + lo_v(lo_x)
        torch.cuda.synchronize(dev)
    ed_time = time.perf_counter()
    sparse_time_cost = ed_time - st_time
    print(f"name = qkv")
    print(f"dense mm time cost: {dense_time_cost}")
    print(f"sparse mm time cost: {sparse_time_cost}")
    print(f"speed up: {dense_time_cost / sparse_time_cost}")


with torch.inference_mode():
    # gu
    output_size = hidden_size
    u = nn.Linear(hidden_size, intermediate_size, dtype=torch.float16, device=dev)
    g = nn.Linear(hidden_size, intermediate_size, dtype=torch.float16, device=dev)
    
    u.eval()
    g.eval()
    
    x = torch.randn((bs * seqlen, hidden_size)).to(dev).to(torch.float16)
    st_time = time.perf_counter()
    for _ in range(100):
        ux = u(x)
        gx = g(x)
        torch.cuda.synchronize(dev)
    ed_time = time.perf_counter()
    dense_time_cost = ed_time - st_time

    u = SparseLinear(hidden_size, intermediate_size, dtype=torch.float16, device=dev)
    g = SparseLinear(hidden_size, intermediate_size, dtype=torch.float16, device=dev)
    lo_proj = nn.Linear(hidden_size, rank, dtype=torch.float16, device=dev)
    lo_u = nn.Linear(rank, intermediate_size, dtype=torch.float16, device=dev)
    lo_g = nn.Linear(rank, intermediate_size, dtype=torch.float16, device=dev)

    u.eval()
    g.eval()
    lo_proj.eval()
    lo_u.eval()
    lo_g.eval()

    u.to_sparse(random=True)
    g.to_sparse(random=True)
    torch.cuda.synchronize(dev)
    
    x = torch.randn((bs * seqlen, hidden_size)).to(dev).to(torch.float16)
    st_time = time.perf_counter()
    for _ in range(100):
        lo_x = lo_proj(x)
        ux = u(x) + lo_u(lo_x)
        gx = g(x) + lo_g(lo_x)
        torch.cuda.synchronize(dev)
    ed_time = time.perf_counter()
    sparse_time_cost = ed_time - st_time
    print(f"name = ug")
    print(f"dense mm time cost: {dense_time_cost}")
    print(f"sparse mm time cost: {sparse_time_cost}")
    print(f"speed up: {dense_time_cost / sparse_time_cost}")
