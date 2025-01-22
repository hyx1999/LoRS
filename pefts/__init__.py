import torch
from peft import LoraConfig, get_peft_model
from .base import BaseConfig, BaseModel
from .model_utils import (
    LorsConfig,
    SqftConfig, 
    SppConfig, 
    get_lors_model,
    get_sqft_model,
    get_spp_model,
    sqft_methods, 
    spp_methods,
    get_model_with_adapters,
    merge_and_unload,
)
from .dispatch_config import DispatchConfig

# # https://github.com/pytorch/pytorch/issues/124565
# torch.empty(1, device='cuda', requires_grad=True).backward()
