import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union

from .sqft_linear import SqftLinear
from .sqft_linear_gc import SqftLinearGC
from .config import SqftConfig
from pefts.base.base_model import BaseModel

sqft_methods: Dict[str, Union[SqftLinear, SqftLinearGC]] = {
    "sqft-gc": SqftLinearGC,
    "sqft": SqftLinear,
}

class SqftModel(BaseModel):
    
    def __init__(self, config, base_model):
        super().__init__(config, base_model)
        self.add_adapters()
        self.frozen_params()
        
    def add_adapters(self):
        linears: List[Tuple[str, nn.Linear]] = \
            [(name, module) for name, module in self.base_model.named_modules() 
                    if isinstance(module, nn.Linear) and any(name.endswith(x) for x in self.config.target_modules)]
        LinearModule = sqft_methods[self.config.method]
        for name, module in linears:
            parent = self.base_model.get_submodule(".".join(name.split(".")[:-1]))
            name = name.split(".")[-1]
            new_module = LinearModule(
                module.in_features,
                module.out_features,
                True if module.bias is not None else False,
                module.weight.device,
                module.weight.dtype,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
            )
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(parent, name, new_module)
    
    def frozen_params(self):
        for name, param in self.base_model.named_parameters():
            if not any(x in name for x in ["lora_A", "lora_B"]):
                param.requires_grad = False
    
    def merge_and_unload(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, (SqftLinear, SqftLinearGC)):
                module.merge_adapter()
        return self.base_model
    
def get_sqft_model(model, config):
    return SqftModel(config, model)

def sqft_merge_adapters(model: SqftModel):
    return model.merge_and_unload()
