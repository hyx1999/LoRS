import torch
import torch.nn as nn
from typing import List, Tuple
from transformers import LlamaForCausalLM
from pefts.base.base_model import BaseModel
from .lors_linear import LorsLinear
from .config import LorsConfig
from .fn_utils import LoraInitFn
from tqdm import tqdm
from typing import Dict

class LorsModel(BaseModel):
    
    def __init__(self, config, base_model):
        super().__init__(config, base_model)
        self.lora_init_fn = LoraInitFn()
        self.add_adapters(base_model.named_grad)
        self.frozen_params()
    
    def add_adapters(self, named_grad: Dict[str, torch.Tensor]):
        linears: List[Tuple[str, nn.Linear]] = \
            [(name, module) for name, module in self.base_model.named_modules() 
                    if isinstance(module, nn.Linear) and \
                        any(name.endswith(x) for x in self.config.target_modules)]
        for name, module in tqdm(linears, desc="init adapters..."):
            parent = self.base_model.get_submodule(".".join(name.split(".")[:-1]))
            module_name = name.split(".")[-1]
            new_module = LorsLinear(
                module.in_features,
                module.out_features,
                True if module.bias is not None else False,
                module.weight.device,
                module.weight.dtype,
                r=self.config.r,
                lora_alpha=self.config.lora_alpha,
            )
            new_module.load_state_dict(module.state_dict(), strict=False)
            setattr(parent, module_name, new_module)
            self.lora_init_fn.init_lora(
                new_module.weight,
                named_grad[name],
                new_module.lora_A,
                new_module.lora_B,
                self.config.stable_gamma,
            )
        self.lors_linears = [m for m in self.base_model.modules() if isinstance(m, LorsLinear)]
    
    def frozen_params(self):
        for name, param in self.base_model.named_parameters():
            if not any(x in name for x in ["lora_A", "lora_B"]):
                param.requires_grad = False
    
    def merge_and_unload(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, LorsLinear):
                module.merge_adapter()
        return self.base_model

def get_lors_model(
    model: LlamaForCausalLM,
    config: LorsConfig,
):
    return LorsModel(config, model)

def lors_merge_adapters(model: LorsModel):
    return model.merge_and_unload()
