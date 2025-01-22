import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from .base_config import BaseConfig

class BaseModel(nn.Module):
    
    def __init__(self, 
        config: BaseConfig, 
        base_model: LlamaForCausalLM,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def merge_and_unload(self):
        pass
    
    def get_input_embeddings(self, *args, **kwargs):
        return self.base_model.get_input_embeddings(*args, **kwargs)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)
