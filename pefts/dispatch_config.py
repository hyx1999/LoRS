from dataclasses import dataclass, field
from peft import LoraConfig
from typing import Literal, Any
from peft import LoraConfig
from .lors import LorsConfig
from .sqft import SqftConfig
from .spp import SppConfig

config_dict = {
    "lora": LoraConfig,
    "lors": LorsConfig,
    "sqft": SqftConfig,
    "sqft-gc": SqftConfig,
    "spp": SppConfig,
    "spp-gc": SppConfig,
}

@dataclass
class DispatchConfig:
    
    method: Literal[
        "lora",
        "lors",
        "sqft", "sqft-gc",
        "spp", "spp-gc"
        "none"
    ] = field(default="lora")

    config: Any = field(default=None)

    def __init__(self, method: str, **kwargs):
        self.method = method
        self.config = config_dict[method]()
        self.config.method = method
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
