from pefts.base.base_config import BaseConfig
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class SppConfig(BaseConfig):
    r: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
