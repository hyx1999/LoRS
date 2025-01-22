from pefts.base.base_config import BaseConfig
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass
class LorsConfig(BaseConfig):
    stable_gamma: float = field(default=1.0)
