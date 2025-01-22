from dataclasses import dataclass, field
from typing import Literal, Any, List

@dataclass
class BaseConfig:
    method: str = field(default=None)
    r: int = field(default=16)
    lora_alpha: float = field(default=8.0)
    target_modules: List[str] = field(default=None)

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'up_proj', 'gate_proj', 'down_proj'
            ]
