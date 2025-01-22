from peft import LoraConfig, get_peft_model
from .base import BaseConfig, BaseModel
from .lors import (
    LorsConfig,
    get_lors_model,
)
from .sqft import (
    SqftConfig, 
    get_sqft_model,
    sqft_methods, 
)
from .spp import (
    SppConfig, 
    get_spp_model,
    spp_methods,
)
from .dispatch_config import DispatchConfig

def get_model_with_adapters(model, config: DispatchConfig, adapter_name: str = "default", mixed: bool = False):
    if config.method == "none":
        return model
    elif config.method == "lora":
        return get_peft_model(model, config.config, adapter_name, mixed)
    elif config.method in sqft_methods.keys():
        return get_sqft_model(model, config.config)
    elif config.method in spp_methods.keys():
        return get_spp_model(model, config.config)
    elif config.method == "lors":
        return get_lors_model(model, config.config)
    else:
        raise ValueError

def merge_and_unload(model, config: DispatchConfig):
    if config.method == "none":
        return model
    elif config.method == "lora":
        return model.base_model.merge_and_unload()
    else:
        return model.merge_and_unload()
