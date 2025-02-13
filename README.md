# LoRS: Efficient Low-Rank Adaptation for Sparse Large Language Model

## Introduction

LoRS is a parameter-efficient fine-tuning method for sparse large language models. Based on LoRA, LoRS ensures the sparsity of the model by introducing a mask in the computation process. At the same time, LoRS uses weight recomputation and computational graph rearrangement to reduce the extra computational and memory overhead caused by mask. Finally, it also employs gradient-based adapter initialization to improve performance of fine-tuned model.

## Example

```python
# ...
from pefts import DispatchConfig, get_model_with_adapters, merge_and_unload
from pefts.lors.lors_utils import estimate_gradient, LoraGAContext

peft_config = DispatchConfig(
        method=args.peft_method,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
collate_fn=transformers.DataCollatorForSeq2Seq(
    tokenizer, 
    pad_to_multiple_of=8, 
    return_tensors="pt", 
    padding=True,
)
tmpset = train_dataset.select(range(args.lors_samples))
dataloader = DataLoader(tmpset, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)
named_grad = estimate_gradient(
    model=model,
    dataloader=dataloader,
    accelerator=accelerator,
    quant_flag=False,
)
with LoraGAContext(model=model, named_grad=named_grad):
    model = get_model_with_adapters(model, peft_config)
# ...
```
