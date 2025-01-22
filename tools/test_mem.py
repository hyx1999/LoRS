import time
import torch
from transformers import AutoModelForCausalLM, LlamaForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/home/huyuxuan/projects/sp-lora/checkpoints/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
)
model.to("cuda:0")

print("load finish...")
while True:
    time.sleep(1)
