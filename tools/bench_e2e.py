from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from utils.data_utils.calib_data import load_calib_data
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import argparse
import gc
from torch.sparse.semi_structured import SparseSemiStructuredTensor, to_sparse_semi_structured
from peft import LoraConfig, get_peft_model, TaskType

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--cutlass', action="store_true")
    return parser.parse_args()


@torch.no_grad()
def to_sparse_linear(model: nn.Module):
    for module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(to_sparse_semi_structured(module.weight))


@torch.inference_mode()
def bench_generate(args, all_input_ids, model_type, dev):
    model = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map=dev
    )
    model.eval()

    if model_type == "sparse":
        print("dense model => sparse model...")
        to_sparse_linear(model)
    
    torch.cuda.empty_cache()
    gc.collect()

    times = []
    
    model: LlamaForCausalLM = model
    
    gen_config = GenerationConfig(max_new_tokens=8)
    
    # one epoch for warmup
    for j in range(2):
        for i in tqdm(range(0, all_input_ids.shape[0], args.batch_size), desc=f"epoch = {j}"):
            if i + args.batch_size > all_input_ids.shape[0]:
                break
            input_ids = all_input_ids[i:i+args.batch_size].to(dev)
            st_time = time.perf_counter()
            _ = model.generate(input_ids, generation_config=gen_config)
            torch.cuda.synchronize(dev)
            ed_time = time.perf_counter()
            if j > 0:
                times.append(ed_time - st_time)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    time_cost = sum(times) / len(times)
    return time_cost



@torch.inference_mode()
def bench_prefill(args, all_input_ids, model_type, dev):
    model = model_type.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    model.to(dev)
    model.eval()
    
    if model_type == "sparse":
        print("dense model => sparse model...")
        to_sparse_linear(model)
    
    torch.cuda.empty_cache()
    gc.collect()
        
    times = []
    
    # one epoch for warmup
    for i in tqdm(range(0, all_input_ids.shape[0], args.batch_size)):
        if i + args.batch_size > all_input_ids.shape[0]:
            break
        input_ids = all_input_ids[i:i+args.batch_size].to(dev)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        st_time = time.perf_counter()
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize(dev)
        ed_time = time.perf_counter()
        times.append(ed_time - st_time)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    time_cost = sum(times) / len(times)
    return time_cost

def main():
    args = parse_args()
    if args.cutlass:
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

    _, testenc = load_calib_data("wikitext2", seed=0, seqlen=args.length, tokenizer=tokenizer)

    # Convert the whole text of evaluation dataset into batches of sequences.
    all_input_ids = testenc.input_ids  # (1, text_len)
    nsamples = all_input_ids.numel() // args.length  # The tail is truncated.
    all_input_ids = all_input_ids[:, :nsamples * args.length].view(nsamples, args.length)  # (nsamples, seqlen)
    all_input_ids = all_input_ids[:16]

    sparse_time = bench_prefill(args, all_input_ids, "sparse", "cuda")
    base_time = bench_prefill(args, all_input_ids, "dense", "cuda")
    # sparse_time = bench_generate(args, all_input_ids, LossLlamaForCausalLM, args.sparse_model, "cuda")    
    # base_time = bench_generate(args, all_input_ids, LlamaForCausalLM, args.base_model, "cuda")
    print("=" * 10 + ">")
    print(f"batch_size: {args.batch_size}, length: {args.length}")
    print(f"Base model prefill time cost: {base_time}")
    print(f"Sparse model prefill time cost: {sparse_time}")
    print(f"Speed Up: {base_time / sparse_time}")


if __name__ == '__main__':
    main()
