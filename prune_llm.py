import argparse
import os 
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune_utils import prune_model
from utils.eval_utils import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(args, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.seqlen = 2048 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--sparsity_type", type=str, default="2:4", choices=["2:4", "unstructured"])
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--prune_method", type=str, choices=["wanda", "sparsegpt"])
    parser.add_argument('--save_result', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda:0")
    print("use device ", device)

    # Handling n:m sparsity
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    else:
        prune_n = prune_m = 0

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    prune_model(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity: {ppl_test}")
    
    if args.save_result is not None:
        if not os.path.exists(args.save_result):
            os.makedirs(args.save_result)
        format_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_filepath = os.path.join(args.save_result, f"log_{model_name}_{format_time}.txt")
        with open(save_filepath, "w") as f:
            print(f"ppl_test: {ppl_test:.4f}\n\n", file=f, flush=True)
            print(f"args:", file=f, flush=True)
            for k, v in vars(args).items():
                print(f"\t{k}={v}", file=f, flush=True)
    
    if args.save_model is not None:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == '__main__':
    main()
