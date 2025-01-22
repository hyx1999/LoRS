import argparse
import os 
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune_utils import check_sparsity
from utils.eval_utils import eval_ppl, eval_zero_shot, eval_humaneval

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(args, device):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    model.seqlen = 2048
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
        type=str, 
        default="nlu", 
        choices=[
            "nlu", 
            "gsm8k",
            "human-eval",
        ]
    )
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument('--model', type=str, help='Path to LLaMA model')
    parser.add_argument('--model_name', type=str, help='Name of LLaMA model')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    device = torch.device("cuda:0")
    print("use device ", device)

    print(f"loading llm model {args.model}")
    model = get_llm(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    
    sparsity = check_sparsity(model)
    print(f"sparsity = {sparsity}")
    
    # # Get the test loader
    if args.eval_ppl:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity: {ppl_test}")
    
    if args.task == "nlu":   
        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model_name, model, tokenizer, task_list, num_shot, False)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
    elif args.task == "gsm8k":
        task_list = ["gsm8k"]
        num_shot = 0
        results = eval_zero_shot(args.model_name, model, tokenizer, task_list, num_shot, False)
        print("********************************")
        print("gsm8k results")
        print(results)
    elif args.task == "human-eval":
        eval_humaneval(args, model, tokenizer)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
