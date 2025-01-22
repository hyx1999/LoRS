import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_infos
from itertools import chain
from typing import Optional

def load_SlimPajama_6B(start_idx: int = 0, num_samples: Optional[int] = None):
    data_files = {
        "train": "data/train-*",
        "test": "data/test-*",
        "validation": "data/validation-*",
    }
    path = "/data/SlimPajama-6B"
    dataset = load_dataset(path, data_files=data_files)
    if num_samples is not None:
        dataset["train"] = dataset["train"].select(range(start_idx, start_idx + num_samples))
    return dataset

def load_FineWeb_edu_10BT(start_idx: int = 0, num_samples: Optional[int] = None):
    data_files = {
        "train": "*.parquet",
    }
    path = "/data2/fineweb-edu-10BT/sample/10BT/"
    dataset = load_dataset(path, data_files=data_files)
    if num_samples is not None:
        dataset["train"] = dataset["train"].select(range(start_idx, start_idx + num_samples))
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="fineweb-edu")
parser.add_argument('--block_size', type=int, default=2048)
parser.add_argument('--model_name_or_path', type=str, default="checkpoints/Llama-2-7b-hf")
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

block_size = args.block_size
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

if args.dataset_name == "fineweb-edu":
    raw_datasets = load_FineWeb_edu_10BT(args.start_idx, args.num_samples)
else:
    raw_datasets = load_SlimPajama_6B(args.start_idx, args.num_samples)

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/process#map

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

lm_datasets.save_to_disk(args.output_path)
