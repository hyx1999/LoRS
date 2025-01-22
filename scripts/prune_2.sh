#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

devices=4

CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
    --model /fs/fast/u2021000902/hyx/Llama-2-13b-hf \
    --sparsity_type 2:4 \
    --prune_method sparsegpt \
    --save_result misc/log/sparsegpt \
    --save_model /fs/fast/u2021000902/hyx/Llama-2-13b-hf-sparsegpt2of4

CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
    --model /fs/fast/u2021000902/hyx/Llama-2-13b-hf \
    --sparsity_type 2:4 \
    --prune_method wanda \
    --save_result misc/log/wanda \
    --save_model /fs/fast/u2021000902/hyx/Llama-2-13b-hf-wand2of4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-3-8b-hf \
#     --sparsity_type 2:4 \
#     --prune_method wanda \
#     --save_result misc/log/wanda \
#     --save_model checkpoints/Llama-3-8b-hf-wand2of4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-3-8b-hf \
#     --sparsity_type 2:4 \
#     --prune_method sparsegpt \
#     --save_result misc/log/sparsegpt \
#     --save_model checkpoints/Llama-3-8b-hf-sparsegpt2of4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-7b-hf \
#     --sparsity_type 2:4 \
#     --prune_method wanda \
#     --save_result misc/log/wanda \
#     --save_model checkpoints/Llama-2-7b-hf-wand2of4

# CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
#     --model checkpoints/Llama-2-7b-hf \
#     --sparsity_type 2:4 \
#     --prune_method sparsegpt \
#     --save_result misc/log/sparsegpt \
#     --save_model checkpoints/Llama-2-7b-hf-sparsegpt2of4

