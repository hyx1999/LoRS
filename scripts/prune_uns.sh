#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0
default_model=/data/models/Llama-3-8b-hf
default_output=/data2/chenxiaodong/model/Llama-3-8b-hf-sparsegptuns

devices=${1:-$default_devices}
model=${4:-$default_model}
output=${5:-$default_output}

CUDA_VISIBLE_DEVICES=${devices} python prune_llm.py \
    --model ${model} \
    --sparsity_type unstructured \
    --prune_method sparsegpt \
    --save_result misc/log/sparsegpt \
    --save_model ${output}
