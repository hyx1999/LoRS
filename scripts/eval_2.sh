#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0,1

devices=${1:-$default_devices}

CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task nlu \
    --model /fs/fast/u2021000902/hyx/lors/results/Llama-3-8b-hf-wanda2of4 \
    --model_name Llama-3-8b-hf-wanda2of4


CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task nlu \
    --model /fs/fast/u2021000902/hyx/lors/results/Llama-2-7b-hf-sparsegpt2of4 \
    --model_name Llama-2-7b-hf-sparsegpt2of4


CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task nlu \
    --model /fs/fast/u2021000902/hyx/lors/results/Llama-2-7b-hf-wanda2of4 \
    --model_name Llama-2-7b-hf-wanda2of4
