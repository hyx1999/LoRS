#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=3

devices=${1:-$default_devices}

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task nlu \
    --model /fs/fast/u2021000902/hyx/lors/results/Llama-3-8b-hf-sparsegptuns-spp-gc \
    --model_name Llama-3-8b-hf-sparsegpt2of4-spp-gc
