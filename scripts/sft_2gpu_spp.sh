#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0,1
default_method=spp-gc
default_model=/data2/chenxiaodong/model/Llama-3-8b-hf-sparsegptuns
default_output=/home/huyuxuan/projects/lors/results/Llama-3-8b-hf-sparsegptuns-spp-gc
# /fs/fast/u2021000902/hyx/Llama-3-8b-hf-wand2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-sparsegpt2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-wand2of4

devices=${1:-$default_devices}
method=${2:-$default_method}
model=${3:-$default_model}
output=${4:-$default_output}

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g2.yaml \
    peft_sft.py \
    --peft_method ${method} \
    --dataset_name alpaca \
    --model_name_or_path ${model} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-5 \
    --cutoff_len 512 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output} \
    --evaluate
