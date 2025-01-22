#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0,1
default_method=lors
default_model=/fs/fast/u2021000902/hyx/Llama-2-7b-hf-wanda2of4
default_output=/fs/fast/u2021000902/hyx/lors/results/Llama-2-7b-hf-wanda2of4-slimpajama
default_dataset=datasets/slimpajama-0.5B-Llama-2-tokenized
# /fs/fast/u2021000902/hyx/Llama-3-8b-hf-wand2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-sparsegpt2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-wand2of4
# datasets/slimpajama-0.5B-Llama-2-tokenized

devices=${1:-$default_devices}
method=${3:-$default_method}
model=${4:-$default_model}
output=${5:-$default_output}

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g16.yaml \
    --main_process_port 25660 \
    peft_pretrain.py \
    --peft_method ${method} \
    --dataset_name ${default_dataset} \
    --model_name_or_path ${model} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --min_learning_rate 1e-6 \
    --block_size 2048 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output}
