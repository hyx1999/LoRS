#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0,1
default_method=lors
default_model=/fs/fast/u2021000902/hyx/Llama-3-8b-hf-wanda2of4
default_output=/fs/fast/u2021000902/hyx/lors/results/Llama-3-8b-hf-wanda2of4

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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --min_learning_rate 1e-6 \
    --cutoff_len 512 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output} \
    --evaluate

model=/fs/fast/u2021000902/hyx/Llama-2-7b-hf-sparsegpt2of4
output=/fs/fast/u2021000902/hyx/lors/results/Llama-2-7b-hf-sparsegpt2of4

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g2.yaml \
    peft_sft.py \
    --peft_method ${method} \
    --dataset_name alpaca \
    --model_name_or_path ${model} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --min_learning_rate 1e-6 \
    --cutoff_len 512 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output} \
    --evaluate

model=/fs/fast/u2021000902/hyx/Llama-2-7b-hf-wanda2of4
output=/fs/fast/u2021000902/hyx/lors/results/Llama-2-7b-hf-wanda2of4

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g2.yaml \
    peft_sft.py \
    --peft_method ${method} \
    --dataset_name alpaca \
    --model_name_or_path ${model} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --min_learning_rate 1e-6 \
    --cutoff_len 512 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output} \
    --evaluate