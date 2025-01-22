#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

default_devices=0,1
default_method=lors
default_model=/fs/fast/u2021000902/hyx/Llama-2-13b-hf-sparsegpt2of4
default_output=/fs/fast/u2021000902/hyx/lors/results/Llama-2-13b-hf-sparsegpt2of4
# /fs/fast/u2021000902/hyx/Llama-3-8b-hf-wand2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-sparsegpt2of4
# /fs/fast/u2021000902/hyx/Llama-2-7b-hf-wand2of4

devices=${1:-$default_devices}
method=${2:-$default_method}
model=${3:-$default_model}
output=${4:-$default_output}

CUDA_VISIBLE_DEVICES=${devices} \
    accelerate launch --config_file config/deepspeed/2gpu_g4.yaml \
    --main_process_port 29777 \
    peft_sft.py \
    --peft_method ${method} \
    --dataset_name alpaca \
    --model_name_or_path ${model} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-5 \
    --min_learning_rate 0 \
    --cutoff_len 512 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --report_to tensorboard \
    --with_tracking \
    --output_dir ${output} \
    --evaluate

CUDA_VISIBLE_DEVICES=${devices} python eval.py \
    --task nlu \
    --model /fs/fast/u2021000902/hyx/lors/results/Llama-2-13b-hf-sparsegpt2of4 \
    --model_name Llama-2-13b-hf-sparsegpt2of4
