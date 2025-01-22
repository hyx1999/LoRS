#!/bin/bash
set -e
set -x

cd $(dirname $0)/..

# datasets: 

export HF_DATASETS_OFFLINE=1

devices=0

for cutoff_len in 512 1024
do
    for lora_rank in 16
    do
        CUDA_VISIBLE_DEVICES=${devices} python peft_sft.py \
            --peft_method sqft \
            --dataset_name alpaca \
            --model_name_or_path checkpoints/Llama-3-8b-hf-wand2of4 \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --lora_rank ${lora_rank} \
            --learning_rate 2e-5 \
            --min_learning_rate 1e-6 \
            --cutoff_len ${cutoff_len} \
            --num_train_epochs 1 \
            --gradient_accumulation_steps 1 \
            --output_dir misc/results/sqft-l${cutoff_len}-r${lora_rank}-Llama-3-8b-hf-wand2of4 \
            --profile \
            --max_train_steps 100
    done
done

# for cutoff_len in 2048
# do
#     for lora_rank in 16 32 64
#     do
#         CUDA_VISIBLE_DEVICES=${devices} python peft_sft.py \
#             --peft_method sqft \
#             --dataset_name alpaca \
#             --model_name_or_path checkpoints/Llama-3-8b-hf-wand2of4 \
#             --per_device_train_batch_size 1 \
#             --per_device_eval_batch_size 1 \
#             --lora_rank ${lora_rank} \
#             --learning_rate 2e-5 \
#             --min_learning_rate 1e-6 \
#             --cutoff_len ${cutoff_len} \
#             --num_train_epochs 1 \
#             --gradient_accumulation_steps 1 \
#             --output_dir misc/results/sqft-r${lora_rank}-l${cutoff_len}-Llama-3-8b-hf-wand2of4 \
#             --profile \
#             --max_train_steps 100
#     done
# done

