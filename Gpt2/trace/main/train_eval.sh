#!/bin/bash

readarray -t memory_used < <(nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader)
readarray -t memory_total < <(nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader)

min_memory_usage=101
min_gpu_id=0

for i in "${!memory_used[@]}"; do
  usage=$((100 * ${memory_used[$i]} / ${memory_total[$i]}))
  if (( usage < min_memory_usage )); then
    min_memory_usage=$usage
    min_gpu_id=$i
  fi
done


export CUDA_VISIBLE_DEVICES=$min_gpu_id

CURRENT_TIME=$(date +"%Y%m%d_%H_%M_%S")


OUTPUT_FILE="./logs/output_$CURRENT_TIME.log"

train(){
    rm ./output/${PROJECT_NAME} -rf
    python ./train_trace_gpt2.py \
    --project ${PROJECT_NAME} \
    --data_dir "${DATA_SOURCE}" \
    --output_dir ./output \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --logging_steps 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 6 \
    --learning_rate 4e-5 
}

eval(){
    python eval_trace_gpt2.py \
    --project ${PROJECT_NAME} \
    --data_dir "${DATA_SOURCE}" \
    --model_path ./output/${PROJECT_NAME} \
    --per_gpu_eval_batch_size 32 \
    --output_dir "./result/" 
}

PROJECT_NAME=hornetq
OUTPUT_FILE="./logs/${PROJECT_NAME}_output_${CURRENT_TIME}.log"
TQDM=300
DATA_SOURCE="/home/yueli/HuYworks1/kept/input_data/"

train
eval

