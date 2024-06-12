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


train(){
    rm ./output/${PROJECT_NAME} -rf
    python ./train_trace_rapt.py \
    --project ${PROJECT_NAME} \
    --data_dir "${DATA_SOURCE}" \
    --output_dir ./output \
    --model_path ./model \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --logging_steps 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --learning_rate 4e-5 \
    --code_bert ../codebert \
    --code_kg_location "${CODE_KG_LOC}" \
    --text_kg_location "${TEXT_KG_LOC}" \
    --code_kg_mode "inner" \
    --tqdm_interval ${TQDM} 
}

eval(){
    python eval_trace_rapt.py \
    --project ${PROJECT_NAME} \
    --data_dir "${DATA_SOURCE}" \
    --model_path ./output/${PROJECT_NAME} \
    --code_bert "../codebert" \
    --per_gpu_eval_batch_size 32 \
    --output_dir "./result/" \
    --code_kg_location "${CODE_KG_LOC}" \
    --text_kg_location "${TEXT_KG_LOC}" \
    --code_kg_mode "inner" \
    --tqdm_interval ${TQDM}
}
PROJECT_NAME=hornetq
TQDM=300
DATA_SOURCE="/home/yueli/HuYworks1/other/codebert_p/data"
KG_LOC="${DATA_SOURCE}/${PROJECT_NAME}/"

CODE_KG_LOC=${KG_LOC}
TEXT_KG_LOC=${KG_LOC}

train

eval

