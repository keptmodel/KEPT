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
python rapt_train.py \
    --data_dir /nvme2n1/HuYworks/pretrain/data/java  \
    --code_bert /nvme2n1/HuYworks/pretrain/unixcoder \
       --output_dir ./output \
           --per_gpu_train_batch_size 8   \
             --per_gpu_eval_batch_size 8   \
               --logging_steps 10   \
                 --save_steps 10000  \
                    --gradient_accumulation_steps 16  \
                       --num_train_epochs 1    \
                        --learning_rate 4e-5  \
                           --valid_num 200   \
                             --valid_step 10000  \
                                --neg_sampling random