#!/bin/bash

# 设置日志文件名（包含时间戳和数据集名称）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="lora_train_${TIMESTAMP}.log"

# 基本训练参数
DATASET_NAME="helpful"  # 可替换为: truthful/helpful/moral/safety/stereotype/toxic/combined
MODEL_PATH="../../../Llama-2-7b-hf"
OUTPUT_DIR="./lora_output_${DATASET_NAME}"

# LoRA特定参数
LORA_RANK=16
LORA_ALPHA=64
LORA_DROPOUT=0.05

# 使用accelerate启动训练并记录日志
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes=2 train_lora.py \
--dataset_name $DATASET_NAME \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--lora_rank $LORA_RANK \
--lora_alpha $LORA_ALPHA \
--lora_dropout $LORA_DROPOUT \
--target_modules "q_proj" "v_proj" \
--num_train_epochs 3 \
--max_samples 100 \
--warmup_steps 100 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-4 \
--bf16 True \
2>&1 | tee -a $LOG_FILE

echo "训练日志已保存到: $LOG_FILE"
