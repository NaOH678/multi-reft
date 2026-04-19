#!/bin/bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 基本训练参数
DATASET_NAME="${DATASET_NAME:-stereotype}"  # 可选: truthful/helpful/moral/safety/stereotype/toxicity/combined
MODEL_PATH="${MODEL_PATH:-../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# LoRA参数：当 LORA_RANK 为空时，脚本会走SFT；有值时走LoRA
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,up_proj,down_proj}"

# 与 first_train.sh 一样：输出目录名由你手动指定，不做路径提取/自动拼接。
RUN_NAME="${RUN_NAME:-llama3-8b-lora-stereotype}"
OUTPUT_ROOT="${OUTPUT_ROOT:-baseline/LoRA}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-./run_lora_train_${RUN_NAME}_${TIMESTAMP}.log}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"


LORA_ARGS=()
TRAIN_MODE="sft"
if [[ -n "${LORA_RANK}" ]]; then
  TRAIN_MODE="lora"
  LORA_ARGS+=(--lora_rank "${LORA_RANK}")
  LORA_ARGS+=(--lora_alpha "${LORA_ALPHA}")
  LORA_ARGS+=(--lora_dropout "${LORA_DROPOUT}")

  IFS=',' read -r -a TARGET_MODULE_ARRAY <<< "${LORA_TARGET_MODULES}"
  for module in "${TARGET_MODULE_ARRAY[@]}"; do
    if [[ -n "${module}" ]]; then
      LORA_ARGS+=(--target_modules "${module}")
    fi
  done
fi

echo "train_mode=${TRAIN_MODE}, dataset=${DATASET_NAME}, output_dir=${OUTPUT_DIR}"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch --num_processes="${NUM_PROCESSES}" "${SCRIPT_DIR}/train_lora.py" \
--dataset_name "${DATASET_NAME}" \
--model_name_or_path "${MODEL_PATH}" \
--output_dir "${OUTPUT_DIR}" \
--num_train_epochs "${NUM_TRAIN_EPOCHS:-6}" \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-16}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
--learning_rate "${LEARNING_RATE:-3e-4}" \
--bf16 "${BF16:-True}" \
"${LORA_ARGS[@]}" \
2>&1 | tee -a "${LOG_FILE}"

echo "训练日志已保存到: ${LOG_FILE}"
