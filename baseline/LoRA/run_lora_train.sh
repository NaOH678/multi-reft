#!/bin/bash
set -euo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# 基本训练参数
DATASET_NAME="${DATASET_NAME:-combined}"  # 可选: truthful/helpful/moral/safety/stereotype/toxicity/combined
MODEL_PATH="${MODEL_PATH:-models/gemma2-9b/snapshots/7725a3ee78e8d86c51efeb1f801a195dfa91c715}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# LoRA参数：当 LORA_RANK 为空时，脚本会走SFT；有值时走LoRA
LORA_RANK="${LORA_RANK:-}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj,k_proj,up_proj,down_proj}"

# 与 first_train.sh 一样：输出目录名由你手动指定，不做路径提取/自动拼接。
RUN_NAME="${RUN_NAME:-gemma2-9b-5}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}_${TIMESTAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/inspurfs/eailab_os/scj_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-./run_lora_train_${RUN_NAME}_${TIMESTAMP}.log}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-shichaojian}/triton-cache/${HOSTNAME:-localhost}/${SLURM_JOB_ID:-manual}/${RUN_NAME}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-auto}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "${TRITON_CACHE_DIR}"


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

if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  LORA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  LORA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

echo "train_mode=${TRAIN_MODE}, dataset=${DATASET_NAME}, output_dir=${OUTPUT_DIR}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "WANDB_RUN_NAME=${WANDB_RUN_NAME}"


CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" WANDB_RUN_NAME="${WANDB_RUN_NAME}" accelerate launch --num_processes="${NUM_PROCESSES}" "${SCRIPT_DIR}/train_lora.py" \
--dataset_name "${DATASET_NAME}" \
--model_name_or_path "${MODEL_PATH}" \
--output_dir "${OUTPUT_DIR}" \
--num_train_epochs "${NUM_TRAIN_EPOCHS:-10}" \
--per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
--gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
--learning_rate "${LEARNING_RATE:-2e-6}" \
--bf16 "${BF16:-True}" \
"${LORA_ARGS[@]}" \
2>&1 | tee -a "${LOG_FILE}"

echo "训练日志已保存到: ${LOG_FILE}"
