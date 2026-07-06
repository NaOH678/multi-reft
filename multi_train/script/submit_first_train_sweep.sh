#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITIONS_RAW="${PARTITIONS:-${PARTITION:-ftmanip}}"
PARTITIONS_NORM="${PARTITIONS_RAW//,/ }"
read -r -a PARTITION_ARR <<< "${PARTITIONS_NORM}"
if [[ "${#PARTITION_ARR[@]}" -eq 0 ]]; then
  echo "No partition configured. Set PARTITION or PARTITIONS." >&2
  exit 1
fi

GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

subspace="${subspace:-${SUBSPACE:-safety}}"
model_name_or_path="${model_name_or_path:-${MODEL_NAME_OR_PATH:-models/qwen25-7b/snapshots/d149729398750b98c0af14eb82c78cfe92750796}}"
WANDB_MODE_VALUE="${WANDB_MODE:-offline}"
num_processes="${num_processes:-${NUM_PROCESSES:-8}}"
per_device_train_batch_size="${per_device_train_batch_size:-${PER_DEVICE_TRAIN_BATCH_SIZE:-}}"
gradient_accumulation_steps="${gradient_accumulation_steps:-${GRADIENT_ACCUMULATION_STEPS:-}}"
subspace_rank="${subspace_rank:-${SUBSPACE_RANK:-8}}"
warmup_ratio="${warmup_ratio:-${WARMUP_RATIO:-0.0125}}"
weight_decay="${weight_decay:-${WEIGHT_DECAY:-0}}"
num_train_epochs="${num_train_epochs:-${NUM_TRAIN_EPOCHS:-6}}"
position="${position:-${POSITION:-f7+l7}}"
dropout="${dropout:-${DROPOUT:-0.05}}"
target_layers="${target_layers:-${TARGET_LAYERS:--1}}"
model_max_length="${model_max_length:-${MODEL_MAX_LENGTH:-512}}"
run_suffix="${run_suffix:-${RUN_SUFFIX:-}}"
safety_dataset_path="${safety_dataset_path:-${SAFETY_DATASET_PATH:-}}"

if [[ "${model_name_or_path}" == */snapshots/* ]]; then
  model_tag="${model_name_or_path%/snapshots/*}"
  model_tag="${model_tag##*/}"
else
  model_tag="${model_name_or_path##*/}"
fi

if [[ -n "${LRS:-}" ]]; then
  read -r -a lr_arr <<< "${LRS}"
else
  case "${model_tag}" in
    llama3-8b)
      lr_arr=("7e-4" "9e-4" "1.1e-3")
      ;;
    mistral-7b-v0.1|mistral-7b-v0.3)
      lr_arr=("6e-4" "8e-4" "1e-3")
      ;;
    qwen25-7b)
      lr_arr=("7e-4" "9e-4" "1.1e-3")
      ;;
    qwen25-3b|qwen3-4b)
      lr_arr=("8e-4" "1e-3" "1.2e-3")
      ;;
    qwen25-14b)
      lr_arr=("4e-4" "6e-4" "8e-4")
      ;;
    *)
      lr_arr=("6e-4" "8e-4" "1e-3")
      ;;
  esac
fi

RUN_TAG="${RUN_TAG:-${model_tag}_${subspace}_sweep_$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"

launch_one() {
  local lr="$1"
  local partition="$2"
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//+}"
  local job_name="ft_${model_tag}_${subspace}_lr${lr_tag}"
  local host_log_file="${HOST_LOG_DIR}/${job_name}.log"

  echo "[launch] model=${model_tag} subspace=${subspace} lr=${lr} partition=${partition} log=${host_log_file}"

  srun \
    -p "${partition}" \
    --gres="${GRES}" \
    --job-name="${job_name}" \
    ${SRUN_EXTRA_ARGS} \
    apptainer exec \
      --cleanenv \
      --nv \
      --bind /mnt:/mnt \
      --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
      "${IMAGE_PATH}" \
      bash -c "
        cd '${WORKSPACE_CONT}' &&
        export WANDB_MODE='${WANDB_MODE_VALUE}' &&
        export model_name_or_path='${model_name_or_path}' &&
        export subspace='${subspace}' &&
        export learning_rate='${lr}' &&
        export num_processes='${num_processes}' &&
        export per_device_train_batch_size='${per_device_train_batch_size}' &&
        export gradient_accumulation_steps='${gradient_accumulation_steps}' &&
        export subspace_rank='${subspace_rank}' &&
        export warmup_ratio='${warmup_ratio}' &&
        export weight_decay='${weight_decay}' &&
        export num_train_epochs='${num_train_epochs}' &&
        export position='${position}' &&
        export dropout='${dropout}' &&
        export target_layers='${target_layers}' &&
        export model_max_length='${model_max_length}' &&
        export run_suffix='${run_suffix}' &&
        export SAFETY_DATASET_PATH='${safety_dataset_path}' &&
        bash multi_train/script/first_train_multi_model.sh
      " \
    > "${host_log_file}" 2>&1 &
}

echo "PARTITIONS=${PARTITIONS_RAW}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
echo "model_name_or_path=${model_name_or_path}"
echo "model_tag=${model_tag}"
echo "subspace=${subspace}"
echo "safety_dataset_path=${safety_dataset_path}"
echo "LRS=${lr_arr[*]}"
echo "HOST_LOG_DIR=${HOST_LOG_DIR}"

for idx in "${!lr_arr[@]}"; do
  partition="${PARTITION_ARR[$((idx % ${#PARTITION_ARR[@]}))]}"
  launch_one "${lr_arr[$idx]}" "${partition}"
done

wait

echo "Sweep finished."
echo "Host logs: ${HOST_LOG_DIR}"
