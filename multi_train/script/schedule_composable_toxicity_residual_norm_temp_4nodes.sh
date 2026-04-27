#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-dexmanip}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

SPEC1="${SPEC1:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

NODE0_TEMPS="${NODE0_TEMPS:-0.5 1}"
NODE1_TEMPS="${NODE1_TEMPS:-2 4}"
NODE2_TEMPS="${NODE2_TEMPS:-8 16}"
NODE3_TEMPS="${NODE3_TEMPS:-32 64}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/scheduler_toxicity_residual_norm_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"

launch_group() {
  local group_name="$1"
  local temps="$2"
  local host_log_file="${HOST_LOG_DIR}/${group_name}.log"

  echo "[launch] ${group_name} temps=${temps} log=${host_log_file}"

  srun \
    -p "${PARTITION}" \
    --gres="${GRES}" \
    --job-name="${group_name}" \
    apptainer exec \
      --cleanenv \
      --nv \
      --bind /mnt:/mnt \
      --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
      "${IMAGE_PATH}" \
      bash -c "
        cd '${WORKSPACE_CONT}' &&
        export PYTHON_BIN='${PYTHON_BIN}' &&
        export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' &&
        export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' &&
        export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}' &&
        export SPEC1='${SPEC1}' &&
        export SPEC2='${SPEC2}' &&
        export SPEC3='${SPEC3}' &&
        export SPEC4='${SPEC4}' &&
        export LOG_DIR='multi_train/logs/${group_name}_${RUN_TAG}' &&
        TEMPS='${temps}' bash multi_train/script/evaluate_composable_toxicity_residual_norm_temp_test.sh
      " \
    > "${host_log_file}" 2>&1 &
}

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "RUN_TAG=${RUN_TAG}"
echo "HOST_LOG_DIR=${HOST_LOG_DIR}"
echo "NODE0_TEMPS=${NODE0_TEMPS}"
echo "NODE1_TEMPS=${NODE1_TEMPS}"
echo "NODE2_TEMPS=${NODE2_TEMPS}"
echo "NODE3_TEMPS=${NODE3_TEMPS}"

launch_group "tox_resnorm_t01" "${NODE0_TEMPS}"
launch_group "tox_resnorm_t02" "${NODE1_TEMPS}"
launch_group "tox_resnorm_t03" "${NODE2_TEMPS}"
launch_group "tox_resnorm_t04" "${NODE3_TEMPS}"

wait

echo "All four toxicity residual-normalization temperature jobs finished."
echo "Host logs: ${HOST_LOG_DIR}"
