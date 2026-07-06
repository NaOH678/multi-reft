#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
RUN_PREFIX="${RUN_PREFIX:-${TASK}_l14_l26}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ITI_TOP_K_HEADS="${ITI_TOP_K_HEADS:-48}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_JSON="${SOURCE_JSON:-}"
SEED="${SEED:-42}"

launch_job() {
  local gpu_id="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" &
}

run_caa() {
  local gpu_id="$1"
  local layer="$2"
  local run_name="${RUN_PREFIX}_caa_l${layer}"
  launch_job "${gpu_id}" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    TASK="${TASK}" \
    BASE_MODEL="${BASE_MODEL}" \
    RUN_NAME="${run_name}" \
    LAYERS="${layer}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    SOURCE_JSON="${SOURCE_JSON}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    DEVICE="cuda:0" \
    bash baseline/caa_pyvene_0/run_train_experiment.sh
}

run_iti() {
  local gpu_id="$1"
  local layer="$2"
  local run_name="${RUN_PREFIX}_iti_l${layer}_k${ITI_TOP_K_HEADS}"
  launch_job "${gpu_id}" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    TASK="${TASK}" \
    BASE_MODEL="${BASE_MODEL}" \
    RUN_NAME="${run_name}" \
    RETAIN_LAYERS="${layer}" \
    TOP_K_HEADS="${ITI_TOP_K_HEADS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    SOURCE_JSON="${SOURCE_JSON}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    DEVICE="cuda:0" \
    SEED="${SEED}" \
    bash baseline/iti_pyvene_0/run_train_experiment.sh
}

run_repe() {
  local gpu_id="$1"
  local layer="$2"
  local run_name="${RUN_PREFIX}_repe_l${layer}"
  launch_job "${gpu_id}" env \
    PYTHON_BIN="${PYTHON_BIN}" \
    TASK="${TASK}" \
    BASE_MODEL="${BASE_MODEL}" \
    RUN_NAME="${run_name}" \
    LAYERS="${layer}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    SOURCE_JSON="${SOURCE_JSON}" \
    TORCH_DTYPE="${TORCH_DTYPE}" \
    DEVICE="cuda:0" \
    SEED="${SEED}" \
    bash baseline/repe_pyvene_0/run_train_experiment.sh
}

gpu_arr=(${GPU_IDS})
if (( ${#gpu_arr[@]} < 6 )); then
  echo "Need at least 6 GPU ids in GPU_IDS, got: ${GPU_IDS}" >&2
  exit 1
fi

echo "TASK=${TASK}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "RUN_PREFIX=${RUN_PREFIX}"
echo "GPU_IDS=${GPU_IDS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "ITI_TOP_K_HEADS=${ITI_TOP_K_HEADS}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "SOURCE_JSON=${SOURCE_JSON}"

run_caa "${gpu_arr[0]}" 14
run_caa "${gpu_arr[1]}" 26
run_iti "${gpu_arr[2]}" 14
run_iti "${gpu_arr[3]}" 26
run_repe "${gpu_arr[4]}" 14
run_repe "${gpu_arr[5]}" 26

wait

echo "Finished 6-way training for TASK=${TASK}, RUN_PREFIX=${RUN_PREFIX}"
