#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/shared-storage-user/zhoujiawei/miniconda3/envs/multi-reft/bin/python}"

BASE_MODEL="${BASE_MODEL:-../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
GREEDY_DECODING="${GREEDY_DECODING:-0}"
POSITIONS="${POSITIONS:-7}"
PROMPT_TYPES="${PROMPT_TYPES:-0 1 2 3 4 5}"
DATASET_FILE="${DATASET_FILE:-multi_train/eval_ethics/data/ethics/cm_test.csv}"
ONLY_SHORT="${ONLY_SHORT:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
N_GENERATIONS="${N_GENERATIONS:-1}"
MAX_TOKENS="${MAX_TOKENS:-100}"
TEMPERATURE="${TEMPERATURE:-0.6}"
RESIDUAL_TEMPS="${RESIDUAL_TEMPS:-2 4 8 16 18 24 32 64}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a PROMPT_TYPE_ARR <<< "${PROMPT_TYPES}"
read -r -a RESIDUAL_TEMP_ARR <<< "${RESIDUAL_TEMPS}"

SPEC1="${SPEC1:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model/}"
SPEC3="${SPEC3:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model/}"
SPEC4="${SPEC4:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -lt "${#RESIDUAL_TEMP_ARR[@]}" ]]; then
  echo "Need at least ${#RESIDUAL_TEMP_ARR[@]} GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-multi_train/logs/composable_ethics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --base_model "${BASE_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --positions "${POSITIONS}"
  --greedy_decoding "${GREEDY_DECODING}"
  --prompt_types "${PROMPT_TYPE_ARR[@]}"
  --dataset_file "${DATASET_FILE}"
  --only_short "${ONLY_SHORT}"
  --n_generations "${N_GENERATIONS}"
  --max_tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

run_exp() {
  local exp_id="$1"
  local gpu_id="$2"
  shift 2
  local log_file="${LOG_DIR}/${exp_id}.log"

  echo "[${exp_id}] gpu=${gpu_id} log=${log_file}"
  {
    echo "===== ${exp_id} ====="
    echo "GPU=cuda:${gpu_id}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "TARGET_LAYERS=${TARGET_LAYERS}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "PROMPT_TYPES=${PROMPT_TYPES}"
    echo "DATASET_FILE=${DATASET_FILE}"
    echo "ONLY_SHORT=${ONLY_SHORT}"
    echo "MAX_SAMPLES=${MAX_SAMPLES}"
    echo "N_GENERATIONS=${N_GENERATIONS}"
    echo "MAX_TOKENS=${MAX_TOKENS}"
    echo "TEMPERATURE=${TEMPERATURE}"
    echo "SPEC1=${SPEC1}"
    echo "SPEC2=${SPEC2}"
    echo "SPEC3=${SPEC3}"
    echo "SPEC4=${SPEC4}"
    echo "EXTRA_ARGS=$*"
    echo
    "${PYTHON_BIN}" multi_train/eval_ethics/machine_ethics_exp.py \
      "${COMMON_ARGS[@]}" \
      --device "cuda:${gpu_id}" \
      "$@"
  } >"${log_file}" 2>&1
}

echo "BASE_MODEL=${BASE_MODEL}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "POSITIONS=${POSITIONS}"
echo "PROMPT_TYPES=${PROMPT_TYPES}"
echo "DATASET_FILE=${DATASET_FILE}"
echo "ONLY_SHORT=${ONLY_SHORT}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "N_GENERATIONS=${N_GENERATIONS}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "RESIDUAL_TEMPS=${RESIDUAL_TEMPS}"
echo "SPEC1=${SPEC1}"
echo "SPEC2=${SPEC2}"
echo "SPEC3=${SPEC3}"
echo "SPEC4=${SPEC4}"
echo "GPU_IDS=${GPU_IDS_VALUE}"
echo "LOG_DIR=${LOG_DIR}"

for idx in "${!RESIDUAL_TEMP_ARR[@]}"; do
  temp="${RESIDUAL_TEMP_ARR[$idx]}"
  gpu_id="${GPU_IDS_ARR[$idx]}"
  exp_id="E3_residual_output_tmp${temp}"
  run_exp "${exp_id}" "${gpu_id}" \
    --compose_domain output \
    --composition_method residual_softmax \
    --composition_temperature "${temp}" &
done

wait

echo "All composable ethics residual temperature experiments finished."
echo "Logs saved to: ${LOG_DIR}"
echo "Generated summaries:"
find multi_train/eval_ethics/data/generations -maxdepth 1 -type f \
  \( -name '*composable*summary.json' -o -name '*composable*debug_summary.json' -o -name '*composable*.csv' \) \
  -printf '%T@ %p\n' | sort -n | tail -n 60 | cut -d' ' -f2-
