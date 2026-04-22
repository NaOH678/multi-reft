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
# truthfulqa_mc or bbq
DATASET="${DATASET:-bbq}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
GREEDY_DECODING="${GREEDY_DECODING:-True}"
POSITIONS="${POSITIONS:-7}"
COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE:-1.0}"

STATS_SHARED="${STATS_SHARED:-multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json}"
STATS_SPECIALIST="${STATS_SPECIALIST:-multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"

SPEC1="${SPEC1:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model/}"
SPEC3="${SPEC3:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model/}"
SPEC4="${SPEC4:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 4 ]]; then
  echo "Need at least 4 GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-multi_train/logs/composable_truth_residual_stats_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --dataset "${DATASET}"
  --base_model "${BASE_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --greedy_decoding "${GREEDY_DECODING}"
  --positions "${POSITIONS}"
  --compose_domain output
  --composition_temperature "${COMPOSITION_TEMPERATURE}"
)

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
    echo "DATASET=${DATASET}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "TARGET_LAYERS=${TARGET_LAYERS}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "COMPOSITION_TEMPERATURE=${COMPOSITION_TEMPERATURE}"
    echo "STATS_SHARED=${STATS_SHARED}"
    echo "STATS_SPECIALIST=${STATS_SPECIALIST}"
    echo "SPEC1=${SPEC1}"
    echo "SPEC2=${SPEC2}"
    echo "SPEC3=${SPEC3}"
    echo "SPEC4=${SPEC4}"
    echo "EXTRA_ARGS=$*"
    echo
    "${PYTHON_BIN}" multi_train/eval_truth/evaluate_truth.py \
      "${COMMON_ARGS[@]}" \
      --device "cuda:${gpu_id}" \
      "$@"
  } >"${log_file}" 2>&1
}

run_exp "E1_shared_scaled" "${GPU_IDS_ARR[0]}" \
  --composition_method residual_scaled_softmax \
  --score_normalizer mean_ratio \
  --score_stats_path "${STATS_SHARED}" &

run_exp "E2_shared_logz" "${GPU_IDS_ARR[1]}" \
  --composition_method residual_logz_softmax \
  --score_normalizer log_zscore \
  --score_stats_path "${STATS_SHARED}" &

run_exp "E3_specialist_scaled" "${GPU_IDS_ARR[2]}" \
  --composition_method residual_scaled_softmax \
  --score_normalizer mean_ratio \
  --score_stats_path "${STATS_SPECIALIST}" &

run_exp "E4_specialist_logz" "${GPU_IDS_ARR[3]}" \
  --composition_method residual_logz_softmax \
  --score_normalizer log_zscore \
  --score_stats_path "${STATS_SPECIALIST}" &

wait

echo "All composable truth residual-stats experiments finished."
echo "Logs saved to: ${LOG_DIR}"
