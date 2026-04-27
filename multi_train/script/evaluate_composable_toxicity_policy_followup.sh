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

BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
GREEDY_DECODING="${GREEDY_DECODING:-0}"
POSITIONS="${POSITIONS:-11}"
DATASETS="${DATASETS:-toxic nontoxic}"
PROMPTS="${PROMPTS:-benign adversarial}"
DATASET_ROOT="${DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
N_GENERATIONS="${N_GENERATIONS:-5}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-16}"
DETOXIFY_MODEL="${DETOXIFY_MODEL:-original}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cuda:0}"
COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE:-1.0}"
COMPOSITION_TOPK="${COMPOSITION_TOPK:-2}"
COMPAT_THRESHOLD="${COMPAT_THRESHOLD:-0.0}"

STATS_RES_SHARED="${STATS_RES_SHARED:-multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json}"
STATS_RES_SPECIALIST="${STATS_RES_SPECIALIST:-multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json}"
STATS_INT_SHARED="${STATS_INT_SHARED:-multi_train/calibration/train_input/stats/intervention_stats_shared_train_input.json}"
STATS_INT_SPECIALIST="${STATS_INT_SPECIALIST:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a PROMPT_ARR <<< "${PROMPTS}"

SPEC1="${SPEC1:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model/}"
SPEC3="${SPEC3:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model/}"
SPEC4="${SPEC4:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 4 ]]; then
  echo "Need at least 4 GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS_ARR[@]}}"

LOG_DIR="${LOG_DIR:-multi_train/logs/composable_toxicity_policy_followup_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

COMMON_ARGS=(
  --base_model "${BASE_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --positions "${POSITIONS}"
  --greedy_decoding "${GREEDY_DECODING}"
  --datasets "${DATASET_ARR[@]}"
  --prompts "${PROMPT_ARR[@]}"
  --dataset_root "${DATASET_ROOT}"
  --n_generations "${N_GENERATIONS}"
  --max_tokens "${MAX_TOKENS}"
  --temperature "${TEMPERATURE}"
  --analysis_batch_size "${ANALYSIS_BATCH_SIZE}"
  --detoxify_model "${DETOXIFY_MODEL}"
  --analysis_device "${ANALYSIS_DEVICE}"
  --compose_domain output
  --composition_temperature "${COMPOSITION_TEMPERATURE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

run_exp() {
  local exp_id="$1"
  local gpu_id="$2"
  shift 2
  local log_file="${LOG_DIR}/${exp_id}.log"

  echo "[${exp_id}] gpu=cuda:${gpu_id} log=${log_file}"
  {
    echo "===== ${exp_id} ====="
    echo "GPU=cuda:${gpu_id}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "TARGET_LAYERS=${TARGET_LAYERS}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "DATASETS=${DATASETS}"
    echo "PROMPTS=${PROMPTS}"
    echo "DATASET_ROOT=${DATASET_ROOT}"
    echo "MAX_SAMPLES=${MAX_SAMPLES}"
    echo "COMPOSITION_TEMPERATURE=${COMPOSITION_TEMPERATURE}"
    echo "COMPOSITION_TOPK=${COMPOSITION_TOPK}"
    echo "COMPAT_THRESHOLD=${COMPAT_THRESHOLD}"
    echo "STATS_RES_SHARED=${STATS_RES_SHARED}"
    echo "STATS_RES_SPECIALIST=${STATS_RES_SPECIALIST}"
    echo "STATS_INT_SHARED=${STATS_INT_SHARED}"
    echo "STATS_INT_SPECIALIST=${STATS_INT_SPECIALIST}"
    echo "EXTRA_ARGS=$*"
    echo
    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_ARGS[@]}" \
      --device "cuda:${gpu_id}" \
      "$@"
  } >"${log_file}" 2>&1
}

job_idx=0
for cfg in \
  "E1 compat_delta_none --composition_method compat_filtered_topk --score_source delta_norm --score_normalizer none --composition_topk ${COMPOSITION_TOPK} --compat_threshold ${COMPAT_THRESHOLD}" \
  "E2 compat_delta_mean --composition_method compat_filtered_topk --score_source delta_norm --score_normalizer mean_ratio --score_stats_path ${STATS_RES_SHARED} --composition_topk ${COMPOSITION_TOPK} --compat_threshold ${COMPAT_THRESHOLD}" \
  "E3 compat_delta_logz --composition_method compat_filtered_topk --score_source delta_norm --score_normalizer log_zscore --score_stats_path ${STATS_RES_SPECIALIST} --composition_topk ${COMPOSITION_TOPK} --compat_threshold ${COMPAT_THRESHOLD}" \
  "E4 intervention_none --composition_method intervention_softmax --score_source intervention_norm --score_normalizer none" \
  "E5 compat_intervention_mean --composition_method compat_filtered_topk --score_source intervention_norm --score_normalizer mean_ratio --score_stats_path ${STATS_INT_SHARED} --composition_topk ${COMPOSITION_TOPK} --compat_threshold ${COMPAT_THRESHOLD}" \
  "E6 compat_intervention_logz --composition_method compat_filtered_topk --score_source intervention_norm --score_normalizer log_zscore --score_stats_path ${STATS_INT_SPECIALIST} --composition_topk ${COMPOSITION_TOPK} --compat_threshold ${COMPAT_THRESHOLD}" \
  "E7 intervention_mean --composition_method intervention_softmax --score_source intervention_norm --score_normalizer mean_ratio --score_stats_path ${STATS_INT_SHARED}" \
  "E8 intervention_logz --composition_method intervention_softmax --score_source intervention_norm --score_normalizer log_zscore --score_stats_path ${STATS_INT_SPECIALIST}"
do
  read -r exp_id tag rest <<< "${cfg}"
  gpu_id="${GPU_IDS_ARR[$((job_idx % ${#GPU_IDS_ARR[@]}))]}"
  run_exp "${exp_id}_${tag}" "${gpu_id}" ${rest} &
  job_idx=$((job_idx + 1))
  if (( job_idx % MAX_PARALLEL == 0 )); then
    wait
  fi
done

wait

echo "All composable toxicity follow-up policy experiments finished."
echo "Logs saved to: ${LOG_DIR}"
