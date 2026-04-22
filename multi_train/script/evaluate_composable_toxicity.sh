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
POSITIONS="${POSITIONS:-11}"
DATASETS="${DATASETS:-toxic nontoxic}"
PROMPTS="${PROMPTS:-benign adversarial}"
DATASET_ROOT="${DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
N_GENERATIONS="${N_GENERATIONS:-5}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-32}"
DETOXIFY_MODEL="${DETOXIFY_MODEL:-original}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-cuda:0}"
SHARED_BASIS_RANK="${SHARED_BASIS_RANK:-8}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a DATASET_ARR <<< "${DATASETS}"
read -r -a PROMPT_ARR <<< "${PROMPTS}"

SPEC1="${SPEC1:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model/}"
SPEC3="${SPEC3:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model/}"
SPEC4="${SPEC4:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 8 ]]; then
  echo "Need at least 8 GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-multi_train/logs/composable_toxicity_$(date +%Y%m%d_%H%M%S)}"
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
    echo "DATASETS=${DATASETS}"
    echo "PROMPTS=${PROMPTS}"
    echo "DATASET_ROOT=${DATASET_ROOT}"
    echo "MAX_SAMPLES=${MAX_SAMPLES}"
    echo "SPEC1=${SPEC1}"
    echo "SPEC2=${SPEC2}"
    echo "SPEC3=${SPEC3}"
    echo "SPEC4=${SPEC4}"
    echo "EXTRA_ARGS=$*"
    echo
    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_ARGS[@]}" \
      --device "cuda:${gpu_id}" \
      "$@"
  } >"${log_file}" 2>&1
}

echo "BASE_MODEL=${BASE_MODEL}"
echo "DATASETS=${DATASETS}"
echo "PROMPTS=${PROMPTS}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "SPEC1=${SPEC1}"
echo "SPEC2=${SPEC2}"
echo "SPEC3=${SPEC3}"
echo "SPEC4=${SPEC4}"
echo "GPU_IDS=${GPU_IDS_VALUE}"
echo "LOG_DIR=${LOG_DIR}"

run_exp "E1_single_output" "${GPU_IDS_ARR[0]}" \
  --compose_domain output \
  --composition_method single \
  --single_index 3 &

run_exp "E2_equal_output" "${GPU_IDS_ARR[1]}" \
  --compose_domain output \
  --composition_method equal &

run_exp "E3_residual_output" "${GPU_IDS_ARR[2]}" \
  --compose_domain output \
  --composition_method residual_softmax \
  --composition_temperature 1.0 &

run_exp "E4_topk1_output" "${GPU_IDS_ARR[3]}" \
  --compose_domain output \
  --composition_method topk_residual \
  --composition_topk 1 \
  --composition_temperature 1.0 &

run_exp "E5_equal_shared_orth_identity" "${GPU_IDS_ARR[4]}" \
  --compose_domain shared_latent \
  --composition_method equal \
  --shared_basis_type orth_mean \
  --shared_basis_rank "${SHARED_BASIS_RANK}" \
  --transport_type identity &

run_exp "E6_residual_shared_orth_identity" "${GPU_IDS_ARR[5]}" \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type orth_mean \
  --shared_basis_rank "${SHARED_BASIS_RANK}" \
  --transport_type identity &

run_exp "E7_residual_shared_svd_identity" "${GPU_IDS_ARR[6]}" \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank "${SHARED_BASIS_RANK}" \
  --transport_type identity &

run_exp "E8_residual_shared_svd_overlap" "${GPU_IDS_ARR[7]}" \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank "${SHARED_BASIS_RANK}" \
  --transport_type overlap &

wait

run_exp "E9_residual_projected_svd" "${GPU_IDS_ARR[0]}" \
  --compose_domain projected_output \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank "${SHARED_BASIS_RANK}"

echo "All composable toxicity experiments finished."
echo "Logs saved to: ${LOG_DIR}"
echo "Generated summaries:"
find multi_train/eval_toxicity -type f \( -name '*composable*summary.json' -o -name '*composable*.csv' \) -printf '%T@ %p\n' | sort -n | tail -n 40 | cut -d' ' -f2-
