#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
SPEC1="${SPEC1:-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"
SCORE_STATS_PATH="${SCORE_STATS_PATH:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"
SCORE_SOURCE="${SCORE_SOURCE:-intervention_norm}"
SCORE_NORMALIZER="${SCORE_NORMALIZER:-log_zscore}"
COMPOSITION_METHOD="${COMPOSITION_METHOD:-compat_filtered_topk}"
COMPOSE_DOMAIN="${COMPOSE_DOMAIN:-output}"
COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE:-1.0}"
COMPOSITION_TOPK="${COMPOSITION_TOPK:-2}"
COMPAT_THRESHOLD="${COMPAT_THRESHOLD:-0.0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE:-1}"
DEVICE="${DEVICE:-cuda:0}"
SUMMARY_PATH="${SUMMARY_PATH:-}"

TOXICITY_BATCH_SIZE="${TOXICITY_BATCH_SIZE:-8}"
TOXICITY_POSITIONS="${TOXICITY_POSITIONS:-11}"
TOXICITY_GREEDY_DECODING="${TOXICITY_GREEDY_DECODING:-0}"
TOXICITY_DATASETS="${TOXICITY_DATASETS:-toxic nontoxic}"
TOXICITY_PROMPTS="${TOXICITY_PROMPTS:-benign adversarial}"
TOXICITY_DATASET_ROOT="${TOXICITY_DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}"
TOXICITY_MAX_SAMPLES="${TOXICITY_MAX_SAMPLES:-200}"
TOXICITY_N_GENERATIONS="${TOXICITY_N_GENERATIONS:-5}"
TOXICITY_MAX_TOKENS="${TOXICITY_MAX_TOKENS:-50}"
TOXICITY_TEMPERATURE="${TOXICITY_TEMPERATURE:-0.7}"
TOXICITY_RUN_ANALYSIS="${TOXICITY_RUN_ANALYSIS:-1}"
TOXICITY_ANALYSIS_BATCH_SIZE="${TOXICITY_ANALYSIS_BATCH_SIZE:-16}"
TOXICITY_DETOXIFY_MODEL="${TOXICITY_DETOXIFY_MODEL:-original}"
TOXICITY_ANALYSIS_DEVICE="${TOXICITY_ANALYSIS_DEVICE:-${DEVICE}}"
TOXICITY_TRUTHFUL_SCORE_PENALTY="${TOXICITY_TRUTHFUL_SCORE_PENALTY:-0.0}"
TOXICITY_SAMPLE_MODE="${TOXICITY_SAMPLE_MODE:-random}"
TOXICITY_SAMPLE_SEED="${TOXICITY_SAMPLE_SEED:-42}"
TOXICITY_MECHANISM_TRACE="${TOXICITY_MECHANISM_TRACE:-1}"
TOXICITY_MECHANISM_TRACE_OUTPUT="${TOXICITY_MECHANISM_TRACE_OUTPUT:-}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a DATASET_ARR <<< "${TOXICITY_DATASETS}"
read -r -a PROMPT_ARR <<< "${TOXICITY_PROMPTS}"

RESULTS_CSV="${RESULTS_CSV:-multi_train/eval_toxicity/data/generations/$(basename "${SUMMARY_PATH%.json}")-toxicity.csv}"

COMMON_COMPOSABLE_ARGS=(
  --base_model "${BASE_MODEL}"
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --device "${DEVICE}"
  --compose_domain "${COMPOSE_DOMAIN}"
  --composition_method "${COMPOSITION_METHOD}"
  --composition_temperature "${COMPOSITION_TEMPERATURE}"
  --composition_topk "${COMPOSITION_TOPK}"
  --compat_threshold "${COMPAT_THRESHOLD}"
  --score_source "${SCORE_SOURCE}"
  --score_normalizer "${SCORE_NORMALIZER}"
  --score_stats_path "${SCORE_STATS_PATH}"
)

EXTRA_ARGS=()
if [[ -n "${TOXICITY_MAX_SAMPLES}" ]]; then
  EXTRA_ARGS+=(--max_samples "${TOXICITY_MAX_SAMPLES}")
fi
if [[ -n "${TOXICITY_MECHANISM_TRACE_OUTPUT}" ]]; then
  EXTRA_ARGS+=(--mechanism_trace_output "${TOXICITY_MECHANISM_TRACE_OUTPUT}")
fi

"${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
  "${COMMON_COMPOSABLE_ARGS[@]}" \
  --batch_size "${TOXICITY_BATCH_SIZE}" \
  --positions "${TOXICITY_POSITIONS}" \
  --greedy_decoding "${TOXICITY_GREEDY_DECODING}" \
  --datasets "${DATASET_ARR[@]}" \
  --prompts "${PROMPT_ARR[@]}" \
  --dataset_root "${TOXICITY_DATASET_ROOT}" \
  --sample_mode "${TOXICITY_SAMPLE_MODE}" \
  --sample_seed "${TOXICITY_SAMPLE_SEED}" \
  --n_generations "${TOXICITY_N_GENERATIONS}" \
  --max_tokens "${TOXICITY_MAX_TOKENS}" \
  --temperature "${TOXICITY_TEMPERATURE}" \
  --run_analysis "${TOXICITY_RUN_ANALYSIS}" \
  --analysis_batch_size "${TOXICITY_ANALYSIS_BATCH_SIZE}" \
  --detoxify_model "${TOXICITY_DETOXIFY_MODEL}" \
  --analysis_device "${TOXICITY_ANALYSIS_DEVICE}" \
  --truthful_score_penalty "${TOXICITY_TRUTHFUL_SCORE_PENALTY}" \
  --enable_debug_cache "${ENABLE_DEBUG_CACHE}" \
  --mechanism_trace "${TOXICITY_MECHANISM_TRACE}" \
  --results_csv "${RESULTS_CSV}" \
  --summary_json "${SUMMARY_PATH}" \
  "${EXTRA_ARGS[@]}"
