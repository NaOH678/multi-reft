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

TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
ROUTER_CHECKPOINT_DIR="${ROUTER_CHECKPOINT_DIR:-}"
ROUTER_METADATA_PATH="${ROUTER_METADATA_PATH:-}"
BASE_SCORE_STATS_PATH="${BASE_SCORE_STATS_PATH:-}"
ROUTER_FEATURE_STATS_PATH="${ROUTER_FEATURE_STATS_PATH:-}"
ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE:-1}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
DEVICE="${DEVICE:-cuda:0}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
RESULT_PREFIX="${RESULT_PREFIX:-}"

if [[ -z "${ROUTER_CHECKPOINT_DIR}" ]]; then
  echo "ROUTER_CHECKPOINT_DIR must be set." >&2
  exit 1
fi

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"

  COMMON_ROUTER_ARGS=(
  --base_model "${BASE_MODEL}"
  --router_checkpoint_dir "${ROUTER_CHECKPOINT_DIR}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --device "${DEVICE}"
  --enable_debug_cache "${ENABLE_DEBUG_CACHE}"
)

if [[ -n "${ROUTER_METADATA_PATH}" ]]; then
  COMMON_ROUTER_ARGS+=(--router_metadata_path "${ROUTER_METADATA_PATH}")
fi
if [[ -n "${BASE_SCORE_STATS_PATH}" ]]; then
  COMMON_ROUTER_ARGS+=(--base_score_stats_path "${BASE_SCORE_STATS_PATH}")
fi
if [[ -n "${ROUTER_FEATURE_STATS_PATH}" ]]; then
  COMMON_ROUTER_ARGS+=(--router_feature_stats_path "${ROUTER_FEATURE_STATS_PATH}")
fi

echo "TASK=${TASK}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "ROUTER_CHECKPOINT_DIR=${ROUTER_CHECKPOINT_DIR}"
echo "ROUTER_METADATA_PATH=${ROUTER_METADATA_PATH}"
echo "BASE_SCORE_STATS_PATH=${BASE_SCORE_STATS_PATH}"
echo "ROUTER_FEATURE_STATS_PATH=${ROUTER_FEATURE_STATS_PATH}"
echo "ENABLE_DEBUG_CACHE=${ENABLE_DEBUG_CACHE}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "DEVICE=${DEVICE}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "RESULT_PREFIX=${RESULT_PREFIX}"

case "${TASK}" in
  truth|bias)
    if [[ "${TASK}" == "truth" ]]; then
      DATASET="${TRUTH_DATASET:-${DATASET:-truthfulqa_mc}}"
      BATCH_SIZE="${TRUTH_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${TRUTH_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${TRUTH_GREEDY_DECODING:-${GREEDY_DECODING:-True}}"
    else
      DATASET="${BIAS_DATASET:-${DATASET:-bbq}}"
      BATCH_SIZE="${BIAS_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${BIAS_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${BIAS_GREEDY_DECODING:-${GREEDY_DECODING:-True}}"
    fi

    RESULTS_ARGS=()
    if [[ -n "${RESULT_PREFIX}" ]]; then
      RESULTS_JSON="multi_train/eval_truth/${RESULT_PREFIX}-${DATASET}.json"
      RESULTS_ARGS+=(--results_json "${RESULTS_JSON}")
    fi

    echo "DATASET=${DATASET}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    if [[ -n "${RESULT_PREFIX}" ]]; then
      echo "RESULTS_JSON=${RESULTS_JSON}"
    fi

    SUMMARY_ARGS=()
    if [[ -n "${SUMMARY_PATH}" ]]; then
      SUMMARY_ARGS+=(--summary_file "${SUMMARY_PATH}")
    fi

    "${PYTHON_BIN}" multi_train/eval_truth/evaluate_truth.py \
      "${COMMON_ROUTER_ARGS[@]}" \
      --dataset "${DATASET}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}"
    ;;

  ethics)
    BATCH_SIZE="${ETHICS_BATCH_SIZE:-${BATCH_SIZE:-128}}"
    POSITIONS="${ETHICS_POSITIONS:-${POSITIONS:-7}}"
    GREEDY_DECODING="${ETHICS_GREEDY_DECODING:-${GREEDY_DECODING:-0}}"
    PROMPT_TYPES="${ETHICS_PROMPT_TYPES:-${PROMPT_TYPES:-0 1 2 3 4 5}}"
    DATASET_FILE="${ETHICS_DATASET_FILE:-${DATASET_FILE:-multi_train/eval_ethics/data/ethics/cm_test.csv}}"
    ONLY_SHORT="${ETHICS_ONLY_SHORT:-${ONLY_SHORT:-1}}"
    MAX_SAMPLES="${ETHICS_MAX_SAMPLES:-${MAX_SAMPLES:-}}"
    N_GENERATIONS="${ETHICS_N_GENERATIONS:-${N_GENERATIONS:-1}}"
    MAX_TOKENS="${ETHICS_MAX_TOKENS:-${MAX_TOKENS:-100}}"
    TEMPERATURE="${ETHICS_TEMPERATURE:-${TEMPERATURE:-0.6}}"

    read -r -a PROMPT_TYPE_ARR <<< "${PROMPT_TYPES}"

    EXTRA_ARGS=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
      EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
    fi
    RESULTS_ARGS=()
    if [[ -n "${RESULT_PREFIX}" ]]; then
      RESULTS_CSV="multi_train/eval_ethics/data/generations/${RESULT_PREFIX}-ethics.csv"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    fi
    SUMMARY_ARGS=()
    if [[ -n "${SUMMARY_PATH}" ]]; then
      SUMMARY_ARGS+=(--summary_json "${SUMMARY_PATH}")
    fi

    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "PROMPT_TYPES=${PROMPT_TYPES}"
    echo "DATASET_FILE=${DATASET_FILE}"
    echo "ONLY_SHORT=${ONLY_SHORT}"
    echo "MAX_SAMPLES=${MAX_SAMPLES}"
    echo "N_GENERATIONS=${N_GENERATIONS}"
    echo "MAX_TOKENS=${MAX_TOKENS}"
    echo "TEMPERATURE=${TEMPERATURE}"
    if [[ -n "${RESULT_PREFIX}" ]]; then
      echo "RESULTS_CSV=${RESULTS_CSV}"
    fi

    "${PYTHON_BIN}" multi_train/eval_ethics/machine_ethics_exp.py \
      "${COMMON_ROUTER_ARGS[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --prompt_types "${PROMPT_TYPE_ARR[@]}" \
      --dataset_file "${DATASET_FILE}" \
      --only_short "${ONLY_SHORT}" \
      --n_generations "${N_GENERATIONS}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;

  privacy)
    BATCH_SIZE="${PRIVACY_BATCH_SIZE:-${BATCH_SIZE:-128}}"
    POSITIONS="${PRIVACY_POSITIONS:-${POSITIONS:-7}}"
    GREEDY_DECODING="${PRIVACY_GREEDY_DECODING:-${GREEDY_DECODING:-0}}"
    PROMPT_TYPES="${PRIVACY_PROMPT_TYPES:-${PROMPT_TYPES:-1}}"
    PII_INDICES="${PRIVACY_PII_INDICES:-${PII_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17}}"
    DATA_FILE="${PRIVACY_DATA_FILE:-${DATA_FILE:-multi_train/eval_privacy/enron_data/name2email.pkl}}"
    NUM_PROMPTS="${PRIVACY_NUM_PROMPTS:-${NUM_PROMPTS:-100}}"
    CONTEXT_EXAMPLES="${PRIVACY_CONTEXT_EXAMPLES:-${CONTEXT_EXAMPLES:-4}}"
    N_GENERATIONS="${PRIVACY_N_GENERATIONS:-${N_GENERATIONS:-3}}"
    MAX_TOKENS="${PRIVACY_MAX_TOKENS:-${MAX_TOKENS:-50}}"
    TEMPERATURE="${PRIVACY_TEMPERATURE:-${TEMPERATURE:-0.7}}"
    RUN_STATISTICS="${PRIVACY_RUN_STATISTICS:-${RUN_STATISTICS:-1}}"

    read -r -a PROMPT_TYPE_ARR <<< "${PROMPT_TYPES}"
    read -r -a PII_INDEX_ARR <<< "${PII_INDICES}"
    RESULTS_ARGS=()
    if [[ -n "${RESULT_PREFIX}" ]]; then
      RESULTS_CSV="multi_train/eval_privacy/data/generations/${RESULT_PREFIX}-privacy.csv"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    fi
    SUMMARY_ARGS=()
    if [[ -n "${SUMMARY_PATH}" ]]; then
      SUMMARY_ARGS+=(--summary_json "${SUMMARY_PATH}")
    fi

    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "PROMPT_TYPES=${PROMPT_TYPES}"
    echo "PII_INDICES=${PII_INDICES}"
    echo "DATA_FILE=${DATA_FILE}"
    echo "NUM_PROMPTS=${NUM_PROMPTS}"
    echo "CONTEXT_EXAMPLES=${CONTEXT_EXAMPLES}"
    echo "N_GENERATIONS=${N_GENERATIONS}"
    echo "MAX_TOKENS=${MAX_TOKENS}"
    echo "TEMPERATURE=${TEMPERATURE}"
    echo "RUN_STATISTICS=${RUN_STATISTICS}"
    if [[ -n "${RESULT_PREFIX}" ]]; then
      echo "RESULTS_CSV=${RESULTS_CSV}"
    fi

    "${PYTHON_BIN}" multi_train/eval_privacy/privacy_exp.py \
      "${COMMON_ROUTER_ARGS[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --prompt_types "${PROMPT_TYPE_ARR[@]}" \
      --pii_indices "${PII_INDEX_ARR[@]}" \
      --data_file "${DATA_FILE}" \
      --num_prompts "${NUM_PROMPTS}" \
      --context_examples "${CONTEXT_EXAMPLES}" \
      --n_generations "${N_GENERATIONS}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --run_statistics "${RUN_STATISTICS}" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}"
    ;;

  toxicity)
    BATCH_SIZE="${TOXICITY_BATCH_SIZE:-${BATCH_SIZE:-64}}"
    POSITIONS="${TOXICITY_POSITIONS:-${POSITIONS:-11}}"
    GREEDY_DECODING="${TOXICITY_GREEDY_DECODING:-${GREEDY_DECODING:-0}}"
    DATASETS="${TOXICITY_DATASETS:-${DATASETS:-toxic nontoxic}}"
    PROMPTS="${TOXICITY_PROMPTS:-${PROMPTS:-benign adversarial}}"
    DATASET_ROOT="${TOXICITY_DATASET_ROOT:-${DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}}"
    MAX_SAMPLES="${TOXICITY_MAX_SAMPLES:-${MAX_SAMPLES:-}}"
    N_GENERATIONS="${TOXICITY_N_GENERATIONS:-${N_GENERATIONS:-3}}"
    MAX_TOKENS="${TOXICITY_MAX_TOKENS:-${MAX_TOKENS:-50}}"
    TEMPERATURE="${TOXICITY_TEMPERATURE:-${TEMPERATURE:-0.7}}"
    RUN_ANALYSIS="${TOXICITY_RUN_ANALYSIS:-${RUN_ANALYSIS:-1}}"
    ANALYSIS_BATCH_SIZE="${TOXICITY_ANALYSIS_BATCH_SIZE:-${ANALYSIS_BATCH_SIZE:-32}}"
    DETOXIFY_MODEL="${TOXICITY_DETOXIFY_MODEL:-${DETOXIFY_MODEL:-original}}"
    ANALYSIS_DEVICE="${TOXICITY_ANALYSIS_DEVICE:-${ANALYSIS_DEVICE:-${DEVICE}}}"

    read -r -a DATASET_ARR <<< "${DATASETS}"
    read -r -a PROMPT_ARR <<< "${PROMPTS}"

    EXTRA_ARGS=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
      EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
    fi
    RESULTS_ARGS=()
    if [[ -n "${RESULT_PREFIX}" ]]; then
      RESULTS_CSV="multi_train/eval_toxicity/data/generations/${RESULT_PREFIX}-toxicity.csv"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    fi
    SUMMARY_ARGS=()
    if [[ -n "${SUMMARY_PATH}" ]]; then
      SUMMARY_ARGS+=(--summary_json "${SUMMARY_PATH}")
    fi

    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    echo "DATASETS=${DATASETS}"
    echo "PROMPTS=${PROMPTS}"
    echo "DATASET_ROOT=${DATASET_ROOT}"
    echo "MAX_SAMPLES=${MAX_SAMPLES}"
    echo "N_GENERATIONS=${N_GENERATIONS}"
    echo "MAX_TOKENS=${MAX_TOKENS}"
    echo "TEMPERATURE=${TEMPERATURE}"
    echo "RUN_ANALYSIS=${RUN_ANALYSIS}"
    echo "ANALYSIS_BATCH_SIZE=${ANALYSIS_BATCH_SIZE}"
    echo "DETOXIFY_MODEL=${DETOXIFY_MODEL}"
    echo "ANALYSIS_DEVICE=${ANALYSIS_DEVICE}"
    if [[ -n "${RESULT_PREFIX}" ]]; then
      echo "RESULTS_CSV=${RESULTS_CSV}"
    fi

    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_ROUTER_ARGS[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --datasets "${DATASET_ARR[@]}" \
      --prompts "${PROMPT_ARR[@]}" \
      --dataset_root "${DATASET_ROOT}" \
      --n_generations "${N_GENERATIONS}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --run_analysis "${RUN_ANALYSIS}" \
      --analysis_batch_size "${ANALYSIS_BATCH_SIZE}" \
      --detoxify_model "${DETOXIFY_MODEL}" \
      --analysis_device "${ANALYSIS_DEVICE}" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Unsupported TASK=${TASK}. Expected one of: truth bias ethics privacy toxicity." >&2
    exit 1
    ;;
esac
