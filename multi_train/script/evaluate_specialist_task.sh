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
REFT_WEIGHTS="${REFT_WEIGHTS:-}"
LORA_WEIGHTS="${LORA_WEIGHTS:-}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
DEVICE="${DEVICE:-cuda:0}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
RESULT_PREFIX="${RESULT_PREFIX:-}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-}"

resolve_base_model_path() {
  local candidate="$1"
  local normalized="${candidate%/}"
  local refs_main
  local snapshot_id
  local repo_root

  if [[ -d "${normalized}" && -f "${normalized}/config.json" ]]; then
    echo "${normalized}"
    return 0
  fi

  refs_main="${normalized}/refs/main"
  if [[ -f "${refs_main}" ]]; then
    snapshot_id="$(tr -d '[:space:]' < "${refs_main}")"
    if [[ -n "${snapshot_id}" && -d "${normalized}/snapshots/${snapshot_id}" ]]; then
      echo "${normalized}/snapshots/${snapshot_id}"
      return 0
    fi
  fi

  if [[ "${normalized}" == */snapshots/* ]]; then
    repo_root="${normalized%/snapshots/*}"
    refs_main="${repo_root}/refs/main"
    if [[ -f "${refs_main}" ]]; then
      snapshot_id="$(tr -d '[:space:]' < "${refs_main}")"
      if [[ -n "${snapshot_id}" && -d "${repo_root}/snapshots/${snapshot_id}" ]]; then
        echo "${repo_root}/snapshots/${snapshot_id}"
        return 0
      fi
    fi
  fi

  echo "${normalized}"
}

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"

if [[ -n "${REFT_WEIGHTS}" && -n "${LORA_WEIGHTS}" ]]; then
  echo "REFT_WEIGHTS and LORA_WEIGHTS cannot be set at the same time." >&2
  exit 1
fi

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"

COMMON_MODEL_ARGS=(
  --base_model "${BASE_MODEL}"
  --target_layers "${TARGET_LAYER_ARR[@]}"
  --subspace_rank "${SUBSPACE_RANK}"
  --device "${DEVICE}"
)

if [[ -n "${REFT_WEIGHTS}" ]]; then
  COMMON_MODEL_ARGS+=(--reft_weights "${REFT_WEIGHTS}")
fi
if [[ -n "${LORA_WEIGHTS}" ]]; then
  COMMON_MODEL_ARGS+=(--lora_weights "${LORA_WEIGHTS}")
fi

echo "TASK=${TASK}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "REFT_WEIGHTS=${REFT_WEIGHTS}"
echo "LORA_WEIGHTS=${LORA_WEIGHTS}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "SUBSPACE_RANK=${SUBSPACE_RANK}"
echo "DEVICE=${DEVICE}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"
echo "RESULT_PREFIX=${RESULT_PREFIX}"
echo "CHECKPOINT_NAME=${CHECKPOINT_NAME}"
echo "TASK_OUTPUT_ROOT=${TASK_OUTPUT_ROOT}"

case "${TASK}" in
  truth|bias)
    if [[ "${TASK}" == "truth" ]]; then
      DATASET="${TRUTH_DATASET:-${DATASET:-truthfulqa_mc}}"
      BATCH_SIZE="${TRUTH_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${TRUTH_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${TRUTH_GREEDY_DECODING:-${GREEDY_DECODING:-1}}"
    else
      DATASET="${BIAS_DATASET:-${DATASET:-bbq}}"
      BATCH_SIZE="${BIAS_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${BIAS_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${BIAS_GREEDY_DECODING:-${GREEDY_DECODING:-1}}"
    fi

    RESULTS_ARGS=()
    if [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
      RESULTS_JSON="${TASK_OUTPUT_ROOT}/${CHECKPOINT_NAME}/${TASK}/${DATASET}.json"
      mkdir -p "$(dirname "${RESULTS_JSON}")"
      RESULTS_ARGS+=(--results_json "${RESULTS_JSON}")
    elif [[ -n "${RESULT_PREFIX}" ]]; then
      RESULTS_JSON="multi_train/eval_truth/${RESULT_PREFIX}-${DATASET}.json"
      RESULTS_ARGS+=(--results_json "${RESULTS_JSON}")
    fi

    SUMMARY_ARGS=()
    if [[ -n "${SUMMARY_PATH}" ]]; then
      SUMMARY_ARGS+=(--summary_file "${SUMMARY_PATH}")
    fi

    echo "DATASET=${DATASET}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "POSITIONS=${POSITIONS}"
    echo "GREEDY_DECODING=${GREEDY_DECODING}"
    if [[ -n "${RESULTS_JSON:-}" ]]; then
      echo "RESULTS_JSON=${RESULTS_JSON}"
    fi

    "${PYTHON_BIN}" multi_train/eval_truth/evaluate_truth.py \
      "${COMMON_MODEL_ARGS[@]}" \
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
    if [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
      RESULTS_CSV="${TASK_OUTPUT_ROOT}/${CHECKPOINT_NAME}/ethics/generations.csv"
      mkdir -p "$(dirname "${RESULTS_CSV}")"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    elif [[ -n "${RESULT_PREFIX}" ]]; then
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
    if [[ -n "${RESULTS_CSV:-}" ]]; then
      echo "RESULTS_CSV=${RESULTS_CSV}"
    fi

    "${PYTHON_BIN}" multi_train/eval_ethics/machine_ethics_exp.py \
      "${COMMON_MODEL_ARGS[@]}" \
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
    ALLOW_ANALYSIS_FALLBACK="${TOXICITY_ALLOW_ANALYSIS_FALLBACK:-${ALLOW_ANALYSIS_FALLBACK:-0}}"
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
    if [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
      RESULTS_CSV="${TASK_OUTPUT_ROOT}/${CHECKPOINT_NAME}/toxicity/generations.csv"
      mkdir -p "$(dirname "${RESULTS_CSV}")"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    elif [[ -n "${RESULT_PREFIX}" ]]; then
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
    echo "ALLOW_ANALYSIS_FALLBACK=${ALLOW_ANALYSIS_FALLBACK}"
    echo "ANALYSIS_BATCH_SIZE=${ANALYSIS_BATCH_SIZE}"
    echo "DETOXIFY_MODEL=${DETOXIFY_MODEL}"
    echo "ANALYSIS_DEVICE=${ANALYSIS_DEVICE}"
    if [[ -n "${RESULTS_CSV:-}" ]]; then
      echo "RESULTS_CSV=${RESULTS_CSV}"
    fi

    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_MODEL_ARGS[@]}" \
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
      --allow_analysis_fallback "${ALLOW_ANALYSIS_FALLBACK}" \
      --analysis_batch_size "${ANALYSIS_BATCH_SIZE}" \
      --detoxify_model "${DETOXIFY_MODEL}" \
      --analysis_device "${ANALYSIS_DEVICE}" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Unsupported TASK=${TASK}. Expected one of: truth bias ethics toxicity." >&2
    exit 1
    ;;
esac
