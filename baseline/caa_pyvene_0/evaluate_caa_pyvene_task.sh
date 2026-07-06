#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
CAA_VECTOR_DIR="${CAA_VECTOR_DIR:-}"
CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS:-}"
CAA_LAYERS="${CAA_LAYERS:-}"
CAA_ALPHA="${CAA_ALPHA:-1.0}"
CAA_COMPOSITION="${CAA_COMPOSITION:-single}"
CAA_WEIGHTS="${CAA_WEIGHTS:-}"
CAA_INTERVENE_ON_PROMPT="${CAA_INTERVENE_ON_PROMPT:-0}"
CAA_BASE_UNIT_LOCATION="${CAA_BASE_UNIT_LOCATION:-}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
RESULTS_PATH="${RESULTS_PATH:-}"
RESULT_PREFIX="${RESULT_PREFIX:-}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-}"

if [[ -n "${CAA_VECTOR_DIR}" && -n "${CAA_VECTOR_DIRS}" ]]; then
  echo "Use either CAA_VECTOR_DIR or CAA_VECTOR_DIRS, not both." >&2
  exit 1
fi

if [[ -z "${CAA_VECTOR_DIR}" && -z "${CAA_VECTOR_DIRS}" ]]; then
  echo "CAA_VECTOR_DIR or CAA_VECTOR_DIRS must be set." >&2
  exit 1
fi

COMMON_PYVENE_ARGS=(
  --pyvene_method_tag "caa-pyvene"
  --pyvene_component "block_output"
  --pyvene_alpha "${CAA_ALPHA}"
  --pyvene_composition "${CAA_COMPOSITION}"
  --pyvene_intervene_on_prompt "${CAA_INTERVENE_ON_PROMPT}"
)

if [[ -n "${CAA_BASE_UNIT_LOCATION}" ]]; then
  COMMON_PYVENE_ARGS+=(--pyvene_base_unit_location "${CAA_BASE_UNIT_LOCATION}")
fi

if [[ -n "${CAA_VECTOR_DIR}" ]]; then
  COMMON_PYVENE_ARGS+=(--pyvene_vector_dir "${CAA_VECTOR_DIR}")
fi

if [[ -n "${CAA_VECTOR_DIRS}" ]]; then
  read -r -a CAA_VECTOR_DIR_ARR <<< "${CAA_VECTOR_DIRS}"
  COMMON_PYVENE_ARGS+=(--pyvene_vector_dirs "${CAA_VECTOR_DIR_ARR[@]}")
fi

if [[ -n "${CAA_WEIGHTS}" ]]; then
  read -r -a CAA_WEIGHT_ARR <<< "${CAA_WEIGHTS}"
  COMMON_PYVENE_ARGS+=(--pyvene_weights "${CAA_WEIGHT_ARR[@]}")
fi

if [[ -n "${CAA_LAYERS}" && "${CAA_LAYERS}" != "-1" ]]; then
  read -r -a CAA_LAYER_ARR <<< "${CAA_LAYERS}"
  COMMON_PYVENE_ARGS+=(--pyvene_layers "${CAA_LAYER_ARR[@]}")
fi

COMMON_MODEL_ARGS=(
  --base_model "${BASE_MODEL}"
  --target_layers "${TARGET_LAYERS}"
  --subspace_rank "${SUBSPACE_RANK}"
  --device "${DEVICE}"
)

case "${TASK}" in
  truth|bias)
    if [[ "${TASK}" == "truth" ]]; then
      DATASET="${TRUTH_DATASET:-${DATASET:-truthfulqa_mc}}"
      BATCH_SIZE="${TRUTH_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      POSITIONS="${TRUTH_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${TRUTH_GREEDY_DECODING:-${GREEDY_DECODING:-1}}"
      EVALUATION_MODE="${TRUTH_EVALUATION_MODE:-${EVALUATION_MODE:-auto}}"
      CANDIDATE_SCORE_REDUCTION="${TRUTH_CANDIDATE_SCORE_REDUCTION:-${CANDIDATE_SCORE_REDUCTION:-mean}}"
    else
      DATASET="${BIAS_DATASET:-${DATASET:-bbq}}"
      BATCH_SIZE="${BIAS_BATCH_SIZE:-${BATCH_SIZE:-16}}"
      POSITIONS="${BIAS_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${BIAS_GREEDY_DECODING:-${GREEDY_DECODING:-1}}"
    fi
    RESULTS_ARGS=()
    if [[ -n "${RESULTS_PATH}" ]]; then
      RESULTS_JSON="${RESULTS_PATH}"
      mkdir -p "$(dirname "${RESULTS_JSON}")"
      RESULTS_ARGS+=(--results_json "${RESULTS_JSON}")
    elif [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
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
    "${PYTHON_BIN}" multi_train/eval_truth/evaluate_truth.py \
      "${COMMON_MODEL_ARGS[@]}" \
      "${COMMON_PYVENE_ARGS[@]}" \
      --dataset "${DATASET}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --truth_eval_mode "direct" \
      "${RESULTS_ARGS[@]}" \
      "${SUMMARY_ARGS[@]}"
    ;;
  ethics)
    BATCH_SIZE="${ETHICS_BATCH_SIZE:-${BATCH_SIZE:-8}}"
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
    if [[ -n "${RESULTS_PATH}" ]]; then
      RESULTS_CSV="${RESULTS_PATH}"
      mkdir -p "$(dirname "${RESULTS_CSV}")"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    elif [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
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
    "${PYTHON_BIN}" multi_train/eval_ethics/machine_ethics_exp.py \
      "${COMMON_MODEL_ARGS[@]}" \
      "${COMMON_PYVENE_ARGS[@]}" \
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
    BATCH_SIZE="${TOXICITY_BATCH_SIZE:-${BATCH_SIZE:-8}}"
    POSITIONS="${TOXICITY_POSITIONS:-${POSITIONS:-11}}"
    GREEDY_DECODING="${TOXICITY_GREEDY_DECODING:-${GREEDY_DECODING:-0}}"
    DATASETS="${TOXICITY_DATASETS:-${DATASETS:-toxic nontoxic}}"
    PROMPTS="${TOXICITY_PROMPTS:-${PROMPTS:-benign adversarial}}"
    DATASET_ROOT="${TOXICITY_DATASET_ROOT:-${DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}}"
    MAX_SAMPLES="${TOXICITY_MAX_SAMPLES:-${MAX_SAMPLES:-}}"
    N_GENERATIONS="${TOXICITY_N_GENERATIONS:-${N_GENERATIONS:-1}}"
    MAX_TOKENS="${TOXICITY_MAX_TOKENS:-${MAX_TOKENS:-50}}"
    TEMPERATURE="${TOXICITY_TEMPERATURE:-${TEMPERATURE:-0.7}}"
    RUN_ANALYSIS="${TOXICITY_RUN_ANALYSIS:-${RUN_ANALYSIS:-1}}"
    ALLOW_ANALYSIS_FALLBACK="${TOXICITY_ALLOW_ANALYSIS_FALLBACK:-${ALLOW_ANALYSIS_FALLBACK:-0}}"
    ANALYSIS_BATCH_SIZE="${TOXICITY_ANALYSIS_BATCH_SIZE:-${ANALYSIS_BATCH_SIZE:-16}}"
    DETOXIFY_MODEL="${TOXICITY_DETOXIFY_MODEL:-${DETOXIFY_MODEL:-original}}"
    ANALYSIS_DEVICE="${TOXICITY_ANALYSIS_DEVICE:-${ANALYSIS_DEVICE:-${DEVICE}}}"
    read -r -a DATASET_ARR <<< "${DATASETS}"
    read -r -a PROMPT_ARR <<< "${PROMPTS}"
    EXTRA_ARGS=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
      EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
    fi
    RESULTS_ARGS=()
    if [[ -n "${RESULTS_PATH}" ]]; then
      RESULTS_CSV="${RESULTS_PATH}"
      mkdir -p "$(dirname "${RESULTS_CSV}")"
      RESULTS_ARGS+=(--results_csv "${RESULTS_CSV}")
    elif [[ -n "${TASK_OUTPUT_ROOT}" && -n "${CHECKPOINT_NAME}" ]]; then
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
    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_MODEL_ARGS[@]}" \
      "${COMMON_PYVENE_ARGS[@]}" \
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
