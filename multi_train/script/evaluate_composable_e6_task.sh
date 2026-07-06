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
FORCED_SINGLE_LAYERS="${FORCED_SINGLE_LAYERS:-}"
FORCED_SINGLE_INDEX="${FORCED_SINGLE_INDEX:-0}"
DEVICE="${DEVICE:-cuda:0}"
SUMMARY_PATH="${SUMMARY_PATH:-}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"

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

echo "TASK=${TASK}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "SPEC1=${SPEC1}"
echo "SPEC2=${SPEC2}"
echo "SPEC3=${SPEC3}"
echo "SPEC4=${SPEC4}"
echo "SCORE_STATS_PATH=${SCORE_STATS_PATH}"
echo "SCORE_SOURCE=${SCORE_SOURCE}"
echo "SCORE_NORMALIZER=${SCORE_NORMALIZER}"
echo "COMPOSITION_METHOD=${COMPOSITION_METHOD}"
echo "COMPOSE_DOMAIN=${COMPOSE_DOMAIN}"
echo "COMPOSITION_TEMPERATURE=${COMPOSITION_TEMPERATURE}"
echo "COMPOSITION_TOPK=${COMPOSITION_TOPK}"
echo "COMPAT_THRESHOLD=${COMPAT_THRESHOLD}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "FORCED_SINGLE_LAYERS=${FORCED_SINGLE_LAYERS}"
echo "FORCED_SINGLE_INDEX=${FORCED_SINGLE_INDEX}"
echo "DEVICE=${DEVICE}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"

case "${TASK}" in
  truth|bias)
    if [[ "${TASK}" == "truth" ]]; then
      DATASET="${TRUTH_DATASET:-${DATASET:-truthfulqa_mc}}"
      BATCH_SIZE="${TRUTH_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${TRUTH_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${TRUTH_GREEDY_DECODING:-${GREEDY_DECODING:-True}}"
      TRUTHFUL_SCORE_PENALTY="${TRUTH_TRUTHFUL_SCORE_PENALTY:-${TRUTHFUL_SCORE_PENALTY:-0.0}}"
    else
      DATASET="${BIAS_DATASET:-${DATASET:-bbq}}"
      BATCH_SIZE="${BIAS_BATCH_SIZE:-${BATCH_SIZE:-256}}"
      POSITIONS="${BIAS_POSITIONS:-${POSITIONS:-7}}"
      GREEDY_DECODING="${BIAS_GREEDY_DECODING:-${GREEDY_DECODING:-True}}"
      TRUTHFUL_SCORE_PENALTY="${BIAS_TRUTHFUL_SCORE_PENALTY:-${TRUTHFUL_SCORE_PENALTY:-0.0}}"
    fi
    RESULTS_JSON="${RESULTS_JSON:-multi_train/eval_truth/$(basename "${SUMMARY_PATH%.json}")-${DATASET}.json}"
    EXTRA_TRUTH_ARGS=()
    if [[ -n "${FORCED_SINGLE_LAYERS}" ]]; then
      read -r -a FORCED_SINGLE_LAYER_ARR <<< "${FORCED_SINGLE_LAYERS}"
      EXTRA_TRUTH_ARGS+=(--forced_single_layers "${FORCED_SINGLE_LAYER_ARR[@]}")
      EXTRA_TRUTH_ARGS+=(--forced_single_index "${FORCED_SINGLE_INDEX}")
    fi

    "${PYTHON_BIN}" multi_train/eval_truth/evaluate_truth.py \
      "${COMMON_COMPOSABLE_ARGS[@]}" \
      --dataset "${DATASET}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --truthful_score_penalty "${TRUTHFUL_SCORE_PENALTY}" \
      --results_json "${RESULTS_JSON}" \
      --summary_file "${SUMMARY_PATH}" \
      "${EXTRA_TRUTH_ARGS[@]}"
    ;;

  ethics)
    BATCH_SIZE="${ETHICS_BATCH_SIZE:-${BATCH_SIZE:-32}}"
    POSITIONS="${ETHICS_POSITIONS:-${POSITIONS:-7}}"
    GREEDY_DECODING="${ETHICS_GREEDY_DECODING:-${GREEDY_DECODING:-0}}"
    PROMPT_TYPES="${ETHICS_PROMPT_TYPES:-${PROMPT_TYPES:-0 1 2 3 4 5}}"
    DATASET_FILE="${ETHICS_DATASET_FILE:-${DATASET_FILE:-multi_train/eval_ethics/data/ethics/cm_test.csv}}"
    ONLY_SHORT="${ETHICS_ONLY_SHORT:-${ONLY_SHORT:-1}}"
    MAX_SAMPLES="${ETHICS_MAX_SAMPLES:-${MAX_SAMPLES:-}}"
    N_GENERATIONS="${ETHICS_N_GENERATIONS:-${N_GENERATIONS:-1}}"
    MAX_TOKENS="${ETHICS_MAX_TOKENS:-${MAX_TOKENS:-100}}"
    TEMPERATURE="${ETHICS_TEMPERATURE:-${TEMPERATURE:-0.6}}"
    TRUTHFUL_SCORE_PENALTY="${ETHICS_TRUTHFUL_SCORE_PENALTY:-${TRUTHFUL_SCORE_PENALTY:-0.0}}"
    RESULTS_CSV="${RESULTS_CSV:-multi_train/eval_ethics/data/generations/$(basename "${SUMMARY_PATH%.json}")-ethics.csv}"
    read -r -a PROMPT_TYPE_ARR <<< "${PROMPT_TYPES}"

    EXTRA_ARGS=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
      EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
    fi
    if [[ -n "${FORCED_SINGLE_LAYERS}" ]]; then
      read -r -a FORCED_SINGLE_LAYER_ARR <<< "${FORCED_SINGLE_LAYERS}"
      EXTRA_ARGS+=(--forced_single_layers "${FORCED_SINGLE_LAYER_ARR[@]}")
      EXTRA_ARGS+=(--forced_single_index "${FORCED_SINGLE_INDEX}")
    fi

    "${PYTHON_BIN}" multi_train/eval_ethics/machine_ethics_exp.py \
      "${COMMON_COMPOSABLE_ARGS[@]}" \
      --batch_size "${BATCH_SIZE}" \
      --positions "${POSITIONS}" \
      --greedy_decoding "${GREEDY_DECODING}" \
      --prompt_types "${PROMPT_TYPE_ARR[@]}" \
      --dataset_file "${DATASET_FILE}" \
      --only_short "${ONLY_SHORT}" \
      --n_generations "${N_GENERATIONS}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --truthful_score_penalty "${TRUTHFUL_SCORE_PENALTY}" \
      --results_csv "${RESULTS_CSV}" \
      --summary_json "${SUMMARY_PATH}" \
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
    TRUTHFUL_SCORE_PENALTY="${PRIVACY_TRUTHFUL_SCORE_PENALTY:-${TRUTHFUL_SCORE_PENALTY:-0.0}}"
    RESULTS_CSV="${RESULTS_CSV:-multi_train/eval_privacy/data/generations/$(basename "${SUMMARY_PATH%.json}")-privacy.csv}"
    read -r -a PROMPT_TYPE_ARR <<< "${PROMPT_TYPES}"
    read -r -a PII_INDEX_ARR <<< "${PII_INDICES}"
    EXTRA_ARGS=()
    if [[ -n "${FORCED_SINGLE_LAYERS}" ]]; then
      read -r -a FORCED_SINGLE_LAYER_ARR <<< "${FORCED_SINGLE_LAYERS}"
      EXTRA_ARGS+=(--forced_single_layers "${FORCED_SINGLE_LAYER_ARR[@]}")
      EXTRA_ARGS+=(--forced_single_index "${FORCED_SINGLE_INDEX}")
    fi

    "${PYTHON_BIN}" multi_train/eval_privacy/privacy_exp.py \
      "${COMMON_COMPOSABLE_ARGS[@]}" \
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
      --truthful_score_penalty "${TRUTHFUL_SCORE_PENALTY}" \
      --results_csv "${RESULTS_CSV}" \
      --summary_json "${SUMMARY_PATH}" \
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
    N_GENERATIONS="${TOXICITY_N_GENERATIONS:-${N_GENERATIONS:-5}}"
    MAX_TOKENS="${TOXICITY_MAX_TOKENS:-${MAX_TOKENS:-50}}"
    TEMPERATURE="${TOXICITY_TEMPERATURE:-${TEMPERATURE:-0.7}}"
    RUN_ANALYSIS="${TOXICITY_RUN_ANALYSIS:-${RUN_ANALYSIS:-1}}"
    ANALYSIS_BATCH_SIZE="${TOXICITY_ANALYSIS_BATCH_SIZE:-${ANALYSIS_BATCH_SIZE:-16}}"
    DETOXIFY_MODEL="${TOXICITY_DETOXIFY_MODEL:-${DETOXIFY_MODEL:-original}}"
    ANALYSIS_DEVICE="${TOXICITY_ANALYSIS_DEVICE:-${ANALYSIS_DEVICE:-${DEVICE}}}"
    TRUTHFUL_SCORE_PENALTY="${TOXICITY_TRUTHFUL_SCORE_PENALTY:-${TRUTHFUL_SCORE_PENALTY:-0.0}}"
    RESULTS_CSV="${RESULTS_CSV:-multi_train/eval_toxicity/data/generations/$(basename "${SUMMARY_PATH%.json}")-toxicity.csv}"
    read -r -a DATASET_ARR <<< "${DATASETS}"
    read -r -a PROMPT_ARR <<< "${PROMPTS}"

    EXTRA_ARGS=()
    if [[ -n "${MAX_SAMPLES}" ]]; then
      EXTRA_ARGS+=(--max_samples "${MAX_SAMPLES}")
    fi
    if [[ -n "${FORCED_SINGLE_LAYERS}" ]]; then
      read -r -a FORCED_SINGLE_LAYER_ARR <<< "${FORCED_SINGLE_LAYERS}"
      EXTRA_ARGS+=(--forced_single_layers "${FORCED_SINGLE_LAYER_ARR[@]}")
      EXTRA_ARGS+=(--forced_single_index "${FORCED_SINGLE_INDEX}")
    fi

    "${PYTHON_BIN}" multi_train/eval_toxicity/toxicity_exp.py \
      "${COMMON_COMPOSABLE_ARGS[@]}" \
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
      --truthful_score_penalty "${TRUTHFUL_SCORE_PENALTY}" \
      --results_csv "${RESULTS_CSV}" \
      --summary_json "${SUMMARY_PATH}" \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Unsupported TASK=${TASK}" >&2
    exit 1
    ;;
esac
