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
DEVICE="${DEVICE:-cuda:1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
POSITIONS="${POSITIONS:-7}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
COMPOSE_DOMAIN="${COMPOSE_DOMAIN:-output}"
COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE:-1.0}"
COMPOSITION_TOPK="${COMPOSITION_TOPK:-2}"
COMPAT_THRESHOLD="${COMPAT_THRESHOLD:-0.0}"
SCORE_SOURCE="${SCORE_SOURCE:-intervention_norm}"
SCORE_NORMALIZER="${SCORE_NORMALIZER:-log_zscore}"
TEXT_COLUMN="${TEXT_COLUMN:-input}"

SPEC1="${SPEC1:-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

TRUTH_DATASET="${TRUTH_DATASET:-dataset/alignment_truthful_format}"
MORAL_DATASET="${MORAL_DATASET:-dataset/alignment_moral_cls}"
BIAS_DATASET="${BIAS_DATASET:-dataset/alignment_stereotype_format}"
TOXICITY_DATASET="${TOXICITY_DATASET:-dataset/alignment_toxic_format}"

SCORE_STATS_PATH="${SCORE_STATS_PATH:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"
CALIB_DIR="${CALIB_DIR:-multi_train/calibration/router_features/train_input}"
PROMPT_DIR="${PROMPT_DIR:-${CALIB_DIR}/prompt_files}"
REPORT_DIR="${REPORT_DIR:-${CALIB_DIR}/compat_reports}"
mkdir -p "${PROMPT_DIR}" "${REPORT_DIR}"

TRUTH_PROMPTS="${PROMPT_DIR}/truthful_train_input.jsonl"
MORAL_PROMPTS="${PROMPT_DIR}/moral_train_input.jsonl"
BIAS_PROMPTS="${PROMPT_DIR}/stereotype_train_input.jsonl"
TOXICITY_PROMPTS="${PROMPT_DIR}/toxicity_train_input.jsonl"
OUTPUT_JSON="${OUTPUT_JSON:-${REPORT_DIR}/compat_correctness_e6.json}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"

EXPORT_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXPORT_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Exporting prompt-only calibration files for compatibility correctness..."
"${PYTHON_BIN}" -m multi_train.eval_common.export_prompt_field \
  --dataset_path "${TRUTH_DATASET}" \
  --output_file "${TRUTH_PROMPTS}" \
  --split train \
  --text_column "${TEXT_COLUMN}" \
  --strip \
  "${EXPORT_ARGS[@]}"

"${PYTHON_BIN}" -m multi_train.eval_common.export_prompt_field \
  --dataset_path "${MORAL_DATASET}" \
  --output_file "${MORAL_PROMPTS}" \
  --split train \
  --text_column "${TEXT_COLUMN}" \
  --strip \
  "${EXPORT_ARGS[@]}"

"${PYTHON_BIN}" -m multi_train.eval_common.export_prompt_field \
  --dataset_path "${BIAS_DATASET}" \
  --output_file "${BIAS_PROMPTS}" \
  --split train \
  --text_column "${TEXT_COLUMN}" \
  --strip \
  "${EXPORT_ARGS[@]}"

"${PYTHON_BIN}" -m multi_train.eval_common.export_prompt_field \
  --dataset_path "${TOXICITY_DATASET}" \
  --output_file "${TOXICITY_PROMPTS}" \
  --split train \
  --text_column "${TEXT_COLUMN}" \
  --strip \
  "${EXPORT_ARGS[@]}"

echo "Running compatibility correctness check..."
"${PYTHON_BIN}" -m multi_train.eval_common.check_compat_correctness \
  --base_model "${BASE_MODEL}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --max_samples "${MAX_SAMPLES}" \
  --positions "${POSITIONS}" \
  --target_layers "${TARGET_LAYER_ARR[@]}" \
  --compose_domain "${COMPOSE_DOMAIN}" \
  --composition_temperature "${COMPOSITION_TEMPERATURE}" \
  --composition_topk "${COMPOSITION_TOPK}" \
  --compat_threshold "${COMPAT_THRESHOLD}" \
  --score_source "${SCORE_SOURCE}" \
  --score_normalizer "${SCORE_NORMALIZER}" \
  --score_stats_path "${SCORE_STATS_PATH}" \
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}" \
  --prompt_files "${TRUTH_PROMPTS}" "${MORAL_PROMPTS}" "${BIAS_PROMPTS}" "${TOXICITY_PROMPTS}" \
  --output_json "${OUTPUT_JSON}"

echo "Compatibility correctness report saved to: ${OUTPUT_JSON}"
