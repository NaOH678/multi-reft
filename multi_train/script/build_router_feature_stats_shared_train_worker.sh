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
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
POSITIONS="${POSITIONS:-7}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
FEATURE_NAMES="${FEATURE_NAMES:-delta_norm intervention_norm subspace_energy mean_compat neg_compat_mass}"
TEXT_COLUMN="${TEXT_COLUMN:-input}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

SPEC1="${SPEC1:-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

TRUTH_DATASET="${TRUTH_DATASET:-dataset/alignment_truthful_format}"
MORAL_DATASET="${MORAL_DATASET:-dataset/alignment_moral_cls}"
BIAS_DATASET="${BIAS_DATASET:-dataset/alignment_stereotype_format}"
TOXICITY_DATASET="${TOXICITY_DATASET:-dataset/alignment_toxic_format}"

CALIB_DIR="${CALIB_DIR:-multi_train/calibration/router_features/train_input}"
PROMPT_DIR="${PROMPT_DIR:-${CALIB_DIR}/prompt_files}"
STATS_DIR="${STATS_DIR:-${CALIB_DIR}/stats}"
mkdir -p "${PROMPT_DIR}" "${STATS_DIR}"

TRUTH_PROMPTS="${PROMPT_DIR}/truthful_train_input.jsonl"
MORAL_PROMPTS="${PROMPT_DIR}/moral_train_input.jsonl"
BIAS_PROMPTS="${PROMPT_DIR}/stereotype_train_input.jsonl"
TOXICITY_PROMPTS="${PROMPT_DIR}/toxicity_train_input.jsonl"
OUTPUT_JSON="${OUTPUT_JSON:-${STATS_DIR}/router_feature_stats_shared_train_input.json}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a FEATURE_NAME_ARR <<< "${FEATURE_NAMES}"

EXPORT_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  EXPORT_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "Exporting prompt-only calibration files for shared router feature stats..."
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

echo "Building shared router feature stats..."
"${PYTHON_BIN}" -m multi_train.eval_common.build_router_feature_stats \
  --base_model "${BASE_MODEL}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --positions "${POSITIONS}" \
  --target_layers "${TARGET_LAYER_ARR[@]}" \
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}" \
  --calibration_mode shared \
  --prompt_files "${TRUTH_PROMPTS}" "${MORAL_PROMPTS}" "${BIAS_PROMPTS}" "${TOXICITY_PROMPTS}" \
  --feature_names "${FEATURE_NAME_ARR[@]}" \
  --text_column "${TEXT_COLUMN}" \
  --output_json "${OUTPUT_JSON}"

echo "Shared router feature stats saved to: ${OUTPUT_JSON}"
