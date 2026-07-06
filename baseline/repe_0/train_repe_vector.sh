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
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
LAYERS="${LAYERS:--1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_JSON="${SOURCE_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
SEED="${SEED:-42}"
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE:-0}"

read -r -a LAYER_ARR <<< "${LAYERS}"

CMD=(
  "${PYTHON_BIN}" baseline/repe_0/train_vectors.py
  --task "${TASK}"
  --base_model "${BASE_MODEL}"
  --layers "${LAYER_ARR[@]}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
  --keep_feature_cache "${KEEP_FEATURE_CACHE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${SOURCE_JSON}" ]]; then
  CMD+=(--source_json "${SOURCE_JSON}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${FEATURE_CACHE_DIR}" ]]; then
  CMD+=(--feature_cache_dir "${FEATURE_CACHE_DIR}")
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "TASK=${TASK}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "LAYERS=${LAYERS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "SOURCE_JSON=${SOURCE_JSON}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}"
echo "DEVICE=${DEVICE}"
echo "TORCH_DTYPE=${TORCH_DTYPE}"
echo "SEED=${SEED}"
echo "KEEP_FEATURE_CACHE=${KEEP_FEATURE_CACHE}"

"${CMD[@]}"
