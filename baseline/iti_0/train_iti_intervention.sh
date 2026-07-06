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
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_JSON="${SOURCE_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TOP_K_HEADS="${TOP_K_HEADS:-48}"
MAX_ITER="${MAX_ITER:-1000}"
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE:-0}"
RETAIN_LAYERS="${RETAIN_LAYERS:-}"

CMD=(
  "${PYTHON_BIN}" baseline/iti_0/train_iti.py
  --task "${TASK}"
  --base_model "${BASE_MODEL}"
  --batch_size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
  --val_ratio "${VAL_RATIO}"
  --top_k_heads "${TOP_K_HEADS}"
  --max_iter "${MAX_ITER}"
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
if [[ -n "${RETAIN_LAYERS}" ]]; then
  read -r -a RETAIN_LAYER_ARR <<< "${RETAIN_LAYERS}"
  CMD+=(--retain_layers "${RETAIN_LAYER_ARR[@]}")
fi

"${CMD[@]}"
