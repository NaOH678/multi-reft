#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

resolve_default_layers() {
  local task_name="$1"
  local base_model_path="$2"
  PYTHON_BIN="${PYTHON_BIN}" TASK_NAME="${task_name}" BASE_MODEL_PATH="${base_model_path}" "${PYTHON_BIN}" - <<'PY'
import os
from baseline.repe_0.task_registry import resolve_default_single_layer, should_use_default_single_layer

task_name = os.environ["TASK_NAME"]
base_model_path = os.environ["BASE_MODEL_PATH"]

if should_use_default_single_layer(task_name):
    print(resolve_default_single_layer(base_model_path, task_name))
else:
    print("-1")
PY
}

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

normalize_model_name() {
  local path_value="$1"
  local normalized="${path_value%/}"
  local stripped="${normalized#models/}"
  if [[ "${stripped}" != "${normalized}" ]]; then
    echo "${stripped%%/*}" | tr '[:upper:]' '[:lower:]'
    return 0
  fi
  if [[ "${normalized}" == */snapshots/* ]]; then
    local before_snapshots="${normalized%/snapshots/*}"
    echo "${before_snapshots##*/}" | tr '[:upper:]' '[:lower:]'
    return 0
  fi
  echo "${normalized##*/}" | tr '[:upper:]' '[:lower:]'
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
RUN_NAME="${RUN_NAME:-default}"
RAW_LAYERS="${LAYERS-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_JSON="${SOURCE_JSON:-}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
SEED="${SEED:-42}"
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE:-0}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
if [[ -n "${RAW_LAYERS}" ]]; then
  LAYERS="${RAW_LAYERS}"
else
  LAYERS="$(resolve_default_layers "${TASK}" "${BASE_MODEL}")"
fi

OUTPUT_DIR="${OUTPUT_DIR:-baseline/repe_0/artifacts/vectors/${MODEL_NAME}/${TASK}/${RUN_NAME}}"
RUN_DIR="baseline/repe_0/runs/train/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
LOG_FILE="${RUN_DIR}/train.log"
MANIFEST_FILE="${RUN_DIR}/run_manifest.env"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${RUN_DIR}/feature_cache}"

mkdir -p "${RUN_DIR}"
mkdir -p "${OUTPUT_DIR}"

cat > "${MANIFEST_FILE}" <<EOF
RUN_TS=${RUN_TS}
TASK=${TASK}
RUN_NAME=${RUN_NAME}
PYTHON_BIN=${PYTHON_BIN}
BASE_MODEL=${BASE_MODEL}
MODEL_NAME=${MODEL_NAME}
LAYERS=${LAYERS}
BATCH_SIZE=${BATCH_SIZE}
MAX_SAMPLES=${MAX_SAMPLES}
SOURCE_JSON=${SOURCE_JSON}
OUTPUT_DIR=${OUTPUT_DIR}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}
DEVICE=${DEVICE}
TORCH_DTYPE=${TORCH_DTYPE}
SEED=${SEED}
KEEP_FEATURE_CACHE=${KEEP_FEATURE_CACHE}
LOG_FILE=${LOG_FILE}
EOF

echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "MANIFEST_FILE=${MANIFEST_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

exec >> "${LOG_FILE}" 2>&1

echo "run_ts=${RUN_TS}"
echo "task=${TASK}"
echo "run_name=${RUN_NAME}"
echo "base_model=${BASE_MODEL}"
echo "model_name=${MODEL_NAME}"
echo "output_dir=${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN}" \
TASK="${TASK}" \
BASE_MODEL="${BASE_MODEL}" \
LAYERS="${LAYERS}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
SOURCE_JSON="${SOURCE_JSON}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
DEVICE="${DEVICE}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
SEED="${SEED}" \
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE}" \
bash baseline/repe_0/train_repe_vector.sh

echo "vector_summary=${OUTPUT_DIR}/summary.json"
