#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

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
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SOURCE_JSON="${SOURCE_JSON:-}"
DEVICE="${DEVICE:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TOP_K_HEADS="${TOP_K_HEADS:-48}"
MAX_ITER="${MAX_ITER:-1000}"
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE:-0}"
RETAIN_LAYERS="${RETAIN_LAYERS:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TOXICITY_PROMPT_TYPES_FOR_TRAINING="${TOXICITY_PROMPT_TYPES_FOR_TRAINING:-benign adversarial}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"

OUTPUT_DIR="${OUTPUT_DIR:-baseline/iti_pyvene_0/artifacts/interventions/${MODEL_NAME}/${TASK}/${RUN_NAME}}"
RUN_DIR="baseline/iti_pyvene_0/runs/train/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
LOG_FILE="${RUN_DIR}/train.log"
MANIFEST_FILE="${RUN_DIR}/run_manifest.env"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${RUN_DIR}/feature_cache}"
PREPARED_SOURCE_JSON=""

mkdir -p "${RUN_DIR}"

if [[ "${TASK}" == "toxicity" && -z "${SOURCE_JSON}" ]]; then
  DEFAULT_SOURCE_JSON="dataset/caa/toxicity_10k/train.json"
  PREPARED_SOURCE_JSON="${RUN_DIR}/prepared_toxicity_prompt_source.json"
  read -r -a TOXICITY_PROMPT_TYPE_ARR <<< "${TOXICITY_PROMPT_TYPES_FOR_TRAINING}"
  "${PYTHON_BIN}" baseline/iti_pyvene_0/prepare_toxicity_prompt_dataset.py \
    --source_json "${DEFAULT_SOURCE_JSON}" \
    --output_json "${PREPARED_SOURCE_JSON}" \
    --prompt_types "${TOXICITY_PROMPT_TYPE_ARR[@]}"
  SOURCE_JSON="${PREPARED_SOURCE_JSON}"
fi

if [[ "${TASK}" == "bias" && -z "${SOURCE_JSON}" ]]; then
  DEFAULT_SOURCE_JSON="dataset/caa/bias/train.json"
  PREPARED_SOURCE_JSON="${RUN_DIR}/prepared_bias_train.json"
  "${PYTHON_BIN}" baseline/caa_pyvene_0/prepare_bias_training_dataset.py \
    --source_json "${DEFAULT_SOURCE_JSON}" \
    --output_json "${PREPARED_SOURCE_JSON}"
  SOURCE_JSON="${PREPARED_SOURCE_JSON}"
fi

mkdir -p "${OUTPUT_DIR}"

cat > "${MANIFEST_FILE}" <<EOF
RUN_TS=${RUN_TS}
TASK=${TASK}
RUN_NAME=${RUN_NAME}
PYTHON_BIN=${PYTHON_BIN}
BASE_MODEL=${BASE_MODEL}
MODEL_NAME=${MODEL_NAME}
BATCH_SIZE=${BATCH_SIZE}
MAX_SAMPLES=${MAX_SAMPLES}
SOURCE_JSON=${SOURCE_JSON}
PREPARED_SOURCE_JSON=${PREPARED_SOURCE_JSON}
OUTPUT_DIR=${OUTPUT_DIR}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}
DEVICE=${DEVICE}
TORCH_DTYPE=${TORCH_DTYPE}
SEED=${SEED}
VAL_RATIO=${VAL_RATIO}
TOP_K_HEADS=${TOP_K_HEADS}
MAX_ITER=${MAX_ITER}
KEEP_FEATURE_CACHE=${KEEP_FEATURE_CACHE}
RETAIN_LAYERS=${RETAIN_LAYERS}
TOXICITY_PROMPT_TYPES_FOR_TRAINING=${TOXICITY_PROMPT_TYPES_FOR_TRAINING}
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
echo "source_json=${SOURCE_JSON}"
echo "output_dir=${OUTPUT_DIR}"
echo "retain_layers=${RETAIN_LAYERS}"

PYTHON_BIN="${PYTHON_BIN}" \
TASK="${TASK}" \
BASE_MODEL="${BASE_MODEL}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
SOURCE_JSON="${SOURCE_JSON}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR}" \
DEVICE="${DEVICE}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
SEED="${SEED}" \
VAL_RATIO="${VAL_RATIO}" \
TOP_K_HEADS="${TOP_K_HEADS}" \
MAX_ITER="${MAX_ITER}" \
KEEP_FEATURE_CACHE="${KEEP_FEATURE_CACHE}" \
RETAIN_LAYERS="${RETAIN_LAYERS}" \
bash baseline/iti_pyvene_0/train_iti_intervention.sh

echo "intervention_summary=${OUTPUT_DIR}/summary.json"
