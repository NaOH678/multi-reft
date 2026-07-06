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

normalize_tag() {
  local raw="$1"
  raw="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  raw="${raw// /-}"
  raw="${raw//\//-}"
  raw="${raw//:/-}"
  raw="${raw//,/--}"
  echo "${raw}" | sed 's/[^a-z0-9._-]/-/g; s/--*/-/g; s/^-//; s/-$//'
}

derive_eval_name() {
  if [[ -n "${EVAL_NAME:-}" ]]; then
    echo "$(normalize_tag "${EVAL_NAME}")"
    return 0
  fi

  if [[ -n "${ITI_ARTIFACT_DIR:-}" ]]; then
    echo "single-$(normalize_tag "$(basename "${ITI_ARTIFACT_DIR%/}")")"
    return 0
  fi

  echo "naive-$(normalize_tag "${ITI_COMPOSITION:-mean}")"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
ITI_ARTIFACT_DIR="${ITI_ARTIFACT_DIR:-}"
ITI_ARTIFACT_DIRS="${ITI_ARTIFACT_DIRS:-}"
ITI_ALPHA="${ITI_ALPHA:-1.0}"
ITI_TOKEN_STRATEGY="${ITI_TOKEN_STRATEGY:-last}"
ITI_INCLUDE_PROMPT="${ITI_INCLUDE_PROMPT:-0}"
ITI_COMPOSITION="${ITI_COMPOSITION:-single}"
ITI_WEIGHTS="${ITI_WEIGHTS:-}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
RUN_NAME="$(derive_eval_name)"

RUN_DIR="baseline/iti_0/runs/eval/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
OUTPUT_DIR="${RUN_DIR}/outputs"
LOG_FILE="${RUN_DIR}/eval.log"
MANIFEST_FILE="${RUN_DIR}/run_manifest.env"
SUMMARY_PATH="${RUN_DIR}/summary.json"

mkdir -p "${OUTPUT_DIR}"

if [[ "${TASK}" == "truth" || "${TASK}" == "bias" ]]; then
  RESULTS_PATH="${OUTPUT_DIR}/results.json"
else
  RESULTS_PATH="${OUTPUT_DIR}/generations.csv"
fi

cat > "${MANIFEST_FILE}" <<EOF
RUN_TS=${RUN_TS}
TASK=${TASK}
RUN_NAME=${RUN_NAME}
PYTHON_BIN=${PYTHON_BIN}
BASE_MODEL=${BASE_MODEL}
MODEL_NAME=${MODEL_NAME}
ITI_ARTIFACT_DIR=${ITI_ARTIFACT_DIR}
ITI_ARTIFACT_DIRS=${ITI_ARTIFACT_DIRS}
ITI_ALPHA=${ITI_ALPHA}
ITI_TOKEN_STRATEGY=${ITI_TOKEN_STRATEGY}
ITI_INCLUDE_PROMPT=${ITI_INCLUDE_PROMPT}
ITI_COMPOSITION=${ITI_COMPOSITION}
ITI_WEIGHTS=${ITI_WEIGHTS}
DEVICE=${DEVICE}
TARGET_LAYERS=${TARGET_LAYERS}
SUBSPACE_RANK=${SUBSPACE_RANK}
RUN_DIR=${RUN_DIR}
OUTPUT_DIR=${OUTPUT_DIR}
RESULTS_PATH=${RESULTS_PATH}
SUMMARY_PATH=${SUMMARY_PATH}
LOG_FILE=${LOG_FILE}
EOF

echo "RUN_DIR=${RUN_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "MANIFEST_FILE=${MANIFEST_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "RESULTS_PATH=${RESULTS_PATH}"
echo "SUMMARY_PATH=${SUMMARY_PATH}"

exec >> "${LOG_FILE}" 2>&1

echo "run_ts=${RUN_TS}"
echo "task=${TASK}"
echo "run_name=${RUN_NAME}"
echo "base_model=${BASE_MODEL}"
echo "model_name=${MODEL_NAME}"
echo "iti_artifact_dir=${ITI_ARTIFACT_DIR}"
echo "iti_artifact_dirs=${ITI_ARTIFACT_DIRS}"
echo "iti_alpha=${ITI_ALPHA}"
echo "iti_token_strategy=${ITI_TOKEN_STRATEGY}"
echo "iti_include_prompt=${ITI_INCLUDE_PROMPT}"
echo "iti_composition=${ITI_COMPOSITION}"
echo "output_dir=${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN}" \
TASK="${TASK}" \
BASE_MODEL="${BASE_MODEL}" \
ITI_ARTIFACT_DIR="${ITI_ARTIFACT_DIR}" \
ITI_ARTIFACT_DIRS="${ITI_ARTIFACT_DIRS}" \
ITI_ALPHA="${ITI_ALPHA}" \
ITI_TOKEN_STRATEGY="${ITI_TOKEN_STRATEGY}" \
ITI_INCLUDE_PROMPT="${ITI_INCLUDE_PROMPT}" \
ITI_COMPOSITION="${ITI_COMPOSITION}" \
ITI_WEIGHTS="${ITI_WEIGHTS}" \
DEVICE="${DEVICE}" \
TARGET_LAYERS="${TARGET_LAYERS}" \
SUBSPACE_RANK="${SUBSPACE_RANK}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
RESULTS_PATH="${RESULTS_PATH}" \
bash baseline/iti_0/evaluate_iti_task.sh

echo "summary_file=${SUMMARY_PATH}"
