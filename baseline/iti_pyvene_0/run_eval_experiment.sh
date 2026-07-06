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

resolve_ethics_layers_from_artifact() {
  local artifact_dir="$1"
  PYTHON_BIN="${PYTHON_BIN}" ARTIFACT_DIR="${artifact_dir}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

artifact_dir = os.environ["ARTIFACT_DIR"]
summary_path = Path(artifact_dir) / "summary.json"
if not summary_path.exists():
    raise SystemExit(0)

payload = json.loads(summary_path.read_text())
layers = payload.get("layers")
if not layers:
    vector_files = payload.get("vector_files", {})
    layers = sorted(int(key) for key in vector_files.keys())

if layers:
    print(" ".join(str(int(layer)) for layer in layers))
PY
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
ITI_ARTIFACT_DIR="${ITI_ARTIFACT_DIR:-}"
ITI_ARTIFACT_DIRS="${ITI_ARTIFACT_DIRS:-}"
ITI_LAYERS="${ITI_LAYERS:-}"
ITI_ALPHA="${ITI_ALPHA:-1.0}"
ITI_COMPOSITION="${ITI_COMPOSITION:-single}"
ITI_WEIGHTS="${ITI_WEIGHTS:-}"
ITI_INCLUDE_PROMPT="${ITI_INCLUDE_PROMPT:-}"
ITI_BASE_UNIT_LOCATION="${ITI_BASE_UNIT_LOCATION:-}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
RUN_NAME="$(derive_eval_name)"
if [[ "${TASK}" == "ethics" && ( -z "${ITI_LAYERS}" || "${ITI_LAYERS}" == "-1" ) && -n "${ITI_ARTIFACT_DIR}" ]]; then
  ITI_LAYERS="$(resolve_ethics_layers_from_artifact "${ITI_ARTIFACT_DIR}")"
fi

if [[ -z "${ITI_INCLUDE_PROMPT}" ]]; then
  if [[ "${TASK}" == "truth" ]]; then
    ITI_INCLUDE_PROMPT=1
  else
    ITI_INCLUDE_PROMPT=0
  fi
fi

if [[ -z "${ITI_BASE_UNIT_LOCATION}" ]]; then
  if [[ "${TASK}" == "truth" ]]; then
    ITI_BASE_UNIT_LOCATION=-1
  else
    ITI_BASE_UNIT_LOCATION=
  fi
fi

PYVENE_INTERVENE_ON_PROMPT=${ITI_INCLUDE_PROMPT}
PYVENE_BASE_UNIT_LOCATION=${ITI_BASE_UNIT_LOCATION}

RUN_DIR="baseline/iti_pyvene_0/runs/eval/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
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
ITI_LAYERS=${ITI_LAYERS}
ITI_ALPHA=${ITI_ALPHA}
ITI_COMPOSITION=${ITI_COMPOSITION}
ITI_WEIGHTS=${ITI_WEIGHTS}
ITI_INCLUDE_PROMPT=${ITI_INCLUDE_PROMPT}
ITI_BASE_UNIT_LOCATION=${ITI_BASE_UNIT_LOCATION}
PYVENE_METHOD_TAG=iti-pyvene
PYVENE_COMPONENT=attention_value_output
PYVENE_INTERVENE_ON_PROMPT=${PYVENE_INTERVENE_ON_PROMPT}
PYVENE_BASE_UNIT_LOCATION=${PYVENE_BASE_UNIT_LOCATION}
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
echo "iti_layers=${ITI_LAYERS}"
echo "iti_alpha=${ITI_ALPHA}"
echo "iti_composition=${ITI_COMPOSITION}"
echo "iti_include_prompt=${ITI_INCLUDE_PROMPT}"
echo "iti_base_unit_location=${ITI_BASE_UNIT_LOCATION}"
echo "pyvene_intervene_on_prompt=${PYVENE_INTERVENE_ON_PROMPT}"
echo "pyvene_base_unit_location=${PYVENE_BASE_UNIT_LOCATION}"
echo "output_dir=${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN}" \
TASK="${TASK}" \
BASE_MODEL="${BASE_MODEL}" \
ITI_ARTIFACT_DIR="${ITI_ARTIFACT_DIR}" \
ITI_ARTIFACT_DIRS="${ITI_ARTIFACT_DIRS}" \
ITI_LAYERS="${ITI_LAYERS}" \
ITI_ALPHA="${ITI_ALPHA}" \
ITI_COMPOSITION="${ITI_COMPOSITION}" \
ITI_WEIGHTS="${ITI_WEIGHTS}" \
ITI_INCLUDE_PROMPT="${ITI_INCLUDE_PROMPT}" \
ITI_BASE_UNIT_LOCATION="${ITI_BASE_UNIT_LOCATION}" \
DEVICE="${DEVICE}" \
TARGET_LAYERS="${TARGET_LAYERS}" \
SUBSPACE_RANK="${SUBSPACE_RANK}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
RESULTS_PATH="${RESULTS_PATH}" \
bash baseline/iti_pyvene_0/evaluate_iti_pyvene_task.sh

echo "summary_file=${SUMMARY_PATH}"
