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
  local vector_dir="$3"
  PYTHON_BIN="${PYTHON_BIN}" TASK_NAME="${task_name}" BASE_MODEL_PATH="${base_model_path}" "${PYTHON_BIN}" - <<'PY'
import os
import json
from pathlib import Path
from baseline.CAA_0.task_registry import resolve_default_single_layer, should_use_default_single_layer
from multi_train.eval_common.output_naming import normalize_base_model_name

task_name = os.environ["TASK_NAME"]
base_model_path = os.environ["BASE_MODEL_PATH"]
vector_dir = os.environ.get("VECTOR_DIR", "").strip()

if task_name == "ethics" and vector_dir:
    summary_path = Path(vector_dir) / "summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text())
        layers = payload.get("layers") or sorted(int(k) for k in payload.get("vector_files", {}).keys())
        if layers:
            print(" ".join(str(int(layer)) for layer in layers))
            raise SystemExit(0)

if task_name == "truth":
    model_name = normalize_base_model_name(base_model_path)
    summary_root = Path("baseline/CAA_0/runs/eval") / model_name / task_name / "autolayer-single-truth_default"
    candidates = sorted(summary_root.glob("*/layer_selection_summary.json"))
    if candidates:
        payload = json.loads(candidates[-1].read_text())
        best_layer = payload.get("best_layer")
        if best_layer is not None:
            print(int(best_layer))
            raise SystemExit(0)

if should_use_default_single_layer(task_name):
    print(resolve_default_single_layer(base_model_path, task_name))
else:
    print("")
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

  if [[ -n "${CAA_VECTOR_DIR:-}" ]]; then
    echo "single-$(normalize_tag "$(basename "${CAA_VECTOR_DIR%/}")")"
    return 0
  fi

  echo "naive-$(normalize_tag "${CAA_COMPOSITION:-mean}")"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
CAA_VECTOR_DIR="${CAA_VECTOR_DIR:-}"
CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS:-}"
CAA_LAYERS="${CAA_LAYERS:-}"
CAA_ALPHA="${CAA_ALPHA:-}"
CAA_COMPOSITION="${CAA_COMPOSITION:-single}"
CAA_WEIGHTS="${CAA_WEIGHTS:-}"
CAA_INTERVENE_ON_PROMPT="${CAA_INTERVENE_ON_PROMPT:-0}"
CAA_BASE_UNIT_LOCATION="${CAA_BASE_UNIT_LOCATION:-}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
RUN_NAME="$(derive_eval_name)"
if [[ -z "${CAA_LAYERS}" || "${CAA_LAYERS}" == "-1" ]]; then
  VECTOR_DIR_FOR_RESOLVE="${CAA_VECTOR_DIR:-}"
  CAA_LAYERS="$(VECTOR_DIR="${VECTOR_DIR_FOR_RESOLVE}" resolve_default_layers "${TASK}" "${BASE_MODEL}" "${VECTOR_DIR_FOR_RESOLVE}")"
fi
if [[ -z "${CAA_ALPHA}" ]]; then
  if [[ "${TASK}" == "truth" ]]; then
    CAA_ALPHA="0.2"
  else
    CAA_ALPHA="1.0"
  fi
fi

RUN_DIR="baseline/caa_pyvene_0/runs/eval/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
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
CAA_VECTOR_DIR=${CAA_VECTOR_DIR}
CAA_VECTOR_DIRS=${CAA_VECTOR_DIRS}
CAA_LAYERS=${CAA_LAYERS}
CAA_ALPHA=${CAA_ALPHA}
CAA_COMPOSITION=${CAA_COMPOSITION}
CAA_WEIGHTS=${CAA_WEIGHTS}
PYVENE_METHOD_TAG=caa-pyvene
PYVENE_COMPONENT=block_output
PYVENE_INTERVENE_ON_PROMPT=${CAA_INTERVENE_ON_PROMPT}
PYVENE_BASE_UNIT_LOCATION=${CAA_BASE_UNIT_LOCATION}
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
echo "caa_vector_dir=${CAA_VECTOR_DIR}"
echo "caa_vector_dirs=${CAA_VECTOR_DIRS}"
echo "caa_layers=${CAA_LAYERS}"
echo "caa_alpha=${CAA_ALPHA}"
echo "caa_composition=${CAA_COMPOSITION}"
echo "caa_intervene_on_prompt=${CAA_INTERVENE_ON_PROMPT}"
echo "caa_base_unit_location=${CAA_BASE_UNIT_LOCATION}"
echo "output_dir=${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN}" \
TASK="${TASK}" \
BASE_MODEL="${BASE_MODEL}" \
CAA_VECTOR_DIR="${CAA_VECTOR_DIR}" \
CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS}" \
CAA_LAYERS="${CAA_LAYERS}" \
CAA_ALPHA="${CAA_ALPHA}" \
CAA_COMPOSITION="${CAA_COMPOSITION}" \
CAA_WEIGHTS="${CAA_WEIGHTS}" \
CAA_INTERVENE_ON_PROMPT="${CAA_INTERVENE_ON_PROMPT}" \
CAA_BASE_UNIT_LOCATION="${CAA_BASE_UNIT_LOCATION}" \
DEVICE="${DEVICE}" \
TARGET_LAYERS="${TARGET_LAYERS}" \
SUBSPACE_RANK="${SUBSPACE_RANK}" \
SUMMARY_PATH="${SUMMARY_PATH}" \
RESULTS_PATH="${RESULTS_PATH}" \
bash baseline/caa_pyvene_0/evaluate_caa_pyvene_task.sh

echo "summary_file=${SUMMARY_PATH}"
