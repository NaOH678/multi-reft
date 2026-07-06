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

  if [[ -n "${REPE_VECTOR_DIR:-}" ]]; then
    local name="single-$(normalize_tag "$(basename "${REPE_VECTOR_DIR%/}")")"
    if [[ "${TASK:-}" == "toxicity" && "${REPE_TOXICITY_VARIANT:-canonical}" == "optimized" ]]; then
      name="${name}-optimized"
    fi
    echo "${name}"
    return 0
  fi

  echo "naive-$(normalize_tag "${REPE_COMPOSITION:-mean}")"
}

resolve_ethics_layers_from_vector() {
  local vector_dir="$1"
  PYTHON_BIN="${PYTHON_BIN}" VECTOR_DIR="${vector_dir}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

vector_dir = os.environ["VECTOR_DIR"]
summary_path = Path(vector_dir) / "summary.json"
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

extract_overall_score() {
  local summary_path="$1"
  PYTHON_BIN="${PYTHON_BIN}" SUMMARY_PATH_INPUT="${summary_path}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["SUMMARY_PATH_INPUT"])
if not path.exists():
    raise SystemExit(1)
payload = json.loads(path.read_text())
score = payload.get("overall_score")
if score is None:
    raise SystemExit(2)
print(score)
PY
}

write_toxicity_sweep_summary() {
  local sweep_dir="$1"
  local best_layer="$2"
  local best_alpha="$3"
  local best_score="$4"
  local best_summary="$5"
  local best_results="$6"
  local records_tsv="$7"
  local summary_json="${sweep_dir}/sweep_summary.json"
  local summary_tsv="${sweep_dir}/sweep_results.tsv"
  cp "${records_tsv}" "${summary_tsv}"
  PYTHON_BIN="${PYTHON_BIN}" \
  SWEEP_DIR_INPUT="${sweep_dir}" \
  BEST_LAYER_INPUT="${best_layer}" \
  BEST_ALPHA_INPUT="${best_alpha}" \
  BEST_SCORE_INPUT="${best_score}" \
  BEST_SUMMARY_INPUT="${best_summary}" \
  BEST_RESULTS_INPUT="${best_results}" \
  RECORDS_TSV_INPUT="${records_tsv}" \
  SUMMARY_JSON_INPUT="${summary_json}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

records_path = Path(os.environ["RECORDS_TSV_INPUT"])
rows = []
with records_path.open() as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows.extend(reader)

payload = {
    "best": {
        "layer": int(os.environ["BEST_LAYER_INPUT"]),
        "alpha": float(os.environ["BEST_ALPHA_INPUT"]),
        "overall_score": float(os.environ["BEST_SCORE_INPUT"]),
        "summary_path": os.environ["BEST_SUMMARY_INPUT"],
        "results_path": os.environ["BEST_RESULTS_INPUT"],
    },
    "records": [
        {
            "layer": int(row["layer"]),
            "alpha": float(row["alpha"]),
            "overall_score": float(row["overall_score"]),
            "summary_path": row["summary_path"],
            "results_path": row["results_path"],
        }
        for row in rows
    ],
}
Path(os.environ["SUMMARY_JSON_INPUT"]).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

run_toxicity_sweep() {
  local sweep_root="$1"
  local alpha_candidates="$2"
  local layer_candidates="$3"

  local records_tsv="${sweep_root}/records.tsv"
  mkdir -p "${sweep_root}"
  printf 'layer\talpha\toverall_score\tsummary_path\tresults_path\n' > "${records_tsv}"

  local best_score=""
  local best_layer=""
  local best_alpha=""
  local best_summary=""
  local best_results=""

  local layer
  local alpha
  for layer in ${layer_candidates}; do
    for alpha in ${alpha_candidates}; do
      local candidate_dir="${sweep_root}/layer_${layer}/alpha_$(normalize_tag "${alpha}")"
      local candidate_outputs="${candidate_dir}/outputs"
      local candidate_summary="${candidate_dir}/summary.json"
      local candidate_results="${candidate_outputs}/generations.csv"
      mkdir -p "${candidate_outputs}"

      echo "toxicity_sweep layer=${layer} alpha=${alpha} variant=${REPE_TOXICITY_VARIANT}"

      PYTHON_BIN="${PYTHON_BIN}" \
      TASK="${TASK}" \
      BASE_MODEL="${BASE_MODEL}" \
      REPE_VECTOR_DIR="${REPE_VECTOR_DIR}" \
      REPE_VECTOR_DIRS="${REPE_VECTOR_DIRS}" \
      REPE_LAYERS="${layer}" \
      REPE_ALPHA="${alpha}" \
      REPE_COMPOSITION="${REPE_COMPOSITION}" \
      REPE_WEIGHTS="${REPE_WEIGHTS}" \
      DEVICE="${DEVICE}" \
      TARGET_LAYERS="${TARGET_LAYERS}" \
      SUBSPACE_RANK="${SUBSPACE_RANK}" \
      SUMMARY_PATH="${candidate_summary}" \
      RESULTS_PATH="${candidate_results}" \
      REPE_TOXICITY_VARIANT="${REPE_TOXICITY_VARIANT}" \
      bash baseline/repe_pyvene_0/evaluate_repe_pyvene_task.sh

      local score
      score="$(extract_overall_score "${candidate_summary}")"
      printf '%s\t%s\t%s\t%s\t%s\n' "${layer}" "${alpha}" "${score}" "${candidate_summary}" "${candidate_results}" >> "${records_tsv}"

      if [[ -z "${best_score}" ]] || PYTHON_BIN="${PYTHON_BIN}" BEST_SCORE="${best_score}" CANDIDATE_SCORE="${score}" "${PYTHON_BIN}" - <<'PY'
import os
best = float(os.environ["BEST_SCORE"])
candidate = float(os.environ["CANDIDATE_SCORE"])
raise SystemExit(0 if candidate < best else 1)
PY
      then
        best_score="${score}"
        best_layer="${layer}"
        best_alpha="${alpha}"
        best_summary="${candidate_summary}"
        best_results="${candidate_results}"
      fi
    done
  done

  if [[ -z "${best_summary}" || -z "${best_results}" ]]; then
    echo "Toxicity sweep did not produce any valid candidate outputs." >&2
    exit 1
  fi

  cp "${best_summary}" "${SUMMARY_PATH}"
  cp "${best_results}" "${RESULTS_PATH}"
  write_toxicity_sweep_summary "${sweep_root}" "${best_layer}" "${best_alpha}" "${best_score}" "${best_summary}" "${best_results}" "${records_tsv}"

  echo "selected_toxicity_layer=${best_layer}"
  echo "selected_toxicity_alpha=${best_alpha}"
  echo "selected_toxicity_score=${best_score}"
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b}"
REPE_VECTOR_DIR="${REPE_VECTOR_DIR:-}"
REPE_VECTOR_DIRS="${REPE_VECTOR_DIRS:-}"
REPE_LAYERS="${REPE_LAYERS:-}"
REPE_COMPOSITION="${REPE_COMPOSITION:-single}"
REPE_WEIGHTS="${REPE_WEIGHTS:-}"
REPE_INTERVENE_ON_PROMPT="${REPE_INTERVENE_ON_PROMPT:-1}"
REPE_BASE_UNIT_LOCATION="${REPE_BASE_UNIT_LOCATION:--1}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
REPE_TOXICITY_VARIANT="${REPE_TOXICITY_VARIANT:-canonical}"
REPE_TOXICITY_AUTO_SWEEP="${REPE_TOXICITY_AUTO_SWEEP:-0}"
REPE_TOXICITY_LAYER_CANDIDATES="${REPE_TOXICITY_LAYER_CANDIDATES:-20 22 24 26 28 30 31}"
REPE_TOXICITY_ALPHA_CANDIDATES="${REPE_TOXICITY_ALPHA_CANDIDATES:-0.05 0.1 0.2 0.5}"
REPE_TOXICITY_OPTIMIZED_ALPHA_CANDIDATES="${REPE_TOXICITY_OPTIMIZED_ALPHA_CANDIDATES:-0.02 0.05 0.1 0.2}"
REPE_TOXICITY_DEFAULT_LAYER="${REPE_TOXICITY_DEFAULT_LAYER:-28}"
REPE_TOXICITY_DEFAULT_ALPHA_CANONICAL="${REPE_TOXICITY_DEFAULT_ALPHA_CANONICAL:-0.1}"
REPE_TOXICITY_DEFAULT_ALPHA_OPTIMIZED="${REPE_TOXICITY_DEFAULT_ALPHA_OPTIMIZED:-0.05}"

if [[ "${TASK}" == "toxicity" ]]; then
  if [[ -z "${REPE_LAYERS}" || "${REPE_LAYERS}" == "-1" ]]; then
    REPE_LAYERS="${REPE_TOXICITY_DEFAULT_LAYER}"
  fi
  if [[ -z "${REPE_ALPHA:-}" ]]; then
    if [[ "${REPE_TOXICITY_VARIANT}" == "optimized" ]]; then
      REPE_ALPHA="${REPE_TOXICITY_DEFAULT_ALPHA_OPTIMIZED}"
    else
      REPE_ALPHA="${REPE_TOXICITY_DEFAULT_ALPHA_CANONICAL}"
    fi
  fi
else
  REPE_ALPHA="${REPE_ALPHA:-1.0}"
fi

REPE_ALPHA="${REPE_ALPHA:-1.0}"

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
RUN_NAME="$(derive_eval_name)"
if [[ "${TASK}" == "ethics" && ( -z "${REPE_LAYERS}" || "${REPE_LAYERS}" == "-1" ) && -n "${REPE_VECTOR_DIR}" ]]; then
  REPE_LAYERS="$(resolve_ethics_layers_from_vector "${REPE_VECTOR_DIR}")"
fi

PYVENE_INTERVENE_ON_PROMPT=${REPE_INTERVENE_ON_PROMPT}
PYVENE_BASE_UNIT_LOCATION=${REPE_BASE_UNIT_LOCATION}

RUN_DIR="baseline/repe_pyvene_0/runs/eval/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
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
REPE_VECTOR_DIR=${REPE_VECTOR_DIR}
REPE_VECTOR_DIRS=${REPE_VECTOR_DIRS}
REPE_LAYERS=${REPE_LAYERS}
REPE_ALPHA=${REPE_ALPHA}
REPE_COMPOSITION=${REPE_COMPOSITION}
REPE_WEIGHTS=${REPE_WEIGHTS}
PYVENE_METHOD_TAG=repe-pyvene
PYVENE_COMPONENT=block_output
PYVENE_INTERVENE_ON_PROMPT=${PYVENE_INTERVENE_ON_PROMPT}
PYVENE_BASE_UNIT_LOCATION=${PYVENE_BASE_UNIT_LOCATION}
REPE_TOXICITY_VARIANT=${REPE_TOXICITY_VARIANT}
REPE_TOXICITY_AUTO_SWEEP=${REPE_TOXICITY_AUTO_SWEEP}
REPE_TOXICITY_LAYER_CANDIDATES=${REPE_TOXICITY_LAYER_CANDIDATES}
REPE_TOXICITY_ALPHA_CANDIDATES=${REPE_TOXICITY_ALPHA_CANDIDATES}
REPE_TOXICITY_OPTIMIZED_ALPHA_CANDIDATES=${REPE_TOXICITY_OPTIMIZED_ALPHA_CANDIDATES}
REPE_TOXICITY_DEFAULT_LAYER=${REPE_TOXICITY_DEFAULT_LAYER}
REPE_TOXICITY_DEFAULT_ALPHA_CANONICAL=${REPE_TOXICITY_DEFAULT_ALPHA_CANONICAL}
REPE_TOXICITY_DEFAULT_ALPHA_OPTIMIZED=${REPE_TOXICITY_DEFAULT_ALPHA_OPTIMIZED}
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
echo "repe_vector_dir=${REPE_VECTOR_DIR}"
echo "repe_vector_dirs=${REPE_VECTOR_DIRS}"
echo "repe_layers=${REPE_LAYERS}"
echo "repe_alpha=${REPE_ALPHA}"
echo "repe_composition=${REPE_COMPOSITION}"
echo "pyvene_intervene_on_prompt=${PYVENE_INTERVENE_ON_PROMPT}"
echo "pyvene_base_unit_location=${PYVENE_BASE_UNIT_LOCATION}"
echo "repe_toxicity_variant=${REPE_TOXICITY_VARIANT}"
echo "repe_toxicity_auto_sweep=${REPE_TOXICITY_AUTO_SWEEP}"
echo "output_dir=${OUTPUT_DIR}"

if [[ "${TASK}" == "toxicity" && "${REPE_TOXICITY_AUTO_SWEEP}" == "1" ]]; then
  if [[ "${REPE_TOXICITY_VARIANT}" == "optimized" ]]; then
    SWEEP_ALPHA_CANDIDATES="${REPE_TOXICITY_OPTIMIZED_ALPHA_CANDIDATES}"
  else
    SWEEP_ALPHA_CANDIDATES="${REPE_TOXICITY_ALPHA_CANDIDATES}"
  fi
  SWEEP_DIR="${RUN_DIR}/sweep/${REPE_TOXICITY_VARIANT}"
  run_toxicity_sweep "${SWEEP_DIR}" "${SWEEP_ALPHA_CANDIDATES}" "${REPE_TOXICITY_LAYER_CANDIDATES}"
else
  PYTHON_BIN="${PYTHON_BIN}" \
  TASK="${TASK}" \
  BASE_MODEL="${BASE_MODEL}" \
  REPE_VECTOR_DIR="${REPE_VECTOR_DIR}" \
  REPE_VECTOR_DIRS="${REPE_VECTOR_DIRS}" \
  REPE_LAYERS="${REPE_LAYERS}" \
  REPE_ALPHA="${REPE_ALPHA}" \
  REPE_COMPOSITION="${REPE_COMPOSITION}" \
  REPE_WEIGHTS="${REPE_WEIGHTS}" \
  REPE_INTERVENE_ON_PROMPT="${PYVENE_INTERVENE_ON_PROMPT}" \
  REPE_BASE_UNIT_LOCATION="${PYVENE_BASE_UNIT_LOCATION}" \
  DEVICE="${DEVICE}" \
  TARGET_LAYERS="${TARGET_LAYERS}" \
  SUBSPACE_RANK="${SUBSPACE_RANK}" \
  SUMMARY_PATH="${SUMMARY_PATH}" \
  RESULTS_PATH="${RESULTS_PATH}" \
  REPE_TOXICITY_VARIANT="${REPE_TOXICITY_VARIANT}" \
  bash baseline/repe_pyvene_0/evaluate_repe_pyvene_task.sh
fi

echo "summary_file=${SUMMARY_PATH}"
