#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

resolve_default_eval_layer() {
  local task_name="$1"
  local base_model_path="$2"
  PYTHON_BIN="${PYTHON_BIN}" TASK_NAME="${task_name}" BASE_MODEL_PATH="${base_model_path}" "${PYTHON_BIN}" - <<'PY'
import os
from baseline.CAA_0.task_registry import resolve_default_single_layer, should_use_default_single_layer

task_name = os.environ["TASK_NAME"]
base_model_path = os.environ["BASE_MODEL_PATH"]

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
  local base_name
  if [[ -n "${EVAL_NAME:-}" ]]; then
    base_name="$(normalize_tag "${EVAL_NAME}")"
  elif [[ -n "${CAA_VECTOR_DIR:-}" ]]; then
    local vector_base
    vector_base="$(basename "${CAA_VECTOR_DIR%/}")"
    base_name="single-$(normalize_tag "${vector_base}")"
  else
    local composition_tag
    composition_tag="$(normalize_tag "${CAA_COMPOSITION:-mean}")"
    base_name="naive-${composition_tag}"
  fi

  if [[ "${AUTO_SELECT_LAYER:-0}" != "0" ]]; then
    echo "autolayer-${base_name}"
  else
    echo "${base_name}"
  fi
}

infer_layer_candidates() {
  if [[ -n "${CAA_LAYER_CANDIDATES:-}" ]]; then
    echo "${CAA_LAYER_CANDIDATES}"
    return 0
  fi

  CAA_VECTOR_DIR="${CAA_VECTOR_DIR}" CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS}" PYTHON_BIN="${PYTHON_BIN}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from baseline.CAA_0.steering import infer_layers_from_vector_dirs, resolve_caa_vector_dirs

single = os.environ.get("CAA_VECTOR_DIR", "").strip() or None
multi = [item for item in os.environ.get("CAA_VECTOR_DIRS", "").split() if item.strip()]
dirs = resolve_caa_vector_dirs(vector_dir=single, vector_dirs=multi)
layers = infer_layers_from_vector_dirs(dirs)
print(" ".join(str(layer) for layer in layers))
PY
}

extract_score_from_summary() {
  local summary_path="$1"
  local task_name="$2"
  "${PYTHON_BIN}" - <<'PY' "${summary_path}" "${task_name}"
import json
import sys

summary_path = sys.argv[1]
task_name = sys.argv[2]
with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if task_name in {"truth", "bias"}:
    score = data.get("accuracy")
else:
    score = data.get("overall_score")

if score is None:
    raise SystemExit(2)
print(float(score))
PY
}

normalize_layer_candidates() {
  "${PYTHON_BIN}" - <<'PY' "$@"
import sys

values = []
for arg in sys.argv[1:]:
    for part in str(arg).split():
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
values = sorted(set(values))
print(" ".join(str(v) for v in values))
PY
}

select_coarse_candidates() {
  local full_candidates_str="$1"
  local explicit_coarse_str="$2"
  local coarse_points="$3"
  "${PYTHON_BIN}" - <<'PY' "${full_candidates_str}" "${explicit_coarse_str}" "${coarse_points}"
import sys

full = [int(x) for x in sys.argv[1].split() if x.strip()]
explicit = [int(x) for x in sys.argv[2].split() if x.strip()]
coarse_points = int(sys.argv[3])

if not full:
    print("")
    raise SystemExit(0)

if explicit:
    full_set = set(full)
    coarse = []
    seen = set()
    for value in explicit:
        if value in full_set and value not in seen:
            coarse.append(value)
            seen.add(value)
else:
    if coarse_points <= 0 or coarse_points >= len(full):
        coarse = list(full)
    else:
        coarse = []
        seen = set()
        n = len(full)
        for i in range(coarse_points):
            idx = int((i + 0.5) * n / coarse_points)
            if idx >= n:
                idx = n - 1
            value = full[idx]
            if value not in seen:
                coarse.append(value)
                seen.add(value)
        if len(coarse) < coarse_points:
            for value in full:
                if value not in seen:
                    coarse.append(value)
                    seen.add(value)
                if len(coarse) >= min(coarse_points, len(full)):
                    break

print(" ".join(str(v) for v in coarse))
PY
}

select_fine_candidates() {
  local full_candidates_str="$1"
  local evaluated_candidates_str="$2"
  local best_layer="$3"
  local fine_radius="$4"
  "${PYTHON_BIN}" - <<'PY' "${full_candidates_str}" "${evaluated_candidates_str}" "${best_layer}" "${fine_radius}"
import sys

full = [int(x) for x in sys.argv[1].split() if x.strip()]
evaluated = {int(x) for x in sys.argv[2].split() if x.strip()}
best_layer = int(sys.argv[3])
fine_radius = int(sys.argv[4])

if not full or best_layer not in full or fine_radius <= 0:
    print("")
    raise SystemExit(0)

best_idx = full.index(best_layer)
left = max(0, best_idx - fine_radius)
right = min(len(full), best_idx + fine_radius + 1)
fine = [value for value in full[left:right] if value not in evaluated]
print(" ".join(str(v) for v in fine))
PY
}

write_layer_selection_summary() {
  local records_tsv="$1"
  local out_path="$2"
  local task_name="$3"
  local full_candidates_str="$4"
  local coarse_candidates_str="$5"
  local fine_candidates_str="$6"
  local best_coarse_layer="$7"
  local best_coarse_score="$8"
  local best_layer="$9"
  local best_score="${10}"
  local fine_radius="${11}"
  local coarse_points="${12}"
  "${PYTHON_BIN}" - <<'PY' "${records_tsv}" "${out_path}" "${task_name}" "${full_candidates_str}" "${coarse_candidates_str}" "${fine_candidates_str}" "${best_coarse_layer}" "${best_coarse_score}" "${best_layer}" "${best_score}" "${fine_radius}" "${coarse_points}"
import json
import sys

(
    tsv_path,
    out_path,
    task_name,
    full_candidates_str,
    coarse_candidates_str,
    fine_candidates_str,
    best_coarse_layer,
    best_coarse_score,
    best_layer,
    best_score,
    fine_radius,
    coarse_points,
) = sys.argv[1:13]

def parse_list(value):
    return [int(x) for x in value.split() if x.strip()]

full_candidates = parse_list(full_candidates_str)
coarse_candidates = parse_list(coarse_candidates_str)
fine_candidates = parse_list(fine_candidates_str)
records = []
coarse_records = []
fine_records = []

with open(tsv_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 6:
            continue
        stage, layer, status, score, summary_path, results_path = parts
        record = {
            "stage": stage,
            "layer": int(layer),
            "status": status,
            "score": None if not score else float(score),
            "summary_path": summary_path,
            "results_path": results_path,
        }
        records.append(record)
        if stage == "coarse":
            coarse_records.append(record)
        elif stage == "fine":
            fine_records.append(record)

payload = {
    "task": task_name,
    "sweep_mode": "two_stage",
    "full_candidate_layers": full_candidates,
    "coarse_stage": {
        "requested_points": int(coarse_points),
        "candidate_layers": coarse_candidates,
        "best_layer": None if not best_coarse_layer else int(best_coarse_layer),
        "best_score": None if not best_coarse_score else float(best_coarse_score),
        "records": coarse_records,
    },
    "fine_stage": {
        "radius": int(fine_radius),
        "candidate_layers": fine_candidates,
        "best_layer": None if not best_layer else int(best_layer),
        "best_score": None if not best_score else float(best_score),
        "records": fine_records,
    },
    "best_layer": None if not best_layer else int(best_layer),
    "best_score": None if not best_score else float(best_score),
    "records": records,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY
}

resolve_selection_truth_dataset() {
  if [[ -n "${LAYER_SELECT_TRUTH_DATASET:-}" ]]; then
    echo "${LAYER_SELECT_TRUTH_DATASET}"
    return 0
  fi
  if [[ -f "dataset/truthfulqa_mc_dev/test.json" ]]; then
    echo "truthfulqa_mc_dev"
    return 0
  fi
  echo "${TRUTH_DATASET:-${DATASET:-truthfulqa_mc}}"
}

run_eval_phase() {
  local phase="$1"
  local phase_summary_path="$2"
  local phase_results_path="$3"
  local phase_layers="$4"

  local truth_dataset_env="${TRUTH_DATASET:-}"
  local bias_dataset_env="${BIAS_DATASET:-}"
  local ethics_dataset_file_env="${ETHICS_DATASET_FILE:-}"
  local toxicity_dataset_root_env="${TOXICITY_DATASET_ROOT:-}"
  local toxicity_datasets_env="${TOXICITY_DATASETS:-}"
  local toxicity_prompts_env="${TOXICITY_PROMPTS:-}"

  if [[ "${phase}" == "selection" ]]; then
    truth_dataset_env="${LAYER_SELECT_TRUTH_DATASET_EFFECTIVE}"
    bias_dataset_env="${LAYER_SELECT_BIAS_DATASET_EFFECTIVE}"
    ethics_dataset_file_env="${LAYER_SELECT_ETHICS_DATASET_FILE_EFFECTIVE}"
    toxicity_dataset_root_env="${LAYER_SELECT_TOXICITY_DATASET_ROOT_EFFECTIVE}"
    toxicity_datasets_env="${LAYER_SELECT_TOXICITY_DATASETS_EFFECTIVE}"
    toxicity_prompts_env="${LAYER_SELECT_TOXICITY_PROMPTS_EFFECTIVE}"
  fi

  PYTHON_BIN="${PYTHON_BIN}" \
  TASK="${TASK}" \
  BASE_MODEL="${BASE_MODEL}" \
  CAA_VECTOR_DIR="${CAA_VECTOR_DIR}" \
  CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS}" \
  CAA_LAYERS="${phase_layers}" \
  CAA_ALPHA="${CAA_ALPHA}" \
  CAA_TOKEN_STRATEGY="${CAA_TOKEN_STRATEGY}" \
  CAA_COMPOSITION="${CAA_COMPOSITION}" \
  CAA_WEIGHTS="${CAA_WEIGHTS}" \
  DEVICE="${DEVICE}" \
  TARGET_LAYERS="${TARGET_LAYERS}" \
  SUBSPACE_RANK="${SUBSPACE_RANK}" \
  SUMMARY_PATH="${phase_summary_path}" \
  RESULTS_PATH="${phase_results_path}" \
  TRUTH_DATASET="${truth_dataset_env}" \
  BIAS_DATASET="${bias_dataset_env}" \
  ETHICS_DATASET_FILE="${ethics_dataset_file_env}" \
  TOXICITY_DATASET_ROOT="${toxicity_dataset_root_env}" \
  TOXICITY_DATASETS="${toxicity_datasets_env}" \
  TOXICITY_PROMPTS="${toxicity_prompts_env}" \
  bash baseline/CAA_0/evaluate_caa_task.sh
}

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
CAA_VECTOR_DIR="${CAA_VECTOR_DIR:-}"
CAA_VECTOR_DIRS="${CAA_VECTOR_DIRS:-}"
RAW_CAA_LAYERS="${CAA_LAYERS-}"
CAA_ALPHA="${CAA_ALPHA:-1.0}"
CAA_TOKEN_STRATEGY="${CAA_TOKEN_STRATEGY:-last}"
CAA_COMPOSITION="${CAA_COMPOSITION:-single}"
CAA_WEIGHTS="${CAA_WEIGHTS:-}"
AUTO_SELECT_LAYER="${AUTO_SELECT_LAYER:-0}"
CAA_LAYER_CANDIDATES="${CAA_LAYER_CANDIDATES:-}"
CAA_COARSE_LAYER_CANDIDATES="${CAA_COARSE_LAYER_CANDIDATES:-}"
CAA_COARSE_SWEEP_POINTS="${CAA_COARSE_SWEEP_POINTS:-6}"
CAA_FINE_SWEEP_RADIUS="${CAA_FINE_SWEEP_RADIUS:-2}"
LAYER_SELECT_TRUTH_DATASET="${LAYER_SELECT_TRUTH_DATASET:-}"
LAYER_SELECT_BIAS_DATASET="${LAYER_SELECT_BIAS_DATASET:-}"
LAYER_SELECT_ETHICS_DATASET_FILE="${LAYER_SELECT_ETHICS_DATASET_FILE:-}"
LAYER_SELECT_TOXICITY_DATASET_ROOT="${LAYER_SELECT_TOXICITY_DATASET_ROOT:-}"
LAYER_SELECT_TOXICITY_DATASETS="${LAYER_SELECT_TOXICITY_DATASETS:-}"
LAYER_SELECT_TOXICITY_PROMPTS="${LAYER_SELECT_TOXICITY_PROMPTS:-}"
DEVICE="${DEVICE:-cuda:0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"
MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
RUN_NAME="$(derive_eval_name)"
if [[ -n "${RAW_CAA_LAYERS}" ]]; then
  CAA_LAYERS="${RAW_CAA_LAYERS}"
elif [[ "${AUTO_SELECT_LAYER}" == "0" ]]; then
  CAA_LAYERS="$(resolve_default_eval_layer "${TASK}" "${BASE_MODEL}")"
else
  CAA_LAYERS=""
fi
LAYER_SELECT_TRUTH_DATASET_EFFECTIVE="$(resolve_selection_truth_dataset)"
LAYER_SELECT_BIAS_DATASET_EFFECTIVE="${LAYER_SELECT_BIAS_DATASET:-${BIAS_DATASET:-${DATASET:-bbq}}}"
LAYER_SELECT_ETHICS_DATASET_FILE_EFFECTIVE="${LAYER_SELECT_ETHICS_DATASET_FILE:-${ETHICS_DATASET_FILE:-multi_train/eval_ethics/data/ethics/cm_test.csv}}"
LAYER_SELECT_TOXICITY_DATASET_ROOT_EFFECTIVE="${LAYER_SELECT_TOXICITY_DATASET_ROOT:-${TOXICITY_DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}}"
LAYER_SELECT_TOXICITY_DATASETS_EFFECTIVE="${LAYER_SELECT_TOXICITY_DATASETS:-${TOXICITY_DATASETS:-toxic nontoxic}}"
LAYER_SELECT_TOXICITY_PROMPTS_EFFECTIVE="${LAYER_SELECT_TOXICITY_PROMPTS:-${TOXICITY_PROMPTS:-benign adversarial}}"

RUN_DIR="baseline/CAA_0/runs/eval/${MODEL_NAME}/${TASK}/${RUN_NAME}/${RUN_TS}"
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
CAA_TOKEN_STRATEGY=${CAA_TOKEN_STRATEGY}
CAA_COMPOSITION=${CAA_COMPOSITION}
CAA_WEIGHTS=${CAA_WEIGHTS}
AUTO_SELECT_LAYER=${AUTO_SELECT_LAYER}
CAA_LAYER_CANDIDATES=${CAA_LAYER_CANDIDATES}
CAA_COARSE_LAYER_CANDIDATES=${CAA_COARSE_LAYER_CANDIDATES}
CAA_COARSE_SWEEP_POINTS=${CAA_COARSE_SWEEP_POINTS}
CAA_FINE_SWEEP_RADIUS=${CAA_FINE_SWEEP_RADIUS}
LAYER_SELECT_TRUTH_DATASET=${LAYER_SELECT_TRUTH_DATASET}
LAYER_SELECT_BIAS_DATASET=${LAYER_SELECT_BIAS_DATASET}
LAYER_SELECT_ETHICS_DATASET_FILE=${LAYER_SELECT_ETHICS_DATASET_FILE}
LAYER_SELECT_TOXICITY_DATASET_ROOT=${LAYER_SELECT_TOXICITY_DATASET_ROOT}
LAYER_SELECT_TOXICITY_DATASETS=${LAYER_SELECT_TOXICITY_DATASETS}
LAYER_SELECT_TOXICITY_PROMPTS=${LAYER_SELECT_TOXICITY_PROMPTS}
LAYER_SELECT_TRUTH_DATASET_EFFECTIVE=${LAYER_SELECT_TRUTH_DATASET_EFFECTIVE}
LAYER_SELECT_BIAS_DATASET_EFFECTIVE=${LAYER_SELECT_BIAS_DATASET_EFFECTIVE}
LAYER_SELECT_ETHICS_DATASET_FILE_EFFECTIVE=${LAYER_SELECT_ETHICS_DATASET_FILE_EFFECTIVE}
LAYER_SELECT_TOXICITY_DATASET_ROOT_EFFECTIVE=${LAYER_SELECT_TOXICITY_DATASET_ROOT_EFFECTIVE}
LAYER_SELECT_TOXICITY_DATASETS_EFFECTIVE=${LAYER_SELECT_TOXICITY_DATASETS_EFFECTIVE}
LAYER_SELECT_TOXICITY_PROMPTS_EFFECTIVE=${LAYER_SELECT_TOXICITY_PROMPTS_EFFECTIVE}
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
echo "caa_layers_initial=${CAA_LAYERS}"
echo "caa_composition=${CAA_COMPOSITION}"
echo "auto_select_layer=${AUTO_SELECT_LAYER}"
echo "caa_layer_candidates=${CAA_LAYER_CANDIDATES}"
echo "caa_coarse_layer_candidates=${CAA_COARSE_LAYER_CANDIDATES}"
echo "caa_coarse_sweep_points=${CAA_COARSE_SWEEP_POINTS}"
echo "caa_fine_sweep_radius=${CAA_FINE_SWEEP_RADIUS}"
echo "layer_select_truth_dataset_effective=${LAYER_SELECT_TRUTH_DATASET_EFFECTIVE}"
echo "layer_select_bias_dataset_effective=${LAYER_SELECT_BIAS_DATASET_EFFECTIVE}"
echo "layer_select_ethics_dataset_file_effective=${LAYER_SELECT_ETHICS_DATASET_FILE_EFFECTIVE}"
echo "layer_select_toxicity_dataset_root_effective=${LAYER_SELECT_TOXICITY_DATASET_ROOT_EFFECTIVE}"
echo "layer_select_toxicity_datasets_effective=${LAYER_SELECT_TOXICITY_DATASETS_EFFECTIVE}"
echo "layer_select_toxicity_prompts_effective=${LAYER_SELECT_TOXICITY_PROMPTS_EFFECTIVE}"
echo "output_dir=${OUTPUT_DIR}"

if [[ "${AUTO_SELECT_LAYER}" != "0" ]]; then
  SELECT_DIR="${RUN_DIR}/layer_selection"
  mkdir -p "${SELECT_DIR}"
  CANDIDATES_STR="$(infer_layer_candidates)"
  if [[ -z "${CANDIDATES_STR}" ]]; then
    echo "Failed to infer CAA layer candidates." >&2
    exit 1
  fi
  CANDIDATES_STR="$(normalize_layer_candidates "${CANDIDATES_STR}")"
  read -r -a CANDIDATE_LAYER_ARR <<< "${CANDIDATES_STR}"
  if [[ "${#CANDIDATE_LAYER_ARR[@]}" -eq 0 ]]; then
    echo "No CAA layer candidates found." >&2
    exit 1
  fi

  echo "layer_selection_full_candidates=${CANDIDATES_STR}"
  COARSE_CANDIDATES_STR="$(select_coarse_candidates "${CANDIDATES_STR}" "${CAA_COARSE_LAYER_CANDIDATES}" "${CAA_COARSE_SWEEP_POINTS}")"
  if [[ -z "${COARSE_CANDIDATES_STR}" ]]; then
    echo "Failed to build coarse layer candidates." >&2
    exit 1
  fi
  read -r -a COARSE_LAYER_ARR <<< "${COARSE_CANDIDATES_STR}"
  if [[ "${#COARSE_LAYER_ARR[@]}" -eq 0 ]]; then
    echo "Coarse layer candidate set is empty." >&2
    exit 1
  fi
  echo "layer_selection_coarse_candidates=${COARSE_CANDIDATES_STR}"

  BEST_COARSE_LAYER=""
  BEST_COARSE_SCORE=""
  BEST_LAYER=""
  BEST_SCORE=""
  CANDIDATE_RECORDS_TSV="${SELECT_DIR}/candidate_scores.tsv"
  : > "${CANDIDATE_RECORDS_TSV}"

  for LAYER in "${COARSE_LAYER_ARR[@]}"; do
    CANDIDATE_DIR="${SELECT_DIR}/coarse/layer_${LAYER}"
    mkdir -p "${CANDIDATE_DIR}/outputs"
    if [[ "${TASK}" == "truth" || "${TASK}" == "bias" ]]; then
      CANDIDATE_RESULTS_PATH="${CANDIDATE_DIR}/outputs/results.json"
    else
      CANDIDATE_RESULTS_PATH="${CANDIDATE_DIR}/outputs/generations.csv"
    fi
    CANDIDATE_SUMMARY_PATH="${CANDIDATE_DIR}/summary.json"

    echo "coarse_selecting_layer=${LAYER}"
    if run_eval_phase "selection" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" "${LAYER}"; then
      SCORE="$(extract_score_from_summary "${CANDIDATE_SUMMARY_PATH}" "${TASK}")"
      printf "coarse\t%s\tok\t%s\t%s\t%s\n" "${LAYER}" "${SCORE}" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" >> "${CANDIDATE_RECORDS_TSV}"
      if [[ -z "${BEST_COARSE_SCORE}" ]] || "${PYTHON_BIN}" - <<'PY' "${SCORE}" "${BEST_COARSE_SCORE}"
import sys
candidate = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if candidate > best else 1)
PY
      then
        BEST_COARSE_LAYER="${LAYER}"
        BEST_COARSE_SCORE="${SCORE}"
      fi
    else
      printf "coarse\t%s\tfailed\t\t%s\t%s\n" "${LAYER}" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" >> "${CANDIDATE_RECORDS_TSV}"
    fi
  done

  if [[ -z "${BEST_COARSE_LAYER}" ]]; then
    echo "Automatic layer selection failed in coarse stage: no successful candidate." >&2
    exit 1
  fi

  FINE_CANDIDATES_STR="$(select_fine_candidates "${CANDIDATES_STR}" "${COARSE_CANDIDATES_STR}" "${BEST_COARSE_LAYER}" "${CAA_FINE_SWEEP_RADIUS}")"
  echo "layer_selection_fine_candidates=${FINE_CANDIDATES_STR}"

  BEST_LAYER="${BEST_COARSE_LAYER}"
  BEST_SCORE="${BEST_COARSE_SCORE}"

  if [[ -n "${FINE_CANDIDATES_STR}" ]]; then
    read -r -a FINE_LAYER_ARR <<< "${FINE_CANDIDATES_STR}"
    for LAYER in "${FINE_LAYER_ARR[@]}"; do
      CANDIDATE_DIR="${SELECT_DIR}/fine/layer_${LAYER}"
      mkdir -p "${CANDIDATE_DIR}/outputs"
      if [[ "${TASK}" == "truth" || "${TASK}" == "bias" ]]; then
        CANDIDATE_RESULTS_PATH="${CANDIDATE_DIR}/outputs/results.json"
      else
        CANDIDATE_RESULTS_PATH="${CANDIDATE_DIR}/outputs/generations.csv"
      fi
      CANDIDATE_SUMMARY_PATH="${CANDIDATE_DIR}/summary.json"

      echo "fine_selecting_layer=${LAYER}"
      if run_eval_phase "selection" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" "${LAYER}"; then
        SCORE="$(extract_score_from_summary "${CANDIDATE_SUMMARY_PATH}" "${TASK}")"
        printf "fine\t%s\tok\t%s\t%s\t%s\n" "${LAYER}" "${SCORE}" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" >> "${CANDIDATE_RECORDS_TSV}"
        if [[ -z "${BEST_SCORE}" ]] || "${PYTHON_BIN}" - <<'PY' "${SCORE}" "${BEST_SCORE}"
import sys
candidate = float(sys.argv[1])
best = float(sys.argv[2])
sys.exit(0 if candidate > best else 1)
PY
        then
          BEST_LAYER="${LAYER}"
          BEST_SCORE="${SCORE}"
        fi
      else
        printf "fine\t%s\tfailed\t\t%s\t%s\n" "${LAYER}" "${CANDIDATE_SUMMARY_PATH}" "${CANDIDATE_RESULTS_PATH}" >> "${CANDIDATE_RECORDS_TSV}"
      fi
    done
  fi

  SELECT_SUMMARY_PATH="${RUN_DIR}/layer_selection_summary.json"
  write_layer_selection_summary \
    "${CANDIDATE_RECORDS_TSV}" \
    "${SELECT_SUMMARY_PATH}" \
    "${TASK}" \
    "${CANDIDATES_STR}" \
    "${COARSE_CANDIDATES_STR}" \
    "${FINE_CANDIDATES_STR}" \
    "${BEST_COARSE_LAYER}" \
    "${BEST_COARSE_SCORE}" \
    "${BEST_LAYER}" \
    "${BEST_SCORE}" \
    "${CAA_FINE_SWEEP_RADIUS}" \
    "${CAA_COARSE_SWEEP_POINTS}"

  echo "selected_best_coarse_layer=${BEST_COARSE_LAYER}"
  echo "selected_best_coarse_score=${BEST_COARSE_SCORE}"
  echo "selected_best_layer=${BEST_LAYER}"
  echo "selected_best_score=${BEST_SCORE}"
  echo "layer_selection_summary=${SELECT_SUMMARY_PATH}"
  echo "SELECTION_PHASE=validation" >> "${MANIFEST_FILE}"
  echo "SELECTED_COARSE_LAYER=${BEST_COARSE_LAYER}" >> "${MANIFEST_FILE}"
  echo "SELECTED_COARSE_LAYER_SCORE=${BEST_COARSE_SCORE}" >> "${MANIFEST_FILE}"
  echo "SELECTED_LAYER=${BEST_LAYER}" >> "${MANIFEST_FILE}"
  echo "SELECTED_LAYER_SCORE=${BEST_SCORE}" >> "${MANIFEST_FILE}"
  echo "LAYER_SELECTION_SUMMARY=${SELECT_SUMMARY_PATH}" >> "${MANIFEST_FILE}"
  CAA_LAYERS="${BEST_LAYER}"
fi

run_eval_phase "final" "${SUMMARY_PATH}" "${RESULTS_PATH}" "${CAA_LAYERS}"
