#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK_SCRIPT="${SCRIPT_DIR}/evaluate_composable_e6_task.sh"

format_tag() {
  local value="${1:-unknown}"
  value="${value//-/m}"
  value="${value//./p}"
  value="${value// /}"
  echo "${value}"
}

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
SPEC1="${SPEC1:-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

SCORE_STATS_PATH="${SCORE_STATS_PATH:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"
SCORE_SOURCE="${SCORE_SOURCE:-intervention_norm}"
SCORE_NORMALIZER="${SCORE_NORMALIZER:-log_zscore}"
COMPOSITION_METHOD="${COMPOSITION_METHOD:-compat_filtered_topk}"
COMPOSE_DOMAIN="${COMPOSE_DOMAIN:-output}"
COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE:-1.0}"
COMPOSITION_TOPK="${COMPOSITION_TOPK:-2}"
COMPAT_THRESHOLD="${COMPAT_THRESHOLD:-0.0}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE:-1}"
E6_PARAM_TAG="${E6_PARAM_TAG:-temp$(format_tag "${COMPOSITION_TEMPERATURE}")-topk$(format_tag "${COMPOSITION_TOPK}")}"

TASKS="${TASKS:-truth bias ethics toxicity}"
GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3}"
STAGGER_SECONDS="${STAGGER_SECONDS:-10}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-multi_train/logs/composable_e6_${E6_PARAM_TAG}_${RUN_TS}}"
SUMMARY_DIR="${SUMMARY_DIR:-multi_train/eval_router/summaries}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-}"

mkdir -p "${LOG_DIR}"
mkdir -p "${SUMMARY_DIR}"

read -r -a TASK_ARR <<< "${TASKS}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"

if [[ "${#TASK_ARR[@]}" -eq 0 ]]; then
  echo "TASKS must contain at least 1 task. Current: ${TASKS}" >&2
  exit 1
fi

if [[ "${#GPU_IDS_ARR[@]}" -lt "${#TASK_ARR[@]}" ]]; then
  echo "Need at least ${#TASK_ARR[@]} GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi

PIDS=()
TASK_NAMES_RUN=()
SUMMARY_PATHS_RUN=()

build_summary_prefix() {
  local user_prefix="$1"
  local base_model="$2"
  local exp_suffix="$3"

  python - "$user_prefix" "$base_model" "$exp_suffix" <<'PY'
import sys
from multi_train.eval_common.output_naming import resolve_output_prefix

user_prefix = sys.argv[1] if len(sys.argv) > 1 else ""
base_model = sys.argv[2] if len(sys.argv) > 2 else ""
exp_suffix = sys.argv[3] if len(sys.argv) > 3 else ""
prefix = resolve_output_prefix(user_prefix, base_model, "composable-e6")
if not user_prefix and exp_suffix:
    prefix = f"{prefix}-{exp_suffix}"
print(prefix)
PY
}

build_single_summary_path_with_prefix() {
  local summary_dir="$1"
  local prefix="$2"
  local task_name="$3"
  local run_ts="$4"

  python - "$summary_dir" "$prefix" "$task_name" "$run_ts" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_single_summary_path

summary_dir = sys.argv[1]
prefix = sys.argv[2]
task_name = sys.argv[3]
run_ts = sys.argv[4]
print(build_single_summary_path(summary_dir, prefix, task_name, run_ts, None))
PY
}

build_merge_summary_path_with_prefix() {
  local summary_dir="$1"
  local prefix="$2"
  local run_ts="$3"

  python - "$summary_dir" "$prefix" "$run_ts" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_merge_summary_path

summary_dir = sys.argv[1]
prefix = sys.argv[2]
run_ts = sys.argv[3]
print(build_merge_summary_path(summary_dir, prefix, run_ts))
PY
}

SUMMARY_PREFIX="$(build_summary_prefix "${MERGE_SUMMARY_PATH}" "${BASE_MODEL}" "${E6_PARAM_TAG}")"
MERGE_SUMMARY_FILE="$(build_merge_summary_path_with_prefix "${SUMMARY_DIR}" "${SUMMARY_PREFIX}" "${RUN_TS}")"

cleanup_children() {
  for pid in "${PIDS[@]-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup_children INT TERM

run_task() {
  local task_name="$1"
  local gpu_id="$2"
  local log_file="${LOG_DIR}/${task_name}.log"
  local summary_path
  summary_path="$(build_single_summary_path_with_prefix "${SUMMARY_DIR}" "${SUMMARY_PREFIX}" "${task_name}" "${RUN_TS}")"

  echo "[${task_name}] gpu=${gpu_id} log=${log_file} summary=${summary_path}"
  (
    echo "===== ${task_name} ====="
    echo "TASK=${task_name}"
    echo "GPU=cuda:${gpu_id}"
    echo "PYTHON_BIN=${PYTHON_BIN}"
    echo "BASE_MODEL=${BASE_MODEL}"
    echo "SPEC1=${SPEC1}"
    echo "SPEC2=${SPEC2}"
    echo "SPEC3=${SPEC3}"
    echo "SPEC4=${SPEC4}"
    echo "SCORE_STATS_PATH=${SCORE_STATS_PATH}"
    echo "SCORE_SOURCE=${SCORE_SOURCE}"
    echo "SCORE_NORMALIZER=${SCORE_NORMALIZER}"
    echo "COMPOSITION_METHOD=${COMPOSITION_METHOD}"
    echo "COMPOSE_DOMAIN=${COMPOSE_DOMAIN}"
    echo "COMPOSITION_TEMPERATURE=${COMPOSITION_TEMPERATURE}"
    echo "COMPOSITION_TOPK=${COMPOSITION_TOPK}"
    echo "COMPAT_THRESHOLD=${COMPAT_THRESHOLD}"
    echo "TARGET_LAYERS=${TARGET_LAYERS}"
    echo "ENABLE_DEBUG_CACHE=${ENABLE_DEBUG_CACHE}"
    echo "SUMMARY_PATH=${summary_path}"
    echo
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      TASK="${task_name}" \
      BASE_MODEL="${BASE_MODEL}" \
      SPEC1="${SPEC1}" \
      SPEC2="${SPEC2}" \
      SPEC3="${SPEC3}" \
      SPEC4="${SPEC4}" \
      SCORE_STATS_PATH="${SCORE_STATS_PATH}" \
      SCORE_SOURCE="${SCORE_SOURCE}" \
      SCORE_NORMALIZER="${SCORE_NORMALIZER}" \
      COMPOSITION_METHOD="${COMPOSITION_METHOD}" \
      COMPOSE_DOMAIN="${COMPOSE_DOMAIN}" \
      COMPOSITION_TEMPERATURE="${COMPOSITION_TEMPERATURE}" \
      COMPOSITION_TOPK="${COMPOSITION_TOPK}" \
      COMPAT_THRESHOLD="${COMPAT_THRESHOLD}" \
      TARGET_LAYERS="${TARGET_LAYERS}" \
      ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE}" \
      SUMMARY_PATH="${summary_path}" \
      DEVICE="cuda:${gpu_id}" \
      bash "${TASK_SCRIPT}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  TASK_NAMES_RUN+=("${task_name}")
  SUMMARY_PATHS_RUN+=("${summary_path}")
}

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "SPEC1=${SPEC1}"
echo "SPEC2=${SPEC2}"
echo "SPEC3=${SPEC3}"
echo "SPEC4=${SPEC4}"
echo "SCORE_STATS_PATH=${SCORE_STATS_PATH}"
echo "SCORE_SOURCE=${SCORE_SOURCE}"
echo "SCORE_NORMALIZER=${SCORE_NORMALIZER}"
echo "COMPOSITION_METHOD=${COMPOSITION_METHOD}"
echo "COMPOSE_DOMAIN=${COMPOSE_DOMAIN}"
echo "COMPOSITION_TEMPERATURE=${COMPOSITION_TEMPERATURE}"
echo "COMPOSITION_TOPK=${COMPOSITION_TOPK}"
echo "COMPAT_THRESHOLD=${COMPAT_THRESHOLD}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "ENABLE_DEBUG_CACHE=${ENABLE_DEBUG_CACHE}"
echo "E6_PARAM_TAG=${E6_PARAM_TAG}"
echo "SUMMARY_DIR=${SUMMARY_DIR}"
echo "SUMMARY_PREFIX=${SUMMARY_PREFIX}"
echo "MERGE_SUMMARY_FILE=${MERGE_SUMMARY_FILE}"
echo "TASKS=${TASKS}"
echo "GPU_IDS=${GPU_IDS_VALUE}"
echo "RUN_TS=${RUN_TS}"
echo "STAGGER_SECONDS=${STAGGER_SECONDS}"
echo "LOG_DIR=${LOG_DIR}"

for idx in "${!TASK_ARR[@]}"; do
  run_task "${TASK_ARR[$idx]}" "${GPU_IDS_ARR[$idx]}"
  if [[ "${STAGGER_SECONDS}" -gt 0 && $((idx + 1)) -lt "${#TASK_ARR[@]}" ]]; then
    sleep "${STAGGER_SECONDS}"
  fi
done

status=0
SUCCESS_SUMMARIES=()
FAILED_RUNS=()
for idx in "${!PIDS[@]}"; do
  if wait "${PIDS[$idx]}"; then
    if [[ -f "${SUMMARY_PATHS_RUN[$idx]}" ]]; then
      SUCCESS_SUMMARIES+=("${SUMMARY_PATHS_RUN[$idx]}")
    else
      FAILED_RUNS+=("${TASK_NAMES_RUN[$idx]}:missing_summary")
      status=1
    fi
  else
    FAILED_RUNS+=("${TASK_NAMES_RUN[$idx]}:task_failed")
    status=1
  fi
done

python - "${MERGE_SUMMARY_FILE}" "${RUN_TS}" "${SUMMARY_PREFIX}" "${FAILED_RUNS[*]-}" "${SUCCESS_SUMMARIES[@]-}" <<'PY'
import json
import sys
from pathlib import Path

merge_summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
summary_prefix = sys.argv[3]
failed_runs = [item for item in (sys.argv[4].split() if len(sys.argv) > 4 else []) if item]
summary_files = sys.argv[5:] if len(sys.argv) > 5 else []

task_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    task_runs.append(summary)

payload = {
    "mode": "composable-e6",
    "summary_prefix": summary_prefix,
    "run_ts": run_ts,
    "n_tasks": len(task_runs),
    "failed_runs": failed_runs,
    "task_runs": task_runs,
}
merge_summary_path.parent.mkdir(parents=True, exist_ok=True)
with merge_summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print(f"Merge summary written to: {merge_summary_path}")
PY

exit "${status}"
