#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK_SCRIPT="${SCRIPT_DIR}/evaluate_trained_router_task.sh"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
ROUTER_CHECKPOINT_DIR="${ROUTER_CHECKPOINT_DIR:-}"
ROUTER_METADATA_PATH="${ROUTER_METADATA_PATH:-}"
BASE_SCORE_STATS_PATH="${BASE_SCORE_STATS_PATH:-}"
ROUTER_FEATURE_STATS_PATH="${ROUTER_FEATURE_STATS_PATH:-}"
ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE:-1}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-}"
SUMMARY_DIR="${SUMMARY_DIR:-multi_train/eval_router/summaries}"

TASKS="${TASKS:-truth bias ethics privacy toxicity}"
GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4}"
STAGGER_SECONDS="${STAGGER_SECONDS:-10}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-multi_train/logs/trained_router_eval_${RUN_TS}}"
mkdir -p "${LOG_DIR}"
mkdir -p "${SUMMARY_DIR}"

if [[ -z "${ROUTER_CHECKPOINT_DIR}" ]]; then
  echo "ROUTER_CHECKPOINT_DIR must be set." >&2
  exit 1
fi

read -r -a TASK_ARR <<< "${TASKS}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"

if [[ "${#TASK_ARR[@]}" -eq 0 ]]; then
  echo "No tasks provided in TASKS." >&2
  exit 1
fi

if [[ "${#TASK_ARR[@]}" -gt "${#GPU_IDS_ARR[@]}" ]]; then
  echo "Need at least as many GPU ids as tasks." >&2
  echo "TASKS=${TASKS}" >&2
  echo "GPU_IDS=${GPU_IDS_VALUE}" >&2
  exit 1
fi

PIDS=()
TASK_NAMES_RUN=()
SUMMARY_PATHS_RUN=()

build_summary_prefix() {
  local user_prefix="$1"
  local base_model="$2"
  local router_checkpoint_dir="$3"

  python - "$user_prefix" "$base_model" "$router_checkpoint_dir" <<'PY'
import sys
from pathlib import Path
from multi_train.eval_common.output_naming import resolve_output_prefix

user_prefix = sys.argv[1] if len(sys.argv) > 1 else ""
base_model = sys.argv[2] if len(sys.argv) > 2 else ""
router_checkpoint_dir = sys.argv[3] if len(sys.argv) > 3 else ""

if user_prefix.strip():
    print(resolve_output_prefix(user_prefix, base_model, "trained-router"))
else:
    path = Path(str(router_checkpoint_dir).rstrip("/"))
    if path.name == "intervenable_model":
        path = path.parent
    if path.name.lower().startswith(("checkpoint-", "chpoint-", "ckpt-")) and path.parent.name:
        print(path.parent.name)
    else:
        print(path.name)
PY
}

build_result_prefix() {
  local summary_prefix="$1"
  local router_checkpoint_dir="$2"

  python - "$summary_prefix" "$router_checkpoint_dir" <<'PY'
import sys
from pathlib import Path

summary_prefix = sys.argv[1] if len(sys.argv) > 1 else ""
router_checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else ""

path = Path(str(router_checkpoint_dir).rstrip("/"))
if path.name == "intervenable_model":
    path = path.parent

name = path.name.lower()
if name.startswith(("checkpoint-", "chpoint-", "ckpt-")):
    print(f"{summary_prefix}-{path.name}")
else:
    print(summary_prefix)
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

SUMMARY_PREFIX="$(build_summary_prefix "${MERGE_SUMMARY_PATH}" "${BASE_MODEL}" "${ROUTER_CHECKPOINT_DIR}")"
RESULT_PREFIX="$(build_result_prefix "${SUMMARY_PREFIX}" "${ROUTER_CHECKPOINT_DIR}")"
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
    echo "ROUTER_CHECKPOINT_DIR=${ROUTER_CHECKPOINT_DIR}"
    echo "ROUTER_METADATA_PATH=${ROUTER_METADATA_PATH}"
    echo "BASE_SCORE_STATS_PATH=${BASE_SCORE_STATS_PATH}"
    echo "ROUTER_FEATURE_STATS_PATH=${ROUTER_FEATURE_STATS_PATH}"
    echo "ENABLE_DEBUG_CACHE=${ENABLE_DEBUG_CACHE}"
    echo "TARGET_LAYERS=${TARGET_LAYERS}"
    echo "SUMMARY_PATH=${summary_path}"
    echo "RESULT_PREFIX=${RESULT_PREFIX}"
    echo
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      TASK="${task_name}" \
      BASE_MODEL="${BASE_MODEL}" \
      ROUTER_CHECKPOINT_DIR="${ROUTER_CHECKPOINT_DIR}" \
      ROUTER_METADATA_PATH="${ROUTER_METADATA_PATH}" \
      BASE_SCORE_STATS_PATH="${BASE_SCORE_STATS_PATH}" \
      ROUTER_FEATURE_STATS_PATH="${ROUTER_FEATURE_STATS_PATH}" \
      ENABLE_DEBUG_CACHE="${ENABLE_DEBUG_CACHE}" \
      TARGET_LAYERS="${TARGET_LAYERS}" \
      SUMMARY_PATH="${summary_path}" \
      RESULT_PREFIX="${RESULT_PREFIX}" \
      DEVICE="cuda:${gpu_id}" \
      bash "${TASK_SCRIPT}"
  ) >"${log_file}" 2>&1 &

  PIDS+=("$!")
  TASK_NAMES_RUN+=("${task_name}")
  SUMMARY_PATHS_RUN+=("${summary_path}")
}

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "ROUTER_CHECKPOINT_DIR=${ROUTER_CHECKPOINT_DIR}"
echo "ROUTER_METADATA_PATH=${ROUTER_METADATA_PATH}"
echo "BASE_SCORE_STATS_PATH=${BASE_SCORE_STATS_PATH}"
echo "ROUTER_FEATURE_STATS_PATH=${ROUTER_FEATURE_STATS_PATH}"
echo "ENABLE_DEBUG_CACHE=${ENABLE_DEBUG_CACHE}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "MERGE_SUMMARY_PATH=${MERGE_SUMMARY_PATH}"
echo "SUMMARY_DIR=${SUMMARY_DIR}"
echo "SUMMARY_PREFIX=${SUMMARY_PREFIX}"
echo "RESULT_PREFIX=${RESULT_PREFIX}"
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
    status=$?
  fi
done

python - "${MERGE_SUMMARY_FILE}" "${RUN_TS}" "${ROUTER_CHECKPOINT_DIR}" "${SUMMARY_PREFIX}" "${FAILED_RUNS[*]-}" "${SUCCESS_SUMMARIES[@]-}" <<'PY'
import json
import sys
from pathlib import Path

merge_summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
router_checkpoint_dir = sys.argv[3]
summary_prefix = sys.argv[4]
failed_runs = [item for item in (sys.argv[5].split() if len(sys.argv) > 5 else []) if item]
summary_files = sys.argv[6:] if len(sys.argv) > 6 else []

task_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    task_runs.append(
        {
            "task": summary.get("dataset") or summary.get("task"),
            "summary_file": str(p),
            "result_file": summary.get("result_file") or summary.get("results_csv"),
            "debug_summary_file": summary.get("debug_summary_file"),
            "model_tag": summary.get("model_tag"),
        }
    )

merge_summary = {
    "run_ts": run_ts,
    "router_checkpoint_dir": router_checkpoint_dir,
    "summary_prefix": summary_prefix,
    "failed_runs": failed_runs,
    "task_runs": task_runs,
}
merge_summary_path.parent.mkdir(parents=True, exist_ok=True)
with merge_summary_path.open("w", encoding="utf-8") as f:
    json.dump(merge_summary, f, indent=2)
print(str(merge_summary_path))
PY

if [[ "${status}" -ne 0 ]]; then
  echo "At least one trained-router evaluation task failed. Check logs under ${LOG_DIR}." >&2
  echo "Merge summary saved to: ${MERGE_SUMMARY_FILE}"
  exit "${status}"
fi

echo "All trained-router evaluation tasks finished."
echo "Logs saved to: ${LOG_DIR}"
echo "Merge summary saved to: ${MERGE_SUMMARY_FILE}"
