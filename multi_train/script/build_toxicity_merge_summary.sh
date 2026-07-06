#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DATASETS_VALUE="${DATASETS_VALUE:-}"
PROMPTS_VALUE="${PROMPTS_VALUE:-}"
FAILED_RUNS_VALUE="${FAILED_RUNS_VALUE:-}"

if [[ -z "${MERGE_SUMMARY_PATH}" ]]; then
  echo "MERGE_SUMMARY_PATH is required." >&2
  exit 1
fi

read -r -a DATASETS_ARR <<< "${DATASETS_VALUE}"
read -r -a PROMPTS_ARR <<< "${PROMPTS_VALUE}"
read -r -a FAILED_RUNS_ARR <<< "${FAILED_RUNS_VALUE}"

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MERGE_SUMMARY_PATH=${MERGE_SUMMARY_PATH}"
echo "RUN_TS=${RUN_TS}"
echo "DATASETS_VALUE=${DATASETS_VALUE}"
echo "PROMPTS_VALUE=${PROMPTS_VALUE}"
echo "FAILED_RUNS_VALUE=${FAILED_RUNS_VALUE}"
echo "SUMMARY_FILES=$*"

cmd=(
  "${PYTHON_BIN}" multi_train/eval_toxicity/build_merge_summary.py
  --merge_summary_path "${MERGE_SUMMARY_PATH}"
  --run_ts "${RUN_TS}"
  --datasets "${DATASETS_ARR[@]}"
  --prompts "${PROMPTS_ARR[@]}"
  --summary_files "$@"
)

if [[ ${#FAILED_RUNS_ARR[@]} -gt 0 ]]; then
  cmd+=(--failed_runs "${FAILED_RUNS_ARR[@]}")
fi

"${cmd[@]}"

echo "Toxicity merge summary built."
echo "Output: ${MERGE_SUMMARY_PATH}"
