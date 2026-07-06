#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_DIR="${AUDIT_DIR:-}"
SUMMARY_JSON="${SUMMARY_JSON:-}"
PERPLEXITY_JSON="${PERPLEXITY_JSON:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
JUDGE_METHOD="${JUDGE_METHOD:-}"
ALLOW_INPUT_MISMATCH="${ALLOW_INPUT_MISMATCH:-0}"

if [[ -z "${AUDIT_DIR}" ]]; then
  echo "AUDIT_DIR is required." >&2
  exit 1
fi
if [[ -z "${SUMMARY_JSON}" ]]; then
  echo "SUMMARY_JSON is required." >&2
  exit 1
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "PERPLEXITY_JSON=${PERPLEXITY_JSON}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "JUDGE_METHOD=${JUDGE_METHOD}"
echo "ALLOW_INPUT_MISMATCH=${ALLOW_INPUT_MISMATCH}"
echo "This script runs locally and does not use srun."

PYTHON_BIN="${PYTHON_BIN}" \
AUDIT_DIR="${AUDIT_DIR}" \
bash multi_train/script/run_toxicity_llm_judge.sh

PYTHON_BIN="${PYTHON_BIN}" \
AUDIT_DIR="${AUDIT_DIR}" \
bash multi_train/script/summarize_toxicity_llm_judge.sh

MERGE_ARGS=(
  "PYTHON_BIN=${PYTHON_BIN}"
  "SUMMARY_JSON=${SUMMARY_JSON}"
  "JUDGE_SUMMARY_JSON=${AUDIT_DIR}/judge_summary/judge_summary.json"
  "AUDIT_SUMMARY_JSON=${AUDIT_DIR}/audit_summary.json"
  "ALLOW_INPUT_MISMATCH=${ALLOW_INPUT_MISMATCH}"
)

if [[ -n "${PERPLEXITY_JSON}" ]]; then
  MERGE_ARGS+=("PERPLEXITY_JSON=${PERPLEXITY_JSON}")
fi
if [[ -n "${OUTPUT_JSON}" ]]; then
  MERGE_ARGS+=("OUTPUT_JSON=${OUTPUT_JSON}")
fi
if [[ -n "${JUDGE_METHOD}" ]]; then
  MERGE_ARGS+=("JUDGE_METHOD=${JUDGE_METHOD}")
fi

env "${MERGE_ARGS[@]}" bash multi_train/script/merge_toxicity_eval_summary.sh

echo "Local toxicity judge finalize finished."
echo "Merged summary: ${OUTPUT_JSON:-${SUMMARY_JSON}}"
