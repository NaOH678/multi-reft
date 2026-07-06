#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SUMMARY_JSON="${SUMMARY_JSON:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
PERPLEXITY_JSON="${PERPLEXITY_JSON:-}"
JUDGE_SUMMARY_JSON="${JUDGE_SUMMARY_JSON:-}"
AUDIT_SUMMARY_JSON="${AUDIT_SUMMARY_JSON:-}"
JUDGE_METHOD="${JUDGE_METHOD:-}"
ALLOW_INPUT_MISMATCH="${ALLOW_INPUT_MISMATCH:-0}"

if [[ -z "${SUMMARY_JSON}" ]]; then
  echo "SUMMARY_JSON is required." >&2
  exit 1
fi

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "PERPLEXITY_JSON=${PERPLEXITY_JSON}"
echo "JUDGE_SUMMARY_JSON=${JUDGE_SUMMARY_JSON}"
echo "AUDIT_SUMMARY_JSON=${AUDIT_SUMMARY_JSON}"
echo "JUDGE_METHOD=${JUDGE_METHOD}"
echo "ALLOW_INPUT_MISMATCH=${ALLOW_INPUT_MISMATCH}"

cmd=(
  "${PYTHON_BIN}" multi_train/eval_toxicity/merge_toxicity_eval_artifacts.py
  --summary_json "${SUMMARY_JSON}"
)

if [[ -n "${OUTPUT_JSON}" ]]; then
  cmd+=(--output_json "${OUTPUT_JSON}")
fi
if [[ -n "${PERPLEXITY_JSON}" ]]; then
  cmd+=(--perplexity_json "${PERPLEXITY_JSON}")
fi
if [[ -n "${JUDGE_SUMMARY_JSON}" ]]; then
  cmd+=(--judge_summary_json "${JUDGE_SUMMARY_JSON}")
fi
if [[ -n "${AUDIT_SUMMARY_JSON}" ]]; then
  cmd+=(--audit_summary_json "${AUDIT_SUMMARY_JSON}")
fi
if [[ -n "${JUDGE_METHOD}" ]]; then
  cmd+=(--judge_method "${JUDGE_METHOD}")
fi
if [[ "${ALLOW_INPUT_MISMATCH}" == "1" ]]; then
  cmd+=(--allow_input_mismatch)
fi

"${cmd[@]}"

echo "Toxicity eval summary merge finished."
echo "Output: ${OUTPUT_JSON:-${SUMMARY_JSON}}"
