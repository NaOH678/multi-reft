#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
AUDIT_DIR="${AUDIT_DIR:-}"
API_MODEL="${API_MODEL:-gpt-5.4}"
API_BASE="${API_BASE:-http://35.220.164.252:3888/v1}"
API_KEY_ENV="${API_KEY_ENV:-JUDGE_KEY}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
SAVE_EVERY="${SAVE_EVERY:-10}"
CONCURRENCY="${CONCURRENCY:-8}"
OVERWRITE="${OVERWRITE:-0}"
PAIRWISE_ONLY="${PAIRWISE_ONLY:-0}"

if [[ -z "${AUDIT_DIR}" ]]; then
  echo "AUDIT_DIR is required." >&2
  exit 1
fi

INPUT_DIR="${AUDIT_DIR}/judge_inputs"
OUTPUT_DIR="${AUDIT_DIR}/judge_outputs"
mkdir -p "${OUTPUT_DIR}"

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "API_MODEL=${API_MODEL}"
echo "API_BASE=${API_BASE}"
echo "API_KEY_ENV=${API_KEY_ENV}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo "MAX_RETRIES=${MAX_RETRIES}"
echo "RETRY_SLEEP=${RETRY_SLEEP}"
echo "REQUEST_TIMEOUT=${REQUEST_TIMEOUT}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "OVERWRITE=${OVERWRITE}"
echo "PAIRWISE_ONLY=${PAIRWISE_ONLY}"

shopt -s nullglob
if [[ "${PAIRWISE_ONLY}" == "1" ]]; then
  INPUT_GLOB="${INPUT_DIR}/pairwise_*.jsonl"
else
  INPUT_GLOB="${INPUT_DIR}/*.jsonl"
fi

for input_jsonl in ${INPUT_GLOB}; do
  output_jsonl="${OUTPUT_DIR}/$(basename "${input_jsonl}")"
  echo "Running judge for ${input_jsonl}"
  if [[ "${OVERWRITE}" == "1" ]]; then
    "${PYTHON_BIN}" multi_train/eval_toxicity/run_llm_judge.py \
      --input_jsonl "${input_jsonl}" \
      --output_jsonl "${output_jsonl}" \
      --api_model "${API_MODEL}" \
      --api_base "${API_BASE}" \
      --api_key_env "${API_KEY_ENV}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --max_retries "${MAX_RETRIES}" \
      --retry_sleep "${RETRY_SLEEP}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --save_every "${SAVE_EVERY}" \
      --concurrency "${CONCURRENCY}" \
      --overwrite
  else
    "${PYTHON_BIN}" multi_train/eval_toxicity/run_llm_judge.py \
      --input_jsonl "${input_jsonl}" \
      --output_jsonl "${output_jsonl}" \
      --api_model "${API_MODEL}" \
      --api_base "${API_BASE}" \
      --api_key_env "${API_KEY_ENV}" \
      --max_tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --max_retries "${MAX_RETRIES}" \
      --retry_sleep "${RETRY_SLEEP}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --save_every "${SAVE_EVERY}" \
      --concurrency "${CONCURRENCY}"
  fi
done

echo "Judge run finished."
echo "Judge outputs: ${OUTPUT_DIR}"
