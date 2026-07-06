#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
VECTOR_DIR="${VECTOR_DIR:-baseline/repe_pyvene_0/artifacts/vectors/llama3-8b/ethics/default}"
ALPHAS="${ALPHAS:-0.2 0.5 1.0 1.5 2.0}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
CONCURRENCY="${CONCURRENCY:-8}"
ETHICS_BATCH_SIZE="${ETHICS_BATCH_SIZE:-16}"
ETHICS_MAX_TOKENS="${ETHICS_MAX_TOKENS:-16}"
ETHICS_GREEDY_DECODING="${ETHICS_GREEDY_DECODING:-1}"
ETHICS_PROMPT_TYPES="${ETHICS_PROMPT_TYPES:-0 1 2 3 4 5}"

wait_for_slot() {
  local limit="$1"
  while true; do
    local running
    running="$(jobs -pr | wc -l | tr -d ' ')"
    if (( running < limit )); then
      break
    fi
    sleep 2
  done
}

launch_job() {
  local gpu_id="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" &
}

run_eval_sweep() {
  local gpu_arr=(${GPU_IDS})
  local gpu_count="${#gpu_arr[@]}"
  local job_idx=0

  for alpha in ${ALPHAS}; do
    wait_for_slot "${CONCURRENCY}"
    local gpu_id="${gpu_arr[$((job_idx % gpu_count))]}"
    local eval_name="ethics-a${alpha}"
    launch_job "${gpu_id}" env \
      PYTHON_BIN="${PYTHON_BIN}" \
      EVAL_NAME="${eval_name}" \
      TASK="ethics" \
      BASE_MODEL="${BASE_MODEL}" \
      REPE_VECTOR_DIR="${VECTOR_DIR}" \
      REPE_ALPHA="${alpha}" \
      ETHICS_BATCH_SIZE="${ETHICS_BATCH_SIZE}" \
      ETHICS_MAX_TOKENS="${ETHICS_MAX_TOKENS}" \
      ETHICS_GREEDY_DECODING="${ETHICS_GREEDY_DECODING}" \
      ETHICS_PROMPT_TYPES="${ETHICS_PROMPT_TYPES}" \
      DEVICE="cuda:0" \
      bash baseline/repe_pyvene_0/run_eval_experiment.sh
    job_idx=$((job_idx + 1))
  done

  wait
}

run_eval_sweep
