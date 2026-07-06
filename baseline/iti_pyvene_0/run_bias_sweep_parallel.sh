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
RUN_PREFIX="${RUN_PREFIX:-bias_sweep}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-7}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LAYERS="${LAYERS:-14 26}"
ALPHA="${ALPHA:-1.0 0.1}"
TOKEN_STRATEGIES="${TOKEN_STRATEGIES:-last all}"
MODE="${MODE:-eval}"

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

MODEL_NAME="$(normalize_model_name "${BASE_MODEL}")"
ARTIFACT_NAME="${ARTIFACT_NAME:-default}"

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

  for layer in ${LAYERS}; do
    for alpha in ${ALPHA}; do
      for token_strategy in ${TOKEN_STRATEGIES}; do
        wait_for_slot "${EVAL_CONCURRENCY}"
        local gpu_id="${gpu_arr[$((job_idx % gpu_count))]}"
        local run_name="${RUN_PREFIX}_l${layer}_a${alpha}_${token_strategy}"
        local artifact_dir="baseline/iti_pyvene_0/artifacts/interventions/${MODEL_NAME}/bias/${ARTIFACT_NAME}"
        if [[ ! -f "${artifact_dir}/layer_${layer}.pt" ]]; then
          echo "skip missing layer artifact: ${artifact_dir}/layer_${layer}.pt"
          continue
        fi
        launch_job "${gpu_id}" env \
          PYTHON_BIN="${PYTHON_BIN}" \
          EVAL_NAME="${run_name}" \
          TASK="bias" \
          BASE_MODEL="${BASE_MODEL}" \
          ITI_ARTIFACT_DIR="${artifact_dir}" \
          ITI_ALPHA="${alpha}" \
          ITI_INCLUDE_PROMPT='0' \
          ITI_TOKEN_STRATEGY="${token_strategy}" \
          ITI_BASE_UNIT_LOCATION='' \
          DEVICE="cuda:0" \
          bash baseline/iti_pyvene_0/run_eval_experiment.sh
        job_idx=$((job_idx + 1))
      done
    done
  done

  wait
}

case "${MODE}" in
  eval)
    run_eval_sweep
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use eval." >&2
    exit 1
    ;;
esac
