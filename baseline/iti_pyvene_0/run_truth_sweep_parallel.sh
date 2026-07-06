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
RUN_PREFIX="${RUN_PREFIX:-truth_sweep}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-7}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-7}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LAYERS="${LAYERS:-6 10 14 18 22 26}"
TOP_K_HEADS_LIST="${TOP_K_HEADS_LIST:-8 16 32 48 64}"
ALPHA="${ALPHA:-0.9}"
MODE="${MODE:-train_eval}"

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

run_train_sweep() {
  local gpu_arr=(${GPU_IDS})
  local gpu_count="${#gpu_arr[@]}"
  local job_idx=0

  for layer in ${LAYERS}; do
    for topk in ${TOP_K_HEADS_LIST}; do
      wait_for_slot "${TRAIN_CONCURRENCY}"
      local gpu_id="${gpu_arr[$((job_idx % gpu_count))]}"
      local run_name="${RUN_PREFIX}_l${layer}_k${topk}"
      launch_job "${gpu_id}" env \
        PYTHON_BIN="${PYTHON_BIN}" \
        RUN_NAME="${run_name}" \
        TASK="truth" \
        BASE_MODEL="${BASE_MODEL}" \
        BATCH_SIZE="${BATCH_SIZE}" \
        TOP_K_HEADS="${topk}" \
        RETAIN_LAYERS="${layer}" \
        DEVICE="cuda:0" \
        TORCH_DTYPE="${TORCH_DTYPE}" \
        bash baseline/iti_pyvene_0/run_train_experiment.sh
      job_idx=$((job_idx + 1))
    done
  done

  wait
}

run_eval_sweep() {
  local gpu_arr=(${GPU_IDS})
  local gpu_count="${#gpu_arr[@]}"
  local job_idx=0

  for layer in ${LAYERS}; do
    for topk in ${TOP_K_HEADS_LIST}; do
      wait_for_slot "${EVAL_CONCURRENCY}"
      local gpu_id="${gpu_arr[$((job_idx % gpu_count))]}"
      local run_name="${RUN_PREFIX}_l${layer}_k${topk}"
      local artifact_dir="baseline/iti_pyvene_0/artifacts/interventions/${MODEL_NAME}/truth/${run_name}"
      launch_job "${gpu_id}" env \
        PYTHON_BIN="${PYTHON_BIN}" \
        EVAL_NAME="${run_name}" \
        TASK="truth" \
        BASE_MODEL="${BASE_MODEL}" \
        ITI_ARTIFACT_DIR="${artifact_dir}" \
        ITI_LAYERS="${layer}" \
        ITI_ALPHA="${ALPHA}" \
        DEVICE="cuda:0" \
        bash baseline/iti_pyvene_0/run_eval_experiment.sh
      job_idx=$((job_idx + 1))
    done
  done

  wait
}

case "${MODE}" in
  train)
    run_train_sweep
    ;;
  eval)
    run_eval_sweep
    ;;
  train_eval)
    run_train_sweep
    run_eval_sweep
    ;;
  *)
    echo "Unsupported MODE=${MODE}. Use train, eval, or train_eval." >&2
    exit 1
    ;;
esac
