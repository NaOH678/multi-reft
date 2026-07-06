#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-dexmanip}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

SPEC1="${SPEC1:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-512}"
POSITIONS="${POSITIONS:-7}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
TEXT_COLUMN="${TEXT_COLUMN:-input}"
TRUTH_DATASET="${TRUTH_DATASET:-dataset/alignment_truthful_format}"
MORAL_DATASET="${MORAL_DATASET:-dataset/alignment_moral_cls}"
BIAS_DATASET="${BIAS_DATASET:-dataset/alignment_stereotype_format}"
TOXICITY_DATASET="${TOXICITY_DATASET:-dataset/alignment_toxic_format}"
CALIB_DIR="${CALIB_DIR:-multi_train/calibration/train_input}"
PROMPT_DIR="${PROMPT_DIR:-}"
STATS_DIR="${STATS_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/build_intervention_stats_shared_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/build_intervention_stats_shared.log}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "DEVICE=${DEVICE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "POSITIONS=${POSITIONS}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "TEXT_COLUMN=${TEXT_COLUMN}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "TRUTH_DATASET=${TRUTH_DATASET}"
echo "MORAL_DATASET=${MORAL_DATASET}"
echo "BIAS_DATASET=${BIAS_DATASET}"
echo "TOXICITY_DATASET=${TOXICITY_DATASET}"
echo "CALIB_DIR=${CALIB_DIR}"
echo "PROMPT_DIR=${PROMPT_DIR}"
echo "STATS_DIR=${STATS_DIR}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"
echo "RUN_TAG=${RUN_TAG}"
echo "HOST_LOG_FILE=${HOST_LOG_FILE}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"

srun \
  -p "${PARTITION}" \
  --gres="${GRES}" \
  ${SRUN_EXTRA_ARGS} \
  apptainer exec \
    --cleanenv \
    --nv \
    --bind /mnt:/mnt \
    --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
    "${IMAGE_PATH}" \
    bash -c "
      cd '${WORKSPACE_CONT}' &&
      export PYTHON_BIN='${PYTHON_BIN}' &&
      export BASE_MODEL='${BASE_MODEL}' &&
      export DEVICE='${DEVICE}' &&
      export BATCH_SIZE='${BATCH_SIZE}' &&
      export MAX_LENGTH='${MAX_LENGTH}' &&
      export POSITIONS='${POSITIONS}' &&
      export TARGET_LAYERS='${TARGET_LAYERS}' &&
      export TEXT_COLUMN='${TEXT_COLUMN}' &&
      export TRUTH_DATASET='${TRUTH_DATASET}' &&
      export MORAL_DATASET='${MORAL_DATASET}' &&
      export BIAS_DATASET='${BIAS_DATASET}' &&
      export TOXICITY_DATASET='${TOXICITY_DATASET}' &&
      export CALIB_DIR='${CALIB_DIR}' &&
      export PROMPT_DIR='${PROMPT_DIR}' &&
      export STATS_DIR='${STATS_DIR}' &&
      export OUTPUT_JSON='${OUTPUT_JSON}' &&
      export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' &&
      export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' &&
      export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}' &&
      export MAX_SAMPLES='${MAX_SAMPLES}' &&
      export SPEC1='${SPEC1}' &&
      export SPEC2='${SPEC2}' &&
      export SPEC3='${SPEC3}' &&
      export SPEC4='${SPEC4}' &&
      bash multi_train/script/build_intervention_stats_shared_train_worker.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Shared intervention-stats build job finished."
echo "Host log: ${HOST_LOG_FILE}"
