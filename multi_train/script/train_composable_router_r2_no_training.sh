#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-dexmanip}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

SPEC1="${SPEC1:-${WORKSPACE_CONT}/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-${WORKSPACE_CONT}/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-${WORKSPACE_CONT}/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-${WORKSPACE_CONT}/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

ACCELERATE_BIN="${ACCELERATE_BIN:-/opt/conda/bin/accelerate}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/train_composable_router_r2_no_training_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/train_composable_router_r2_no_training.log}"
OUTPUT_DIR="${OUTPUT_DIR:-multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-no-training-${RUN_TAG}}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
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
      export ACCELERATE_BIN='${ACCELERATE_BIN}' &&
      export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' &&
      export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' &&
      export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}' &&
      export NUM_PROCESSES='${NUM_PROCESSES}' &&
      export OUTPUT_DIR='${OUTPUT_DIR}' &&
      export SPEC1='${SPEC1}' &&
      export SPEC2='${SPEC2}' &&
      export SPEC3='${SPEC3}' &&
      export SPEC4='${SPEC4}' &&
      export TASK_DATASET_MAP_JSON='${TASK_DATASET_MAP_JSON-}' &&
      export SAMPLING_PROBS_JSON='${SAMPLING_PROBS_JSON-}' &&
      export TASK_TRAIN_QUOTA_JSON='${TASK_TRAIN_QUOTA_JSON-}' &&
      export ROUTER_FEATURE_STATS_PATH='${ROUTER_FEATURE_STATS_PATH-}' &&
      export BASE_SCORE_STATS_PATH='${BASE_SCORE_STATS_PATH-}' &&
      export MODEL_MAX_LENGTH='${MODEL_MAX_LENGTH-}' &&
      export POSITIONS='${POSITIONS-}' &&
      export TARGET_LAYERS='${TARGET_LAYERS-}' &&
      export ROUTER_LAYERS='${ROUTER_LAYERS-}' &&
      export ROUTER_LAYER_POLICY='${ROUTER_LAYER_POLICY-}' &&
      export ROUTER_FEATURE_NAMES='${ROUTER_FEATURE_NAMES-}' &&
      export PER_DEVICE_TRAIN_BATCH_SIZE='${PER_DEVICE_TRAIN_BATCH_SIZE-}' &&
      export PER_DEVICE_EVAL_BATCH_SIZE='${PER_DEVICE_EVAL_BATCH_SIZE-}' &&
      export GRADIENT_ACCUMULATION_STEPS='${GRADIENT_ACCUMULATION_STEPS-}' &&
      export LEARNING_RATE='${LEARNING_RATE-}' &&
      export WEIGHT_DECAY='${WEIGHT_DECAY-}' &&
      export WARMUP_RATIO='${WARMUP_RATIO-}' &&
      export NUM_TRAIN_EPOCHS='${NUM_TRAIN_EPOCHS-}' &&
      export LOGGING_STEPS='${LOGGING_STEPS-}' &&
      export SAVE_STRATEGY='${SAVE_STRATEGY-}' &&
      export EVAL_STRATEGY='${EVAL_STRATEGY-}' &&
      export SAVE_TOTAL_LIMIT='${SAVE_TOTAL_LIMIT-}' &&
      export REPORT_TO='${REPORT_TO-}' &&
      export RUN_NAME='${RUN_NAME-}' &&
      export MAX_TRAIN_SAMPLES_PER_TASK='${MAX_TRAIN_SAMPLES_PER_TASK-}' &&
      export MAX_EVAL_SAMPLES_PER_TASK='${MAX_EVAL_SAMPLES_PER_TASK-}' &&
      export EVAL_HOLDOUT_RATIO='${EVAL_HOLDOUT_RATIO-}' &&
      export SEED='${SEED-}' &&
      export SANITY_EVAL_ONLY='${SANITY_EVAL_ONLY-}' &&
      export SANITY_EVAL_SPLITS='${SANITY_EVAL_SPLITS-}' &&
      export BASE_COMPOSITION_TEMPERATURE='${BASE_COMPOSITION_TEMPERATURE-}' &&
      export BASE_COMPOSITION_TOPK='${BASE_COMPOSITION_TOPK-}' &&
      export BASE_COMPAT_THRESHOLD='${BASE_COMPAT_THRESHOLD-}' &&
      export BASE_SCORE_SOURCE='${BASE_SCORE_SOURCE-}' &&
      export BASE_SCORE_NORMALIZER='${BASE_SCORE_NORMALIZER-}' &&
      export ROUTER_TEMPERATURE='${ROUTER_TEMPERATURE-}' &&
      export POLICY_HIDDEN_DIM='${POLICY_HIDDEN_DIM-}' &&
      export POLICY_PROJECTION_DIM='${POLICY_PROJECTION_DIM-}' &&
      export ROUTER_FEATURE_CLIP='${ROUTER_FEATURE_CLIP-}' &&
      export ROUTER_FEATURE_EPS='${ROUTER_FEATURE_EPS-}' &&
      export COMPAT_IMPL='${COMPAT_IMPL-}' &&
      export MAIN_PROCESS_PORT='${MAIN_PROCESS_PORT-}' &&
      bash multi_train/script/train_composable_router_r2_no_training_worker.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Composable router R2 no-training sanity job finished."
echo "Host log: ${HOST_LOG_FILE}"
