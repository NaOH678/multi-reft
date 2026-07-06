#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-dexmanip}"
GRES="${GRES:-gpu:1}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/evaluate_composable_toxicity_case_study_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/evaluate_composable_toxicity_case_study.log}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "HOST_LOG_FILE=${HOST_LOG_FILE}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"

srun \
  -p "${PARTITION}" \
  --gres="${GRES}" \
  -w HOST-10-140-66-138 \
  ${SRUN_EXTRA_ARGS} \
  apptainer exec \
    --cleanenv \
    --nv \
    --bind /mnt:/mnt \
    --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
    "${IMAGE_PATH}" \
    bash -c "
      cd '${WORKSPACE_CONT}' &&
      export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' &&
      export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' &&
      export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}' &&
      export PYTHON_BIN='${PYTHON_BIN-/opt/conda/bin/python}' &&
      export BASE_MODEL='${BASE_MODEL-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}' &&
      export SPEC1='${SPEC1-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}' &&
      export SPEC2='${SPEC2-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}' &&
      export SPEC3='${SPEC3-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}' &&
      export SPEC4='${SPEC4-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}' &&
      export SCORE_STATS_PATH='${SCORE_STATS_PATH-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}' &&
      export SCORE_SOURCE='${SCORE_SOURCE-intervention_norm}' &&
      export SCORE_NORMALIZER='${SCORE_NORMALIZER-log_zscore}' &&
      export COMPOSITION_METHOD='${COMPOSITION_METHOD-compat_filtered_topk}' &&
      export COMPOSE_DOMAIN='${COMPOSE_DOMAIN-output}' &&
      export COMPOSITION_TEMPERATURE='${COMPOSITION_TEMPERATURE-1.0}' &&
      export COMPOSITION_TOPK='${COMPOSITION_TOPK-2}' &&
      export COMPAT_THRESHOLD='${COMPAT_THRESHOLD-0.0}' &&
      export TARGET_LAYERS='${TARGET_LAYERS--1}' &&
      export ENABLE_DEBUG_CACHE='${ENABLE_DEBUG_CACHE-1}' &&
      export TASKS='toxicity' &&
      export GPU_IDS='${GPU_IDS-0}' &&
      export LOG_DIR='${LOG_DIR-multi_train/logs/toxicity_case_study_${RUN_TAG}}' &&
      export DATASETS='${DATASETS-toxic nontoxic}' &&
      export PROMPTS='${PROMPTS-benign adversarial}' &&
      export DATASET_ROOT='${DATASET_ROOT-multi_train/eval_toxicity/data/user_prompts}' &&
      export MAX_SAMPLES='${MAX_SAMPLES-200}' &&
      export TOXICITY_SAMPLE_MODE='${TOXICITY_SAMPLE_MODE-random}' &&
      export TOXICITY_SAMPLE_SEED='${TOXICITY_SAMPLE_SEED-42}' &&
      export TOXICITY_MECHANISM_TRACE='${TOXICITY_MECHANISM_TRACE-1}' &&
      export TOXICITY_MECHANISM_TRACE_OUTPUT='${TOXICITY_MECHANISM_TRACE_OUTPUT-}' &&
      export TOXICITY_BATCH_SIZE='${TOXICITY_BATCH_SIZE-8}' &&
      export TOXICITY_POSITIONS='${TOXICITY_POSITIONS-11}' &&
      export TOXICITY_GREEDY_DECODING='${TOXICITY_GREEDY_DECODING-0}' &&
      export TOXICITY_N_GENERATIONS='${TOXICITY_N_GENERATIONS-5}' &&
      export TOXICITY_MAX_TOKENS='${TOXICITY_MAX_TOKENS-50}' &&
      export TOXICITY_TEMPERATURE='${TOXICITY_TEMPERATURE-0.7}' &&
      export TOXICITY_RUN_ANALYSIS='${TOXICITY_RUN_ANALYSIS-1}' &&
      export TOXICITY_ANALYSIS_BATCH_SIZE='${TOXICITY_ANALYSIS_BATCH_SIZE-16}' &&
      export TOXICITY_DETOXIFY_MODEL='${TOXICITY_DETOXIFY_MODEL-original}' &&
      export TOXICITY_ANALYSIS_DEVICE='${TOXICITY_ANALYSIS_DEVICE-cuda:0}' &&
      export TOXICITY_TRUTHFUL_SCORE_PENALTY='${TOXICITY_TRUTHFUL_SCORE_PENALTY-0.0}' &&
      bash multi_train/script/evaluate_composable_toxicity_case_study_all.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Composable toxicity case study job finished."
echo "Host log: ${HOST_LOG_FILE}"
