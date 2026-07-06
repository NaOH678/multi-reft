#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-dexmanip}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/evaluate_trained_router_all_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/evaluate_trained_router_all.log}"

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
      export ROUTER_CHECKPOINT_DIR='${ROUTER_CHECKPOINT_DIR-}' &&
      export ROUTER_METADATA_PATH='${ROUTER_METADATA_PATH-}' &&
      export BASE_SCORE_STATS_PATH='${BASE_SCORE_STATS_PATH-}' &&
      export ROUTER_FEATURE_STATS_PATH='${ROUTER_FEATURE_STATS_PATH-}' &&
      export ENABLE_DEBUG_CACHE='${ENABLE_DEBUG_CACHE-1}' &&
      export TARGET_LAYERS='${TARGET_LAYERS-}' &&
      export MERGE_SUMMARY_PATH='${MERGE_SUMMARY_PATH-}' &&
      export SUMMARY_DIR='${SUMMARY_DIR-}' &&
      export RUN_TS='${RUN_TS-}' &&
      export TASKS='${TASKS-truth bias ethics privacy toxicity}' &&
      export GPU_IDS='${GPU_IDS-0 1 2 3 4}' &&
      export STAGGER_SECONDS='${STAGGER_SECONDS-10}' &&
      export LOG_DIR='${LOG_DIR-}' &&
      export DATASET='${DATASET-}' &&
      export BATCH_SIZE='${BATCH_SIZE-}' &&
      export POSITIONS='${POSITIONS-}' &&
      export GREEDY_DECODING='${GREEDY_DECODING-}' &&
      export PROMPT_TYPES='${PROMPT_TYPES-}' &&
      export DATASET_FILE='${DATASET_FILE-}' &&
      export ONLY_SHORT='${ONLY_SHORT-}' &&
      export MAX_SAMPLES='${MAX_SAMPLES-}' &&
      export N_GENERATIONS='${N_GENERATIONS-}' &&
      export MAX_TOKENS='${MAX_TOKENS-}' &&
      export TEMPERATURE='${TEMPERATURE-}' &&
      export PII_INDICES='${PII_INDICES-}' &&
      export DATA_FILE='${DATA_FILE-}' &&
      export NUM_PROMPTS='${NUM_PROMPTS-}' &&
      export CONTEXT_EXAMPLES='${CONTEXT_EXAMPLES-}' &&
      export RUN_STATISTICS='${RUN_STATISTICS-}' &&
      export DATASETS='${DATASETS-}' &&
      export PROMPTS='${PROMPTS-}' &&
      export DATASET_ROOT='${DATASET_ROOT-}' &&
      export RUN_ANALYSIS='${RUN_ANALYSIS-}' &&
      export ANALYSIS_BATCH_SIZE='${ANALYSIS_BATCH_SIZE-}' &&
      export DETOXIFY_MODEL='${DETOXIFY_MODEL-}' &&
      export ANALYSIS_DEVICE='${ANALYSIS_DEVICE-}' &&
      export TRUTH_DATASET='${TRUTH_DATASET-}' &&
      export TRUTH_BATCH_SIZE='${TRUTH_BATCH_SIZE-}' &&
      export TRUTH_POSITIONS='${TRUTH_POSITIONS-}' &&
      export TRUTH_GREEDY_DECODING='${TRUTH_GREEDY_DECODING-}' &&
      export BIAS_DATASET='${BIAS_DATASET-}' &&
      export BIAS_BATCH_SIZE='${BIAS_BATCH_SIZE-}' &&
      export BIAS_POSITIONS='${BIAS_POSITIONS-}' &&
      export BIAS_GREEDY_DECODING='${BIAS_GREEDY_DECODING-}' &&
      export ETHICS_BATCH_SIZE='${ETHICS_BATCH_SIZE-}' &&
      export ETHICS_POSITIONS='${ETHICS_POSITIONS-}' &&
      export ETHICS_GREEDY_DECODING='${ETHICS_GREEDY_DECODING-}' &&
      export ETHICS_PROMPT_TYPES='${ETHICS_PROMPT_TYPES-}' &&
      export ETHICS_DATASET_FILE='${ETHICS_DATASET_FILE-}' &&
      export ETHICS_ONLY_SHORT='${ETHICS_ONLY_SHORT-}' &&
      export ETHICS_MAX_SAMPLES='${ETHICS_MAX_SAMPLES-}' &&
      export ETHICS_N_GENERATIONS='${ETHICS_N_GENERATIONS-}' &&
      export ETHICS_MAX_TOKENS='${ETHICS_MAX_TOKENS-}' &&
      export ETHICS_TEMPERATURE='${ETHICS_TEMPERATURE-}' &&
      export PRIVACY_BATCH_SIZE='${PRIVACY_BATCH_SIZE-}' &&
      export PRIVACY_POSITIONS='${PRIVACY_POSITIONS-}' &&
      export PRIVACY_GREEDY_DECODING='${PRIVACY_GREEDY_DECODING-}' &&
      export PRIVACY_PROMPT_TYPES='${PRIVACY_PROMPT_TYPES-}' &&
      export PRIVACY_PII_INDICES='${PRIVACY_PII_INDICES-}' &&
      export PRIVACY_DATA_FILE='${PRIVACY_DATA_FILE-}' &&
      export PRIVACY_NUM_PROMPTS='${PRIVACY_NUM_PROMPTS-}' &&
      export PRIVACY_CONTEXT_EXAMPLES='${PRIVACY_CONTEXT_EXAMPLES-}' &&
      export PRIVACY_N_GENERATIONS='${PRIVACY_N_GENERATIONS-}' &&
      export PRIVACY_MAX_TOKENS='${PRIVACY_MAX_TOKENS-}' &&
      export PRIVACY_TEMPERATURE='${PRIVACY_TEMPERATURE-}' &&
      export PRIVACY_RUN_STATISTICS='${PRIVACY_RUN_STATISTICS-}' &&
      export TOXICITY_BATCH_SIZE='${TOXICITY_BATCH_SIZE-}' &&
      export TOXICITY_POSITIONS='${TOXICITY_POSITIONS-}' &&
      export TOXICITY_GREEDY_DECODING='${TOXICITY_GREEDY_DECODING-}' &&
      export TOXICITY_DATASETS='${TOXICITY_DATASETS-}' &&
      export TOXICITY_PROMPTS='${TOXICITY_PROMPTS-}' &&
      export TOXICITY_DATASET_ROOT='${TOXICITY_DATASET_ROOT-}' &&
      export TOXICITY_MAX_SAMPLES='${TOXICITY_MAX_SAMPLES-}' &&
      export TOXICITY_N_GENERATIONS='${TOXICITY_N_GENERATIONS-}' &&
      export TOXICITY_MAX_TOKENS='${TOXICITY_MAX_TOKENS-}' &&
      export TOXICITY_TEMPERATURE='${TOXICITY_TEMPERATURE-}' &&
      export TOXICITY_RUN_ANALYSIS='${TOXICITY_RUN_ANALYSIS-}' &&
      export TOXICITY_ANALYSIS_BATCH_SIZE='${TOXICITY_ANALYSIS_BATCH_SIZE-}' &&
      export TOXICITY_DETOXIFY_MODEL='${TOXICITY_DETOXIFY_MODEL-}' &&
      export TOXICITY_ANALYSIS_DEVICE='${TOXICITY_ANALYSIS_DEVICE-}' &&
      bash multi_train/script/evaluate_trained_router_all.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Trained-router evaluation job finished."
echo "Host log: ${HOST_LOG_FILE}"
