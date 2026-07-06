#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

experiment_run_name_from_model_dir() {
  local path_value="$1"
  path_value="${path_value%/}"
  if [[ "${path_value}" == */snapshots/* ]]; then
    basename "${path_value%/snapshots/*}"
  else
    basename "${path_value}"
  fi
}

PARTITION="${PARTITION:-eailab_os}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/hwfile/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR must be set." >&2
  exit 1
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$(experiment_run_name_from_model_dir "${MODEL_DIR}")}"
EXPERIMENT_ROOT_REL="${EXPERIMENT_ROOT_REL:-multi_train/logs/checkpoint_multi_runs/${EXPERIMENT_RUN_NAME}/${RUN_TS}}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/${EXPERIMENT_ROOT_REL}/submit_logs}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/submit.log}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "EXPERIMENT_RUN_NAME=${EXPERIMENT_RUN_NAME}"
echo "EXPERIMENT_ROOT_REL=${EXPERIMENT_ROOT_REL}"
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
      export MODEL_DIR='${MODEL_DIR}' &&
      export MODEL_MODE='${MODEL_MODE-auto}' &&
      export BASE_MODEL='${BASE_MODEL-}' &&
      export EXPERIMENT_ROOT='${EXPERIMENT_ROOT_REL}' &&
      export MERGE_SUMMARY_PATH='${MERGE_SUMMARY_PATH-}' &&
      export SUMMARY_DIR='${SUMMARY_DIR-}' &&
      export GPU_IDS='${GPU_IDS-0 1 2 3 4 5 6 7}' &&
      export PARALLEL_CHECKPOINTS='${PARALLEL_CHECKPOINTS-}' &&
      export STAGGER_SECONDS='${STAGGER_SECONDS-5}' &&
      export RUN_TS='${RUN_TS}' &&
      export LOG_DIR='${LOG_DIR-}' &&
      export TARGET_LAYERS='${TARGET_LAYERS--1}' &&
      export SUBSPACE_RANK='${SUBSPACE_RANK-8}' &&
      export ONLY_CHECKPOINTS='${ONLY_CHECKPOINTS-}' &&
      export ONLY_CHECKPOINTS_DEFAULT='${ONLY_CHECKPOINTS_DEFAULT-}' &&
      export TASKS='${TASKS-truth bias ethics toxicity}' &&
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
      export DATASETS='${DATASETS-}' &&
      export PROMPTS='${PROMPTS-}' &&
      export DATASET_ROOT='${DATASET_ROOT-}' &&
      export RUN_ANALYSIS='${RUN_ANALYSIS-}' &&
      export ALLOW_ANALYSIS_FALLBACK='${ALLOW_ANALYSIS_FALLBACK-0}' &&
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
      export TOXICITY_ALLOW_ANALYSIS_FALLBACK='${TOXICITY_ALLOW_ANALYSIS_FALLBACK-0}' &&
      export TOXICITY_ANALYSIS_BATCH_SIZE='${TOXICITY_ANALYSIS_BATCH_SIZE-}' &&
      export TOXICITY_DETOXIFY_MODEL='${TOXICITY_DETOXIFY_MODEL-}' &&
      export TOXICITY_ANALYSIS_DEVICE='${TOXICITY_ANALYSIS_DEVICE-}' &&
      bash multi_train/script/evaluate_checkpoint_all.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Checkpoint evaluation job finished."
echo "Host log: ${HOST_LOG_FILE}"
