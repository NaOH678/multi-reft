#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-eailab_os}"
GRES="${GRES:-gpu:1}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/eval_perplexity_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/eval_perplexity.log}"

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
      export INPUT_PATH='${INPUT_PATH-multi_train/eval_toxicity/data/generations/Llama3-8b-Composable-E6-toxpen07-toxicity-summary_20260504_160408-toxicity_toxicity_scored.csv}' &&
      export TEXT_COLUMN='${TEXT_COLUMN-output}' &&
      export RESPONSE_COLUMN='${RESPONSE_COLUMN-}' &&
      export PROMPT_COLUMN='${PROMPT_COLUMN-prompt}' &&
      export PROMPT_SUFFIX='${PROMPT_SUFFIX-}' &&
      export MODEL_NAME_OR_PATH='${MODEL_NAME_OR_PATH-models/gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2}' &&
      export BATCH_SIZE='${BATCH_SIZE-128}' &&
      export MAX_SAMPLES='${MAX_SAMPLES-}' &&
      export MAX_LENGTH='${MAX_LENGTH-512}' &&
      export DEVICE='${DEVICE-cuda:0}' &&
      export STRIP_PROMPT_ECHO='${STRIP_PROMPT_ECHO-0}' &&
      export PROMPT_ECHO_MIN_CHARS='${PROMPT_ECHO_MIN_CHARS-32}' &&
      export OUTPUT_JSON='${OUTPUT_JSON-}' &&
      \"\${PYTHON_BIN}\" multi_train/eval_common/compute_perplexity.py \
        --input_path \"\${INPUT_PATH}\" \
        --text_column \"\${TEXT_COLUMN}\" \
        \$(if [ -n \"\${RESPONSE_COLUMN}\" ]; then printf -- '--response_column %q ' \"\${RESPONSE_COLUMN}\"; fi) \
        \$(if [ -n \"\${PROMPT_COLUMN}\" ]; then printf -- '--prompt_column %q ' \"\${PROMPT_COLUMN}\"; fi) \
        \$(if [ -n \"\${PROMPT_SUFFIX}\" ]; then printf -- '--prompt_suffix %q ' \"\${PROMPT_SUFFIX}\"; fi) \
        --model_name_or_path \"\${MODEL_NAME_OR_PATH}\" \
        --batch_size \"\${BATCH_SIZE}\" \
        --max_length \"\${MAX_LENGTH}\" \
        \$(if [ -n \"\${MAX_SAMPLES}\" ]; then printf -- '--max_samples %q ' \"\${MAX_SAMPLES}\"; fi) \
        --device \"\${DEVICE}\" \
        \$(if [ \"\${STRIP_PROMPT_ECHO}\" = \"1\" ]; then printf -- '--strip_prompt_echo '; fi) \
        --prompt_echo_min_chars \"\${PROMPT_ECHO_MIN_CHARS}\" \
        \$(if [ -n \"\${OUTPUT_JSON}\" ]; then printf -- '--output_json %q ' \"\${OUTPUT_JSON}\"; fi)
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Perplexity evaluation job finished."
echo "Host log: ${HOST_LOG_FILE}"
