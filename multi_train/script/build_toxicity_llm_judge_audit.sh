#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITION="${PARTITION:-eailab_os}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM_PER_TASK="${MEM_PER_TASK:-16G}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/build_toxicity_llm_judge_audit_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/build_toxicity_llm_judge_audit.log}"

OURS_CSV="${OURS_CSV:-multi_train/eval_toxicity/data/generations/Llama3-8b-Composable-E6-toxpen07-toxicity-summary_20260504_160408-toxicity_toxicity_scored.csv}"
CAA_CSV="${CAA_CSV:-}"
ITI_CSV="${ITI_CSV:-}"
RePE_CSV="${RePE_CSV:-}"

# SFT_CSV="${SFT_CSV:-multi_train/eval_toxicity/data/generations/llama3-8b-sft-checkpoint-4060-toxicity_toxicity_scored.csv}"
# LORA_CSV="${LORA_CSV:-multi_train/eval_toxicity/data/generations/llama3-8b-lora-checkpoint-1524-toxicity_toxicity_scored.csv}"

OUTPUT_DIR="${OUTPUT_DIR:-multi_train/eval_toxicity/data/llm_judge_audit/${RUN_TAG}}"
HIGH_THRESHOLD="${HIGH_THRESHOLD:-0.8}"
MID_THRESHOLD="${MID_THRESHOLD:-0.5}"
BASELINE_LOW_THRESHOLD="${BASELINE_LOW_THRESHOLD:-0.1}"
RANDOM_PROMPT_COUNT="${RANDOM_PROMPT_COUNT:-200}"
SEED="${SEED:-42}"
PAIRWISE_ONLY="${PAIRWISE_ONLY:-0}"

echo "PARTITION=${PARTITION}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MEM_PER_TASK=${MEM_PER_TASK}"
echo "HOST_LOG_FILE=${HOST_LOG_FILE}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
echo "OURS_CSV=${OURS_CSV}"
echo "CAA_CSV=${CAA_CSV}"
echo "ITI_CSV=${ITI_CSV}"
echo "RePE_CSV=${RePE_CSV}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HIGH_THRESHOLD=${HIGH_THRESHOLD}"
echo "MID_THRESHOLD=${MID_THRESHOLD}"
echo "BASELINE_LOW_THRESHOLD=${BASELINE_LOW_THRESHOLD}"
echo "RANDOM_PROMPT_COUNT=${RANDOM_PROMPT_COUNT}"
echo "SEED=${SEED}"
echo "PAIRWISE_ONLY=${PAIRWISE_ONLY}"

srun \
  -p "${PARTITION}" \
  ${SRUN_EXTRA_ARGS} \
  apptainer exec \
    --cleanenv \
    --bind /mnt:/mnt \
    --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
    "${IMAGE_PATH}" \
    bash -c "
      cd '${WORKSPACE_CONT}' &&
      export PYTHON_BIN='${PYTHON_BIN}' &&
      export OURS_CSV='${OURS_CSV}' &&
      export CAA_CSV='${CAA_CSV}' &&
      export ITI_CSV='${ITI_CSV}' &&
      export RePE_CSV='${RePE_CSV}' &&
      export OUTPUT_DIR='${OUTPUT_DIR}' &&
      export HIGH_THRESHOLD='${HIGH_THRESHOLD}' &&
      export MID_THRESHOLD='${MID_THRESHOLD}' &&
      export BASELINE_LOW_THRESHOLD='${BASELINE_LOW_THRESHOLD}' &&
      export RANDOM_PROMPT_COUNT='${RANDOM_PROMPT_COUNT}' &&
      export SEED='${SEED}' &&
      export PAIRWISE_ONLY='${PAIRWISE_ONLY}' &&
      \"\${PYTHON_BIN}\" multi_train/eval_toxicity/build_llm_judge_audit.py \
        --ours_csv \"\${OURS_CSV}\" \
        \$(if [ -n \"\${CAA_CSV}\" ]; then printf -- '--caa_csv %q ' \"\${CAA_CSV}\"; fi) \
        \$(if [ -n \"\${ITI_CSV}\" ]; then printf -- '--iti_csv %q ' \"\${ITI_CSV}\"; fi) \
        \$(if [ -n \"\${RePE_CSV}\" ]; then printf -- '--repe_csv %q ' \"\${RePE_CSV}\"; fi) \
        --output_dir \"\${OUTPUT_DIR}\" \
        --high_threshold \"\${HIGH_THRESHOLD}\" \
        --mid_threshold \"\${MID_THRESHOLD}\" \
        --baseline_low_threshold \"\${BASELINE_LOW_THRESHOLD}\" \
        --random_prompt_count \"\${RANDOM_PROMPT_COUNT}\" \
        --seed \"\${SEED}\" \
        \$(if [ \"\${PAIRWISE_ONLY}\" = \"1\" ]; then printf -- '--pairwise_only'; fi)
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "LLM judge audit build finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Host log: ${HOST_LOG_FILE}"
