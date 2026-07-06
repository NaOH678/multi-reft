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
AUDIT_DIR="${AUDIT_DIR:-}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/summarize_toxicity_llm_judge_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/summarize_toxicity_llm_judge.log}"

if [[ -z "${AUDIT_DIR}" ]]; then
  echo "AUDIT_DIR is required." >&2
  exit 1
fi

INPUT_DIR="${AUDIT_DIR}/judge_outputs"
OUTPUT_DIR="${AUDIT_DIR}/judge_summary"

echo "PARTITION=${PARTITION}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CPUS_PER_TASK=${CPUS_PER_TASK}"
echo "MEM_PER_TASK=${MEM_PER_TASK}"
echo "HOST_LOG_FILE=${HOST_LOG_FILE}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
echo "AUDIT_DIR=${AUDIT_DIR}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

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
      export INPUT_DIR='${INPUT_DIR}' &&
      export OUTPUT_DIR='${OUTPUT_DIR}' &&
      \"\${PYTHON_BIN}\" multi_train/eval_toxicity/summarize_llm_judge.py \
        --input_dir \"\${INPUT_DIR}\" \
        --output_dir \"\${OUTPUT_DIR}\"
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Judge summary finished."
echo "Summary dir: ${OUTPUT_DIR}"
echo "Host log: ${HOST_LOG_FILE}"
