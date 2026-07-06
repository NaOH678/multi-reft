#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

model_key_from_path() {
  local path_value="$1"
  path_value="${path_value%/}"
  if [[ "${path_value}" == *"/snapshots/"* ]]; then
    basename "$(dirname "$(dirname "${path_value}")")"
  else
    basename "${path_value}"
  fi
}

experiment_run_name_from_checkpoint_dir() {
  local path_value="$1"
  path_value="${path_value%/}"
  local leaf
  local parent
  leaf="$(basename "${path_value}")"
  parent="$(basename "$(dirname "${path_value}")")"
  if [[ "${leaf,,}" == checkpoint-* || "${leaf,,}" == chpoint-* || "${leaf,,}" == ckpt-* ]]; then
    echo "${parent}-${leaf}"
  else
    echo "${leaf}"
  fi
}

infer_specialist_task_from_text() {
  local raw_value="${1,,}"
  if [[ "${raw_value}" == *"truthful"* || "${raw_value}" == *"truth"* ]]; then
    echo "truth"
  elif [[ "${raw_value}" == *"stereotype"* || "${raw_value}" == *"bias"* || "${raw_value}" == *"bbq"* ]]; then
    echo "bias"
  elif [[ "${raw_value}" == *"moral"* || "${raw_value}" == *"ethic"* ]]; then
    echo "ethics"
  elif [[ "${raw_value}" == *"toxicity"* || "${raw_value}" == *"toxic"* ]]; then
    echo "toxicity"
  else
    return 1
  fi
}

infer_specialist_task() {
  local explicit_task="$1"
  local checkpoint_dir="$2"
  local train_run_name="$3"

  if [[ -n "${explicit_task}" ]]; then
    echo "${explicit_task}"
    return 0
  fi

  if infer_specialist_task_from_text "${train_run_name}" >/dev/null 2>&1; then
    infer_specialist_task_from_text "${train_run_name}"
    return 0
  fi

  if infer_specialist_task_from_text "${checkpoint_dir}" >/dev/null 2>&1; then
    infer_specialist_task_from_text "${checkpoint_dir}"
    return 0
  fi

  return 1
}

PARTITION="${PARTITION:-eailab_os}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/hwfile/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"
SPECIALIST_SUMMARY_JSON="${SPECIALIST_SUMMARY_JSON:-}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"

if [[ -n "${SPECIALIST_SUMMARY_JSON}" ]]; then
  SUMMARY_EXPORTS="$(
    python3 - "${SPECIALIST_SUMMARY_JSON}" "${BASE_MODEL-}" "${CHECKPOINT_DIRS-}" "${CHECKPOINT_DIR-}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
base_model_in = sys.argv[2]
checkpoint_dirs_in = sys.argv[3]
checkpoint_dir_in = sys.argv[4]

if not summary_path.exists():
    raise FileNotFoundError(summary_path)

with summary_path.open("r", encoding="utf-8") as f:
    payload = json.load(f)

base_model = base_model_in or payload.get("base_model") or ""
specialists = payload.get("reft_specialists") or []
checkpoint_dirs = checkpoint_dirs_in
if not checkpoint_dirs:
    roots = []
    for item in specialists:
        p = Path(str(item).rstrip("/"))
        roots.append(str(p.parent if p.name == "intervenable_model" else p))
    checkpoint_dirs = " ".join(roots)

checkpoint_dir = checkpoint_dir_in
if not checkpoint_dir and checkpoint_dirs:
    checkpoint_dir = checkpoint_dirs.split()[0]

print(f"BASE_MODEL={shlex.quote(base_model)}")
print(f"CHECKPOINT_DIRS={shlex.quote(checkpoint_dirs)}")
print(f"CHECKPOINT_DIR={shlex.quote(checkpoint_dir)}")
PY
  )"
  eval "${SUMMARY_EXPORTS}"
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_BASE_MODEL="${BASE_MODEL-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
DEFAULT_CHECKPOINT_DIR="${CHECKPOINT_DIR-unknown_checkpoint_dir}"
DEFAULT_CHECKPOINT_DIRS="${CHECKPOINT_DIRS-}"
EXPERIMENT_MODEL_KEY="${EXPERIMENT_MODEL_KEY:-$(model_key_from_path "${DEFAULT_BASE_MODEL}")}"
if [[ -n "${DEFAULT_CHECKPOINT_DIRS}" ]]; then
  EXPERIMENT_RUN_NAME="${EXPERIMENT_RUN_NAME:-multi_specialists_bundle}"
  EXPERIMENT_TASK_NAME="${EXPERIMENT_TASK_NAME:-cross_bundle}"
else
  EXPERIMENT_RUN_NAME="${EXPERIMENT_RUN_NAME:-$(experiment_run_name_from_checkpoint_dir "${DEFAULT_CHECKPOINT_DIR}")}"
  EXPERIMENT_TASK_NAME="${EXPERIMENT_TASK_NAME:-$(infer_specialist_task "${SPECIALIST_TASK-}" "${DEFAULT_CHECKPOINT_DIR}" "${EXPERIMENT_RUN_NAME}" 2>/dev/null || echo unknown_task)}"
fi
if [[ -n "${TASKS-}" && "${EXCLUDE_SPECIALIST_TASK-0}" == "1" && "${EXPERIMENT_TASK_NAME}" != "unknown_task" ]]; then
  EXPERIMENT_TASK_NAME="cross_except_${EXPERIMENT_TASK_NAME}"
fi
EXPERIMENT_ROOT_REL="${EXPERIMENT_ROOT_REL:-multi_train/logs/specialist_single_runs/${EXPERIMENT_MODEL_KEY}/${EXPERIMENT_RUN_NAME}/${EXPERIMENT_TASK_NAME}/${RUN_TS}}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/${EXPERIMENT_ROOT_REL}/submit_logs}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/submit.log}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "EXPERIMENT_MODEL_KEY=${EXPERIMENT_MODEL_KEY}"
echo "EXPERIMENT_RUN_NAME=${EXPERIMENT_RUN_NAME}"
echo "EXPERIMENT_TASK_NAME=${EXPERIMENT_TASK_NAME}"
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
      export BASE_MODEL='${BASE_MODEL-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}' &&
      export CHECKPOINT_DIR='${CHECKPOINT_DIR-}' &&
      export CHECKPOINT_DIRS='${CHECKPOINT_DIRS-}' &&
      export MODEL_MODE='${MODEL_MODE-auto}' &&
      export SPECIALIST_TASK='${SPECIALIST_TASK-}' &&
      export TASKS='${TASKS-}' &&
      export EXCLUDE_SPECIALIST_TASK='${EXCLUDE_SPECIALIST_TASK-0}' &&
      export FLAT_PARALLEL_TASKS='${FLAT_PARALLEL_TASKS-0}' &&
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
      bash multi_train/script/evaluate_specialist_all.sh
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Specialist evaluation job finished."
echo "Host log: ${HOST_LOG_FILE}"
