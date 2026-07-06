#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK_SCRIPT="${SCRIPT_DIR}/evaluate_checkpoint_task.sh"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_MODE="${MODEL_MODE:-auto}"
BASE_MODEL="${BASE_MODEL:-}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-}"
GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4 5 6 7}"
PARALLEL_CHECKPOINTS="${PARALLEL_CHECKPOINTS:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-5}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
TASKS_VALUE="${TASKS:-truth bias ethics toxicity}"

ONLY_CHECKPOINTS_DEFAULT="${ONLY_CHECKPOINTS_DEFAULT:-}"
ONLY_CHECKPOINTS_VALUE="${ONLY_CHECKPOINTS:-${ONLY_CHECKPOINTS_DEFAULT}}"

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  echo "${value}"
}

is_checkpoint_like_name() {
  local name="${1,,}"
  [[ "${name}" == checkpoint-* || "${name}" == chpoint-* || "${name}" == ckpt-* ]]
}

normalize_checkpoint_root() {
  local path_value="${1%/}"
  echo "${path_value}"
}

model_label_from_path() {
  local path_value="${1%/}"
  if [[ "${path_value}" == */snapshots/* ]]; then
    basename "${path_value%/snapshots/*}"
  else
    basename "${path_value}"
  fi
}

target_name_from_path() {
  local path_value="${1%/}"
  local leaf
  local parent
  leaf="$(basename "${path_value}")"
  parent="$(basename "$(dirname "${path_value}")")"
  if is_checkpoint_like_name "${leaf}" && [[ -n "${parent}" && "${parent}" != "." && "${parent}" != "/" ]]; then
    echo "${parent}-${leaf}"
  elif [[ "${path_value}" == */snapshots/* ]]; then
    echo "$(model_label_from_path "${path_value}")-base"
  else
    echo "${leaf}"
  fi
}

infer_checkpoint_mode() {
  local checkpoint="$1"
  if [[ -f "${checkpoint}/adapter_config.json" || -f "${checkpoint}/adapter_model.safetensors" || -f "${checkpoint}/adapter_model.bin" ]]; then
    echo "lora"
  elif [[ -f "${checkpoint}/config.json" || -f "${checkpoint}/model.safetensors.index.json" || -f "${checkpoint}/pytorch_model.bin" ]] || compgen -G "${checkpoint}/model-*.safetensors" >/dev/null; then
    echo "base"
  else
    echo "unknown"
  fi
}

resolve_checkpoint_mode() {
  local checkpoint="$1"
  case "${MODEL_MODE}" in
    auto)
      infer_checkpoint_mode "${checkpoint}"
      ;;
    base|sft)
      echo "base"
      ;;
    lora)
      echo "lora"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

resolve_base_model_path() {
  local candidate="$1"
  local normalized="${candidate%/}"
  local refs_main
  local snapshot_id
  local repo_root

  if [[ -z "${normalized}" ]]; then
    echo ""
    return 0
  fi

  if [[ -d "${normalized}" && -f "${normalized}/config.json" ]]; then
    echo "${normalized}"
    return 0
  fi

  refs_main="${normalized}/refs/main"
  if [[ -f "${refs_main}" ]]; then
    snapshot_id="$(tr -d '[:space:]' < "${refs_main}")"
    if [[ -n "${snapshot_id}" && -d "${normalized}/snapshots/${snapshot_id}" ]]; then
      echo "${normalized}/snapshots/${snapshot_id}"
      return 0
    fi
  fi

  if [[ "${normalized}" == */snapshots/* ]]; then
    repo_root="${normalized%/snapshots/*}"
    refs_main="${repo_root}/refs/main"
    if [[ -f "${refs_main}" ]]; then
      snapshot_id="$(tr -d '[:space:]' < "${refs_main}")"
      if [[ -n "${snapshot_id}" && -d "${repo_root}/snapshots/${snapshot_id}" ]]; then
        echo "${repo_root}/snapshots/${snapshot_id}"
        return 0
      fi
    fi
  fi

  echo "${normalized}"
}

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"

experiment_run_name_from_model_dir() {
  local path_value="$1"
  path_value="${path_value%/}"
  model_label_from_path "${path_value}"
}

if [[ -z "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR must be set." >&2
  exit 1
fi

MODEL_DIR="$(resolve_base_model_path "$(normalize_checkpoint_root "${MODEL_DIR}")")"
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

read -r -a TASK_ARR <<< "${TASKS_VALUE}"
if [[ "${#TASK_ARR[@]}" -ne 4 ]]; then
  echo "TASKS must contain exactly 4 tasks. Current: ${TASKS_VALUE}" >&2
  exit 1
fi

for task_name in "${TASK_ARR[@]}"; do
  case "${task_name}" in
    truth|bias|ethics|toxicity)
      ;;
    *)
      echo "Unsupported task in TASKS: ${task_name}" >&2
      exit 1
      ;;
  esac
done

read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -lt 4 ]]; then
  echo "Need at least 4 GPU ids in GPU_IDS. Current: ${GPU_IDS_VALUE}" >&2
  exit 1
fi

GPU_BLOCK_SIZE="${#TASK_ARR[@]}"
MAX_PARALLEL_CHECKPOINTS=$(( ${#GPU_IDS_ARR[@]} / GPU_BLOCK_SIZE ))
if [[ "${MAX_PARALLEL_CHECKPOINTS}" -lt 1 ]]; then
  echo "GPU_IDS=${GPU_IDS_VALUE} does not provide enough GPUs for ${GPU_BLOCK_SIZE} tasks." >&2
  exit 1
fi

if [[ -z "${PARALLEL_CHECKPOINTS}" ]]; then
  PARALLEL_CHECKPOINTS="${MAX_PARALLEL_CHECKPOINTS}"
elif [[ "${PARALLEL_CHECKPOINTS}" -gt "${MAX_PARALLEL_CHECKPOINTS}" ]]; then
  echo "PARALLEL_CHECKPOINTS=${PARALLEL_CHECKPOINTS} exceeds max ${MAX_PARALLEL_CHECKPOINTS} for GPU_IDS=${GPU_IDS_VALUE}." >&2
  exit 1
fi

MODEL_RUN_NAME="$(experiment_run_name_from_model_dir "${MODEL_DIR}")"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-multi_train/logs/checkpoint_multi_runs/${MODEL_RUN_NAME}/${RUN_TS}}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_ROOT}/task_logs}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/task_outputs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXPERIMENT_ROOT}/summaries}"
CHECKPOINT_SUMMARY_DIR="${CHECKPOINT_SUMMARY_DIR:-${SUMMARY_ROOT}/checkpoints}"
GLOBAL_SUMMARY_DIR="${GLOBAL_SUMMARY_DIR:-${SUMMARY_ROOT}/global}"

mkdir -p "${LOG_DIR}"
mkdir -p "${TASK_OUTPUT_ROOT}"
mkdir -p "${CHECKPOINT_SUMMARY_DIR}"
mkdir -p "${GLOBAL_SUMMARY_DIR}"

CHECKPOINTS=()
if is_checkpoint_like_name "$(basename "${MODEL_DIR}")"; then
  CHECKPOINTS+=("${MODEL_DIR}")
else
  CHECKPOINTS_RAW=$(
    {
      ls -d "${MODEL_DIR}"/checkpoint-* 2>/dev/null || true
      ls -d "${MODEL_DIR}"/chpoint-* 2>/dev/null || true
      ls -d "${MODEL_DIR}"/ckpt-* 2>/dev/null || true
    } | awk '!seen[$0]++' | sort -V
  )
  if [[ -n "${CHECKPOINTS_RAW}" ]]; then
    while IFS= read -r cp; do
      [[ -n "${cp}" ]] && CHECKPOINTS+=("${cp}")
    done <<< "${CHECKPOINTS_RAW}"
  else
    model_dir_mode="$(resolve_checkpoint_mode "${MODEL_DIR}")"
    if [[ "${model_dir_mode}" == "base" ]]; then
      CHECKPOINTS+=("${MODEL_DIR}")
    fi
  fi
fi

if [[ -n "${ONLY_CHECKPOINTS_VALUE}" ]]; then
  read -r -a ONLY_CHECKPOINTS_ARR <<< "${ONLY_CHECKPOINTS_VALUE}"
  FILTERED=()
  for cp in "${CHECKPOINTS[@]-}"; do
    cp_name="$(basename "${cp}")"
    for wanted in "${ONLY_CHECKPOINTS_ARR[@]}"; do
      if [[ "${cp_name}" == "${wanted}" ]]; then
        FILTERED+=("${cp}")
        break
      fi
    done
  done
  CHECKPOINTS=("${FILTERED[@]}")
fi

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under ${MODEL_DIR}." >&2
  exit 1
fi

for checkpoint_path in "${CHECKPOINTS[@]}"; do
  resolved_mode="$(resolve_checkpoint_mode "${checkpoint_path}")"
  if [[ "${resolved_mode}" == "unknown" ]]; then
    echo "Unable to infer checkpoint mode for ${checkpoint_path}. Set MODEL_MODE explicitly." >&2
    exit 1
  fi
  if [[ "${resolved_mode}" == "lora" && -z "${BASE_MODEL}" ]]; then
    echo "BASE_MODEL must be set because ${checkpoint_path} is a LoRA checkpoint." >&2
    exit 1
  fi
done

GLOBAL_SUMMARY_FILE="${GLOBAL_SUMMARY_DIR}/${MODEL_RUN_NAME}_global_summary_${RUN_TS}.json"
GROUP_PIDS=()
ACTIVE_PIDS=()
ACTIVE_GPUS=()

cleanup_children() {
  for pid in "${GROUP_PIDS[@]-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup_children INT TERM

task_status_file_for() {
  local checkpoint_path="$1"
  local task_name="$2"
  local target_name_safe
  target_name_safe="$(sanitize_name "$(target_name_from_path "${checkpoint_path}")")"
  echo "${LOG_DIR}/${target_name_safe}/${task_name}.status"
}

task_summary_file_for() {
  local checkpoint_path="$1"
  local task_name="$2"
  local target_name_safe
  target_name_safe="$(sanitize_name "$(target_name_from_path "${checkpoint_path}")")"
  echo "${CHECKPOINT_SUMMARY_DIR}/${target_name_safe}/${task_name}_${RUN_TS}.json"
}

checkpoint_summary_file_for() {
  local checkpoint_path="$1"
  local target_name_safe
  target_name_safe="$(sanitize_name "$(target_name_from_path "${checkpoint_path}")")"
  echo "${CHECKPOINT_SUMMARY_DIR}/${target_name_safe}/summary_${RUN_TS}.json"
}

checkpoint_group_log_dir_for() {
  local checkpoint_path="$1"
  local target_name_safe
  target_name_safe="$(sanitize_name "$(target_name_from_path "${checkpoint_path}")")"
  echo "${LOG_DIR}/${target_name_safe}"
}

first_free_gpu() {
  local candidate
  local active_gpu
  for candidate in "${GPU_IDS_ARR[@]}"; do
    local in_use=0
    for active_gpu in "${ACTIVE_GPUS[@]-}"; do
      if [[ "${candidate}" == "${active_gpu}" ]]; then
        in_use=1
        break
      fi
    done
    if [[ "${in_use}" -eq 0 ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

refresh_active_jobs() {
  local running
  local new_pids=()
  local new_gpus=()
  local idx

  running="$(jobs -pr || true)"
  for idx in "${!ACTIVE_PIDS[@]}"; do
    local pid="${ACTIVE_PIDS[$idx]}"
    if grep -qx "${pid}" <<< "${running}"; then
      new_pids+=("${pid}")
      new_gpus+=("${ACTIVE_GPUS[$idx]}")
    fi
  done
  ACTIVE_PIDS=("${new_pids[@]}")
  ACTIVE_GPUS=("${new_gpus[@]}")
}

launch_task_run() {
  local checkpoint_path="$1"
  local task_name="$2"
  local gpu_id="$3"
  local checkpoint_mode
  local target_name
  local target_name_safe
  local group_log_dir
  local task_log
  local task_summary
  local task_status_file

  checkpoint_mode="$(resolve_checkpoint_mode "${checkpoint_path}")"
  target_name="$(target_name_from_path "${checkpoint_path}")"
  target_name_safe="$(sanitize_name "${target_name}")"
  group_log_dir="$(checkpoint_group_log_dir_for "${checkpoint_path}")"
  task_log="${group_log_dir}/${task_name}.log"
  task_summary="$(task_summary_file_for "${checkpoint_path}" "${task_name}")"
  task_status_file="$(task_status_file_for "${checkpoint_path}" "${task_name}")"

  mkdir -p "${group_log_dir}"
  mkdir -p "$(dirname "${task_summary}")"

  echo "[${target_name}/${task_name}] gpu=${gpu_id} log=${task_log} summary=${task_summary}"

  (
    env \
      PATH="${PATH}" \
      HOME="${HOME:-}" \
      LANG="${LANG:-}" \
      LC_ALL="${LC_ALL:-}" \
      PYTHONPATH="${PYTHONPATH:-}" \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}" \
      TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-}" \
      HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      TASK="${task_name}" \
      MODEL_MODE="${checkpoint_mode}" \
      MODEL_PATH="${checkpoint_path}" \
      BASE_MODEL="${BASE_MODEL}" \
      TARGET_LAYERS="${TARGET_LAYERS}" \
      SUBSPACE_RANK="${SUBSPACE_RANK}" \
      DEVICE="cuda:${gpu_id}" \
      SUMMARY_PATH="${task_summary}" \
      RESULT_PREFIX="${target_name_safe}" \
      CHECKPOINT_NAME="${target_name_safe}" \
      TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT}" \
      DATASET="${DATASET:-}" \
      BATCH_SIZE="${BATCH_SIZE:-}" \
      POSITIONS="${POSITIONS:-}" \
      GREEDY_DECODING="${GREEDY_DECODING:-}" \
      PROMPT_TYPES="${PROMPT_TYPES:-}" \
      DATASET_FILE="${DATASET_FILE:-}" \
      ONLY_SHORT="${ONLY_SHORT:-}" \
      MAX_SAMPLES="${MAX_SAMPLES:-}" \
      N_GENERATIONS="${N_GENERATIONS:-}" \
      MAX_TOKENS="${MAX_TOKENS:-}" \
      TEMPERATURE="${TEMPERATURE:-}" \
      DATASETS="${DATASETS:-}" \
      PROMPTS="${PROMPTS:-}" \
      DATASET_ROOT="${DATASET_ROOT:-}" \
      RUN_ANALYSIS="${RUN_ANALYSIS:-}" \
      ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-}" \
      DETOXIFY_MODEL="${DETOXIFY_MODEL:-}" \
      ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-}" \
      ALLOW_ANALYSIS_FALLBACK="${ALLOW_ANALYSIS_FALLBACK:-0}" \
      TRUTH_DATASET="${TRUTH_DATASET:-}" \
      TRUTH_BATCH_SIZE="${TRUTH_BATCH_SIZE:-}" \
      TRUTH_POSITIONS="${TRUTH_POSITIONS:-}" \
      TRUTH_GREEDY_DECODING="${TRUTH_GREEDY_DECODING:-}" \
      BIAS_DATASET="${BIAS_DATASET:-}" \
      BIAS_BATCH_SIZE="${BIAS_BATCH_SIZE:-}" \
      BIAS_POSITIONS="${BIAS_POSITIONS:-}" \
      BIAS_GREEDY_DECODING="${BIAS_GREEDY_DECODING:-}" \
      ETHICS_BATCH_SIZE="${ETHICS_BATCH_SIZE:-}" \
      ETHICS_POSITIONS="${ETHICS_POSITIONS:-}" \
      ETHICS_GREEDY_DECODING="${ETHICS_GREEDY_DECODING:-}" \
      ETHICS_PROMPT_TYPES="${ETHICS_PROMPT_TYPES:-}" \
      ETHICS_DATASET_FILE="${ETHICS_DATASET_FILE:-}" \
      ETHICS_ONLY_SHORT="${ETHICS_ONLY_SHORT:-}" \
      ETHICS_MAX_SAMPLES="${ETHICS_MAX_SAMPLES:-}" \
      ETHICS_N_GENERATIONS="${ETHICS_N_GENERATIONS:-}" \
      ETHICS_MAX_TOKENS="${ETHICS_MAX_TOKENS:-}" \
      ETHICS_TEMPERATURE="${ETHICS_TEMPERATURE:-}" \
      TOXICITY_BATCH_SIZE="${TOXICITY_BATCH_SIZE:-}" \
      TOXICITY_POSITIONS="${TOXICITY_POSITIONS:-}" \
      TOXICITY_GREEDY_DECODING="${TOXICITY_GREEDY_DECODING:-}" \
      TOXICITY_DATASETS="${TOXICITY_DATASETS:-}" \
      TOXICITY_PROMPTS="${TOXICITY_PROMPTS:-}" \
      TOXICITY_DATASET_ROOT="${TOXICITY_DATASET_ROOT:-}" \
      TOXICITY_MAX_SAMPLES="${TOXICITY_MAX_SAMPLES:-}" \
      TOXICITY_N_GENERATIONS="${TOXICITY_N_GENERATIONS:-}" \
      TOXICITY_MAX_TOKENS="${TOXICITY_MAX_TOKENS:-}" \
      TOXICITY_TEMPERATURE="${TOXICITY_TEMPERATURE:-}" \
      TOXICITY_RUN_ANALYSIS="${TOXICITY_RUN_ANALYSIS:-}" \
      TOXICITY_ALLOW_ANALYSIS_FALLBACK="${TOXICITY_ALLOW_ANALYSIS_FALLBACK:-0}" \
      TOXICITY_ANALYSIS_BATCH_SIZE="${TOXICITY_ANALYSIS_BATCH_SIZE:-}" \
      TOXICITY_DETOXIFY_MODEL="${TOXICITY_DETOXIFY_MODEL:-}" \
      TOXICITY_ANALYSIS_DEVICE="${TOXICITY_ANALYSIS_DEVICE:-}" \
      bash "${TASK_SCRIPT}"
  ) >"${task_log}" 2>&1
  task_status=$?
  printf '%s\n' "${task_status}" > "${task_status_file}"
  exit "${task_status}"
}

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "MODEL_MODE=${MODEL_MODE}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "TASKS=${TASKS_VALUE}"
echo "GPU_IDS=${GPU_IDS_VALUE}"
echo "PARALLEL_CHECKPOINTS=${PARALLEL_CHECKPOINTS}"
echo "RUN_TS=${RUN_TS}"
echo "STAGGER_SECONDS=${STAGGER_SECONDS}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "SUBSPACE_RANK=${SUBSPACE_RANK}"
echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "LOG_DIR=${LOG_DIR}"
echo "TASK_OUTPUT_ROOT=${TASK_OUTPUT_ROOT}"
echo "CHECKPOINT_SUMMARY_DIR=${CHECKPOINT_SUMMARY_DIR}"
echo "GLOBAL_SUMMARY_DIR=${GLOBAL_SUMMARY_DIR}"

QUEUE_CHECKPOINTS=()
QUEUE_TASKS=()
for checkpoint_path in "${CHECKPOINTS[@]}"; do
  for task_name in "${TASK_ARR[@]}"; do
    QUEUE_CHECKPOINTS+=("${checkpoint_path}")
    QUEUE_TASKS+=("${task_name}")
  done
done

queue_idx=0
while [[ "${queue_idx}" -lt "${#QUEUE_CHECKPOINTS[@]}" || "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  while [[ "${queue_idx}" -lt "${#QUEUE_CHECKPOINTS[@]}" && "${#ACTIVE_PIDS[@]}" -lt "${#GPU_IDS_ARR[@]}" ]]; do
    checkpoint_path="${QUEUE_CHECKPOINTS[$queue_idx]}"
    task_name="${QUEUE_TASKS[$queue_idx]}"
    gpu_id="$(first_free_gpu)"

    launch_task_run "${checkpoint_path}" "${task_name}" "${gpu_id}" &
    pid=$!
    GROUP_PIDS+=("${pid}")
    ACTIVE_PIDS+=("${pid}")
    ACTIVE_GPUS+=("${gpu_id}")
    queue_idx=$((queue_idx + 1))

    if [[ "${STAGGER_SECONDS}" -gt 0 && "${queue_idx}" -lt "${#QUEUE_CHECKPOINTS[@]}" ]]; then
      sleep "${STAGGER_SECONDS}"
    fi
  done

  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    wait -n || true
    refresh_active_jobs
  fi
done

SUCCESS_SUMMARY_FILES=()
FAILED_RUNS=()
for checkpoint_path in "${CHECKPOINTS[@]}"; do
  target_name="$(target_name_from_path "${checkpoint_path}")"
  target_name_safe="$(sanitize_name "${target_name}")"
  group_log_dir="$(checkpoint_group_log_dir_for "${checkpoint_path}")"
  group_status_file="${group_log_dir}/group.status"
  checkpoint_summary_file="$(checkpoint_summary_file_for "${checkpoint_path}")"
  checkpoint_mode="$(resolve_checkpoint_mode "${checkpoint_path}")"

  SUCCESS_TASK_SUMMARIES=()
  FAILED_TASKS=()
  checkpoint_status=0

  for task_name in "${TASK_ARR[@]}"; do
    task_status_file="$(task_status_file_for "${checkpoint_path}" "${task_name}")"
    task_summary_file="$(task_summary_file_for "${checkpoint_path}" "${task_name}")"

    task_status=1
    if [[ -f "${task_status_file}" ]]; then
      task_status="$(tr -d '[:space:]' < "${task_status_file}")"
    fi

    if [[ "${task_status}" == "0" && -f "${task_summary_file}" ]]; then
      SUCCESS_TASK_SUMMARIES+=("${task_summary_file}")
    else
      FAILED_TASKS+=("${task_name}")
      checkpoint_status=1
    fi
  done

  python3 - "${checkpoint_summary_file}" "${RUN_TS}" "${MODEL_DIR}" "${checkpoint_path}" "${target_name}" "${checkpoint_mode}" "${BASE_MODEL}" "${FAILED_TASKS[*]-}" "${SUCCESS_TASK_SUMMARIES[@]-}" <<'PY'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
model_dir = sys.argv[3]
checkpoint_path = sys.argv[4]
target_name = sys.argv[5]
checkpoint_mode = sys.argv[6]
base_model = sys.argv[7]
failed_runs = [x for x in sys.argv[8].split() if x] if len(sys.argv) > 8 else []
summary_files = [item for item in sys.argv[9:] if item] if len(sys.argv) > 9 else []

task_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        task_runs.append(json.load(f))

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "model_dir": model_dir,
    "checkpoint_path": checkpoint_path,
    "target_name": target_name,
    "checkpoint_mode": checkpoint_mode,
    "base_model": base_model or None,
    "failed_runs": failed_runs,
    "task_runs": task_runs,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
PY

  printf '%s\n' "${checkpoint_status}" > "${group_status_file}"

  if [[ "${checkpoint_status}" == "0" && -f "${checkpoint_summary_file}" ]]; then
    SUCCESS_SUMMARY_FILES+=("${checkpoint_summary_file}")
  else
    FAILED_RUNS+=("${target_name_safe}")
  fi
done

python3 - "${GLOBAL_SUMMARY_FILE}" "${RUN_TS}" "${MODEL_DIR}" "${MODEL_RUN_NAME}" "${MODEL_MODE}" "${BASE_MODEL}" "${GPU_IDS_VALUE}" "${FAILED_RUNS[*]-}" "${SUCCESS_SUMMARY_FILES[@]-}" <<'PY'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
model_dir = sys.argv[3]
model_run_name = sys.argv[4]
model_mode = sys.argv[5]
base_model = sys.argv[6]
gpu_ids = [x for x in sys.argv[7].split() if x]
failed_runs = [x for x in sys.argv[8].split() if x] if len(sys.argv) > 8 else []
summary_files = [item for item in sys.argv[9:] if item] if len(sys.argv) > 9 else []

checkpoint_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        checkpoint_runs.append(json.load(f))

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "model_dir": model_dir,
    "model_run_name": model_run_name,
    "model_mode": model_mode,
    "base_model": base_model or None,
    "gpu_ids": gpu_ids,
    "failed_runs": failed_runs,
    "checkpoint_runs": checkpoint_runs,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
PY

echo "Global summary saved to: ${GLOBAL_SUMMARY_FILE}"
if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo "Some checkpoint runs failed: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
