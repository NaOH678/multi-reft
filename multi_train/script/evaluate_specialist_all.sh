#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK_SCRIPT="${SCRIPT_DIR}/evaluate_specialist_task.sh"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CHECKPOINT_DIRS_VALUE="${CHECKPOINT_DIRS:-}"
MODEL_MODE="${MODEL_MODE:-auto}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-}"
SPECIALIST_TASK="${SPECIALIST_TASK:-}"
TASKS_VALUE="${TASKS:-}"
EXCLUDE_SPECIALIST_TASK="${EXCLUDE_SPECIALIST_TASK:-0}"
FLAT_PARALLEL_TASKS="${FLAT_PARALLEL_TASKS:-0}"
GPU_IDS_VALUE="${GPU_IDS:-0 1 2 3 4 5 6 7}"
PARALLEL_CHECKPOINTS="${PARALLEL_CHECKPOINTS:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-5}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"

ONLY_CHECKPOINTS_DEFAULT="${ONLY_CHECKPOINTS_DEFAULT:-}"
ONLY_CHECKPOINTS_VALUE="${ONLY_CHECKPOINTS:-${ONLY_CHECKPOINTS_DEFAULT}}"

resolve_base_model_path() {
  local candidate="$1"
  local normalized="${candidate%/}"
  local refs_main
  local snapshot_id
  local repo_root

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

target_name_from_path() {
  local path_value="${1%/}"
  local leaf
  local parent
  leaf="$(basename "${path_value}")"
  parent="$(basename "$(dirname "${path_value}")")"
  if is_checkpoint_like_name "${leaf}" && [[ -n "${parent}" && "${parent}" != "." && "${parent}" != "/" ]]; then
    echo "${parent}-${leaf}"
  else
    echo "${leaf}"
  fi
}

normalize_checkpoint_root() {
  local path_value="${1%/}"
  if [[ "$(basename "${path_value}")" == "intervenable_model" ]]; then
    echo "$(dirname "${path_value}")"
  else
    echo "${path_value}"
  fi
}

infer_checkpoint_mode() {
  local checkpoint="$1"
  if [[ -d "${checkpoint}/intervenable_model" ]]; then
    echo "reft"
  elif [[ -f "${checkpoint}/adapter_config.json" || -f "${checkpoint}/adapter_model.safetensors" || -f "${checkpoint}/adapter_model.bin" ]]; then
    echo "lora"
  else
    echo "base"
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

build_eval_tasks_for_source() {
  local source_task="$1"
  local requested_tasks="$2"
  local exclude_source="$3"
  local -a source_tasks
  local -a filtered_tasks=()

  if [[ -n "${requested_tasks}" ]]; then
    read -r -a source_tasks <<< "${requested_tasks}"
  else
    source_tasks=("${source_task}")
  fi

  for task_name in "${source_tasks[@]}"; do
    case "${task_name}" in
      truth|bias|ethics|toxicity)
        ;;
      *)
        echo "Invalid task in TASKS=${requested_tasks}: ${task_name}" >&2
        return 1
        ;;
    esac
    if [[ "${exclude_source}" == "1" && "${task_name}" == "${source_task}" ]]; then
      continue
    fi
    filtered_tasks+=("${task_name}")
  done

  if [[ "${#filtered_tasks[@]}" -eq 0 ]]; then
    return 0
  fi
  printf '%s\n' "${filtered_tasks[@]}"
}

MODEL_KEY="$(model_key_from_path "${BASE_MODEL}")"
TRAIN_RUN_NAME="$(experiment_run_name_from_checkpoint_dir "${CHECKPOINT_DIR:-unknown_checkpoint_dir}")"

SOURCE_TASK=""
EVAL_TASKS_ARR=()
if [[ -n "${CHECKPOINT_DIRS_VALUE}" ]]; then
  TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-multi_specialists_bundle}"
  EVAL_TASK_LABEL="cross_bundle"
else
  if [[ -z "${CHECKPOINT_DIR}" ]]; then
    echo "CHECKPOINT_DIR must be set unless CHECKPOINT_DIRS is provided." >&2
    exit 1
  fi

  SOURCE_TASK="$(infer_specialist_task "${SPECIALIST_TASK}" "${CHECKPOINT_DIR}" "${TRAIN_RUN_NAME}")" || {
    echo "Unable to infer specialist eval task from CHECKPOINT_DIR=${CHECKPOINT_DIR}." >&2
    echo "Please set SPECIALIST_TASK to one of: truth bias ethics toxicity." >&2
    exit 1
  }

  case "${SOURCE_TASK}" in
    truth|bias|ethics|toxicity)
      ;;
    *)
      echo "Invalid SPECIALIST_TASK=${SOURCE_TASK}, expected one of: truth bias ethics toxicity." >&2
      exit 1
      ;;
  esac

  while IFS= read -r task_name; do
    [[ -n "${task_name}" ]] && EVAL_TASKS_ARR+=("${task_name}")
  done < <(build_eval_tasks_for_source "${SOURCE_TASK}" "${TASKS_VALUE}" "${EXCLUDE_SPECIALIST_TASK}")

  if [[ "${#EVAL_TASKS_ARR[@]}" -eq 0 ]]; then
    echo "No evaluation tasks remain after applying EXCLUDE_SPECIALIST_TASK=${EXCLUDE_SPECIALIST_TASK}." >&2
    exit 1
  fi

  if [[ "${#EVAL_TASKS_ARR[@]}" -eq 1 ]]; then
    EVAL_TASK_LABEL="${EVAL_TASKS_ARR[0]}"
  elif [[ "${EXCLUDE_SPECIALIST_TASK}" == "1" && -n "${TASKS_VALUE}" ]]; then
    EVAL_TASK_LABEL="cross_except_${SOURCE_TASK}"
  else
    EVAL_TASK_LABEL="multi_task"
  fi
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-multi_train/logs/specialist_single_runs/${MODEL_KEY}/${TRAIN_RUN_NAME}/${EVAL_TASK_LABEL}/${RUN_TS}}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_ROOT}/task_logs}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/task_outputs}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EXPERIMENT_ROOT}/summaries}"
CHECKPOINT_SUMMARY_DIR="${CHECKPOINT_SUMMARY_DIR:-${SUMMARY_ROOT}/checkpoints}"
GLOBAL_SUMMARY_DIR="${GLOBAL_SUMMARY_DIR:-${SUMMARY_ROOT}/global}"

mkdir -p "${LOG_DIR}"
mkdir -p "${TASK_OUTPUT_ROOT}"
mkdir -p "${CHECKPOINT_SUMMARY_DIR}"
mkdir -p "${GLOBAL_SUMMARY_DIR}"

read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"
if [[ "${#GPU_IDS_ARR[@]}" -eq 0 ]]; then
  echo "No GPU ids provided in GPU_IDS." >&2
  exit 1
fi

MAX_PARALLEL_CHECKPOINTS="${#GPU_IDS_ARR[@]}"
if [[ -z "${PARALLEL_CHECKPOINTS}" ]]; then
  PARALLEL_CHECKPOINTS="${MAX_PARALLEL_CHECKPOINTS}"
elif [[ "${PARALLEL_CHECKPOINTS}" -gt "${MAX_PARALLEL_CHECKPOINTS}" ]]; then
  echo "PARALLEL_CHECKPOINTS=${PARALLEL_CHECKPOINTS} exceeds max ${MAX_PARALLEL_CHECKPOINTS} for GPU_IDS=${GPU_IDS_VALUE}." >&2
  exit 1
fi

CHECKPOINT_ROOT="$(normalize_checkpoint_root "${CHECKPOINT_DIR}")"
CHECKPOINTS=()
if [[ -n "${CHECKPOINT_DIRS_VALUE}" ]]; then
  read -r -a CHECKPOINT_DIRS_ARR <<< "${CHECKPOINT_DIRS_VALUE}"
  for cp in "${CHECKPOINT_DIRS_ARR[@]}"; do
    [[ -n "${cp}" ]] || continue
    CHECKPOINTS+=("$(normalize_checkpoint_root "${cp}")")
  done
else
  if [[ -d "${CHECKPOINT_ROOT}" ]] && is_checkpoint_like_name "$(basename "${CHECKPOINT_ROOT}")"; then
    CHECKPOINTS+=("${CHECKPOINT_ROOT}")
  elif [[ -d "${CHECKPOINT_ROOT}" ]]; then
    CHECKPOINTS_RAW=$(
      {
        ls -d "${CHECKPOINT_ROOT}"/checkpoint-* 2>/dev/null || true
        ls -d "${CHECKPOINT_ROOT}"/chpoint-* 2>/dev/null || true
        ls -d "${CHECKPOINT_ROOT}"/ckpt-* 2>/dev/null || true
      } | awk '!seen[$0]++' | sort -V
    )
    if [[ -n "${CHECKPOINTS_RAW}" ]]; then
      while IFS= read -r cp; do
        [[ -n "${cp}" ]] && CHECKPOINTS+=("${cp}")
      done <<< "${CHECKPOINTS_RAW}"
    fi
  fi
fi

if [[ -n "${ONLY_CHECKPOINTS_VALUE}" ]]; then
  read -r -a ONLY_CHECKPOINTS_ARR <<< "${ONLY_CHECKPOINTS_VALUE}"
  FILTERED=()
  for cp in "${CHECKPOINTS[@]-}"; do
    cp_name="$(basename "$cp")"
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
  echo "No checkpoints found under ${CHECKPOINT_DIR}." >&2
  exit 1
fi

case "${MODEL_MODE}" in
  auto|reft|lora|base)
    ;;
  *)
    echo "Invalid MODEL_MODE=${MODEL_MODE}, expected one of: auto reft lora base." >&2
    exit 1
    ;;
esac

GLOBAL_SUMMARY_FILE="${GLOBAL_SUMMARY_DIR}/${TRAIN_RUN_NAME}_${EVAL_TASK_LABEL}_global_summary_${RUN_TS}.json"

GROUP_PIDS=()

cleanup_children() {
  for pid in "${GROUP_PIDS[@]-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup_children INT TERM

launch_checkpoint_run() {
  local checkpoint_path="$1"
  local gpu_id="$2"
  local target_name
  local target_mode
  local group_log_dir
  local group_status_file
  local task_summary
  local task_log
  local task_name
  local run_base_model="${BASE_MODEL}"
  local run_reft_weights=""
  local run_lora_weights=""
  local target_name_safe
  local task_summary_tmp
  local task_output_dir
  local manifest_file

  checkpoint_path="$(normalize_checkpoint_root "${checkpoint_path}")"
  target_name="$(target_name_from_path "${checkpoint_path}")"
  target_mode="${MODEL_MODE}"
  if [[ "${target_mode}" == "auto" ]]; then
    target_mode="$(infer_checkpoint_mode "${checkpoint_path}")"
  fi

  case "${target_mode}" in
    reft)
      run_reft_weights="${checkpoint_path}/intervenable_model/"
      ;;
    lora)
      run_lora_weights="${checkpoint_path}"
      ;;
    base)
      run_base_model="${checkpoint_path}"
      ;;
    *)
      echo "Unsupported target mode ${target_mode} for ${checkpoint_path}" >&2
      return 1
      ;;
  esac

  target_name_safe="$(sanitize_name "${target_name}")"
  group_log_dir="${LOG_DIR}/${target_name_safe}"
  mkdir -p "${group_log_dir}"
  group_status_file="${group_log_dir}/group.status"
  task_summary="${CHECKPOINT_SUMMARY_DIR}/${target_name_safe}/summary_${RUN_TS}.json"
  manifest_file="${group_log_dir}/task_manifest_${RUN_TS}.jsonl"
  mkdir -p "$(dirname "${task_summary}")"
  : > "${manifest_file}"

  (
    set +e
    failed_tasks=()

    echo "checkpoint=${checkpoint_path}"
    echo "target_name=${target_name}"
    echo "target_mode=${target_mode}"
    echo "source_task=${SOURCE_TASK}"
    echo "eval_tasks=${EVAL_TASKS_ARR[*]}"
    echo "gpu=${gpu_id}"
    echo "summary=${task_summary}"
    group_status=0

    for task_name in "${EVAL_TASKS_ARR[@]}"; do
      task_summary_tmp="${group_log_dir}/${task_name}.summary_${RUN_TS}.tmp.json"
      task_log="${group_log_dir}/${task_name}.log"
      task_output_dir="${TASK_OUTPUT_ROOT}/${target_name_safe}/${task_name}"
      echo "task=${task_name}"
      echo "task_log=${task_log}"
      echo "task_output_dir=${task_output_dir}"

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
          BASE_MODEL="${run_base_model}" \
          REFT_WEIGHTS="${run_reft_weights}" \
          LORA_WEIGHTS="${run_lora_weights}" \
          TARGET_LAYERS="${TARGET_LAYERS}" \
          SUBSPACE_RANK="${SUBSPACE_RANK}" \
          SUMMARY_PATH="${task_summary_tmp}" \
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
          DEVICE="cuda:${gpu_id}" \
          bash "${TASK_SCRIPT}"
      ) >"${task_log}" 2>&1
      task_status=$?

      if [[ "${task_status}" -eq 0 && -f "${task_summary_tmp}" ]]; then
        python3 - "${manifest_file}" "${task_name}" "${task_summary_tmp}" "${task_log}" "${task_output_dir}" <<'PY_APPEND'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
record = {
    "task": sys.argv[2],
    "summary_file": sys.argv[3],
    "task_log": sys.argv[4],
    "task_output_dir": sys.argv[5],
}
with manifest_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")
PY_APPEND
      else
        if [[ "${task_status}" -ne 0 ]]; then
          failed_tasks+=("${task_name}:task_failed")
        fi
        if [[ ! -f "${task_summary_tmp}" ]]; then
          failed_tasks+=("${task_name}:missing_summary")
        fi
        group_status=1
      fi
    done

    python3 - "${task_summary}" "${RUN_TS}" "${checkpoint_path}" "${target_name}" "${SOURCE_TASK}" "${target_mode}" "${manifest_file}" "${failed_tasks[*]-}" "${EVAL_TASKS_ARR[@]-}" <<'PY2'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
checkpoint_path = sys.argv[3]
target_name = sys.argv[4]
source_task = sys.argv[5]
target_mode = sys.argv[6]
manifest_file = Path(sys.argv[7])
failed_tasks = [item for item in (sys.argv[8].split() if len(sys.argv) > 8 else []) if item]
eval_tasks = [item for item in sys.argv[9:] if item] if len(sys.argv) > 9 else []

task_runs = []
if manifest_file.exists():
    with manifest_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary_file = Path(record["summary_file"])
            task_summary = None
            if summary_file.exists():
                with summary_file.open("r", encoding="utf-8") as sf:
                    task_summary = json.load(sf)
                try:
                    summary_file.unlink()
                except OSError:
                    pass
            task_runs.append(
                {
                    "task": record["task"],
                    "task_log": record["task_log"],
                    "task_output_dir": record["task_output_dir"],
                    "task_summary": task_summary,
                }
            )

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "checkpoint_path": checkpoint_path,
    "target_name": target_name,
    "source_task": source_task,
    "eval_tasks": eval_tasks,
    "target_mode": target_mode,
    "failed_tasks": failed_tasks,
    "task_runs": task_runs,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
PY2

    printf '%s\n' "${group_status}" > "${group_status_file}"
    exit "${group_status}"
  ) &

  GROUP_PIDS+=("$!")
}

launch_single_task_run() {
  local checkpoint_path="$1"
  local source_task="$2"
  local eval_task="$3"
  local gpu_id="$4"
  local checkpoint_idx="$5"
  local target_name
  local target_mode
  local unit_name_safe
  local unit_log_dir
  local unit_status_file
  local unit_summary
  local unit_summary_tmp
  local task_log
  local task_output_dir
  local run_base_model="${BASE_MODEL}"
  local run_reft_weights=""
  local run_lora_weights=""

  checkpoint_path="$(normalize_checkpoint_root "${checkpoint_path}")"
  target_name="$(target_name_from_path "${checkpoint_path}")"
  target_mode="${MODEL_MODE}"
  if [[ "${target_mode}" == "auto" ]]; then
    target_mode="$(infer_checkpoint_mode "${checkpoint_path}")"
  fi

  case "${target_mode}" in
    reft)
      run_reft_weights="${checkpoint_path}/intervenable_model/"
      ;;
    lora)
      run_lora_weights="${checkpoint_path}"
      ;;
    base)
      run_base_model="${checkpoint_path}"
      ;;
    *)
      echo "Unsupported target mode ${target_mode} for ${checkpoint_path}" >&2
      return 1
      ;;
  esac

  unit_name_safe="$(sanitize_name "$(printf '%02d_%s__to__%s' "${checkpoint_idx}" "${target_name}" "${eval_task}")")"
  unit_log_dir="${LOG_DIR}/${unit_name_safe}"
  mkdir -p "${unit_log_dir}"
  unit_status_file="${unit_log_dir}/unit.status"
  unit_summary="${CHECKPOINT_SUMMARY_DIR}/${unit_name_safe}/summary_${RUN_TS}.json"
  unit_summary_tmp="${unit_summary}.task.tmp"
  task_log="${unit_log_dir}/${eval_task}.log"
  task_output_dir="${TASK_OUTPUT_ROOT}/$(sanitize_name "${target_name}")/${eval_task}"
  mkdir -p "$(dirname "${unit_summary}")"

  (
    set +e
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
      TASK="${eval_task}" \
      BASE_MODEL="${run_base_model}" \
      REFT_WEIGHTS="${run_reft_weights}" \
      LORA_WEIGHTS="${run_lora_weights}" \
      TARGET_LAYERS="${TARGET_LAYERS}" \
      SUBSPACE_RANK="${SUBSPACE_RANK}" \
      SUMMARY_PATH="${unit_summary_tmp}" \
      RESULT_PREFIX="$(sanitize_name "${target_name}")" \
      CHECKPOINT_NAME="$(sanitize_name "${target_name}")" \
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
      DEVICE="cuda:${gpu_id}" \
      bash "${TASK_SCRIPT}" >"${task_log}" 2>&1
    task_status=$?

    python3 - "${unit_summary}" "${RUN_TS}" "${checkpoint_path}" "${target_name}" "${source_task}" "${eval_task}" "${target_mode}" "${task_log}" "${task_output_dir}" "${task_status}" "${unit_summary_tmp}" <<'PY_UNIT'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
checkpoint_path = sys.argv[3]
target_name = sys.argv[4]
source_task = sys.argv[5]
eval_task = sys.argv[6]
target_mode = sys.argv[7]
task_log = sys.argv[8]
task_output_dir = sys.argv[9]
task_status = int(sys.argv[10])
tmp_summary = Path(sys.argv[11])

task_summary = None
failed_reasons = []
if task_status == 0 and tmp_summary.exists():
    with tmp_summary.open("r", encoding="utf-8") as f:
        task_summary = json.load(f)
    try:
        tmp_summary.unlink()
    except OSError:
        pass
else:
    if task_status != 0:
        failed_reasons.append("task_failed")
    if not tmp_summary.exists():
        failed_reasons.append("missing_summary")

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "checkpoint_path": checkpoint_path,
    "target_name": target_name,
    "source_task": source_task,
    "eval_task": eval_task,
    "target_mode": target_mode,
    "task_log": task_log,
    "task_output_dir": task_output_dir,
    "failed_reasons": failed_reasons,
    "task_summary": task_summary,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
sys.exit(0 if not failed_reasons else 1)
PY_UNIT
    unit_status=$?
    printf '%s\n' "${unit_status}" > "${unit_status_file}"
    exit "${unit_status}"
  ) &

  GROUP_PIDS+=("$!")
}

echo "PYTHON_BIN=${PYTHON_BIN}"
echo "BASE_MODEL=${BASE_MODEL}"
echo "MODEL_KEY=${MODEL_KEY}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CHECKPOINT_DIRS=${CHECKPOINT_DIRS_VALUE}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "SPECIALIST_TASK=${SPECIALIST_TASK}"
echo "SOURCE_TASK=${SOURCE_TASK}"
echo "TASKS=${TASKS_VALUE}"
echo "EXCLUDE_SPECIALIST_TASK=${EXCLUDE_SPECIALIST_TASK}"
echo "FLAT_PARALLEL_TASKS=${FLAT_PARALLEL_TASKS}"
echo "EVAL_TASKS=${EVAL_TASKS_ARR[*]}"
echo "EVAL_TASK_LABEL=${EVAL_TASK_LABEL}"
echo "EXPERIMENT_ROOT=${EXPERIMENT_ROOT}"
echo "MODEL_MODE=${MODEL_MODE}"
echo "MERGE_SUMMARY_PATH=${MERGE_SUMMARY_PATH}"
echo "LOG_DIR=${LOG_DIR}"
echo "TASK_OUTPUT_ROOT=${TASK_OUTPUT_ROOT}"
echo "CHECKPOINT_SUMMARY_DIR=${CHECKPOINT_SUMMARY_DIR}"
echo "GLOBAL_SUMMARY_DIR=${GLOBAL_SUMMARY_DIR}"
echo "GPU_IDS=${GPU_IDS_VALUE}"
echo "PARALLEL_CHECKPOINTS=${PARALLEL_CHECKPOINTS}"
echo "RUN_TS=${RUN_TS}"
echo "STAGGER_SECONDS=${STAGGER_SECONDS}"
echo "TARGET_LAYERS=${TARGET_LAYERS}"
echo "SUBSPACE_RANK=${SUBSPACE_RANK}"

if [[ "${FLAT_PARALLEL_TASKS}" == "1" ]]; then
  UNIT_CHECKPOINTS=()
  UNIT_SOURCE_TASKS=()
  UNIT_EVAL_TASKS=()

  for checkpoint_idx in "${!CHECKPOINTS[@]}"; do
    checkpoint_path="${CHECKPOINTS[$checkpoint_idx]}"
    checkpoint_source_task="$(infer_specialist_task "${SPECIALIST_TASK}" "${checkpoint_path}" "$(experiment_run_name_from_checkpoint_dir "${checkpoint_path}")")" || {
      echo "Unable to infer specialist task for ${checkpoint_path}" >&2
      exit 1
    }
    while IFS= read -r task_name; do
      [[ -n "${task_name}" ]] || continue
      UNIT_CHECKPOINTS+=("${checkpoint_path}")
      UNIT_SOURCE_TASKS+=("${checkpoint_source_task}")
      UNIT_EVAL_TASKS+=("${task_name}")
    done < <(build_eval_tasks_for_source "${checkpoint_source_task}" "${TASKS_VALUE}" "${EXCLUDE_SPECIALIST_TASK}")
  done

  if [[ "${#UNIT_CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No unit runs were generated." >&2
    exit 1
  fi

  echo "UNIT_RUN_COUNT=${#UNIT_CHECKPOINTS[@]}"

  GROUP_PIDS=()
  for ((batch_start=0; batch_start<${#UNIT_CHECKPOINTS[@]}; batch_start+=${#GPU_IDS_ARR[@]})); do
    GROUP_PIDS=()
    launched=0
    for ((slot=0; slot<${#GPU_IDS_ARR[@]}; slot++)); do
      unit_idx=$((batch_start + slot))
      if [[ "${unit_idx}" -ge "${#UNIT_CHECKPOINTS[@]}" ]]; then
        break
      fi
      checkpoint_path="${UNIT_CHECKPOINTS[$unit_idx]}"
      checkpoint_source_task="${UNIT_SOURCE_TASKS[$unit_idx]}"
      eval_task="${UNIT_EVAL_TASKS[$unit_idx]}"
      gpu_id="${GPU_IDS_ARR[$slot]}"
      launch_single_task_run "${checkpoint_path}" "${checkpoint_source_task}" "${eval_task}" "${gpu_id}" "${unit_idx}"
      launched=$((launched + 1))
      if [[ "${STAGGER_SECONDS}" -gt 0 && "${launched}" -lt "${#GPU_IDS_ARR[@]}" && $((unit_idx + 1)) -lt "${#UNIT_CHECKPOINTS[@]}" ]]; then
        sleep "${STAGGER_SECONDS}"
      fi
    done
    for pid in "${GROUP_PIDS[@]}"; do
      wait "${pid}" || true
    done
  done

  SUCCESS_SUMMARY_FILES=()
  FAILED_RUNS=()
  for unit_idx in "${!UNIT_CHECKPOINTS[@]}"; do
    checkpoint_path="${UNIT_CHECKPOINTS[$unit_idx]}"
    eval_task="${UNIT_EVAL_TASKS[$unit_idx]}"
    target_name="$(target_name_from_path "${checkpoint_path}")"
    unit_name_safe="$(sanitize_name "$(printf '%02d_%s__to__%s' "${unit_idx}" "${target_name}" "${eval_task}")")"
    unit_log_dir="${LOG_DIR}/${unit_name_safe}"
    unit_status_file="${unit_log_dir}/unit.status"
    unit_summary_file="${CHECKPOINT_SUMMARY_DIR}/${unit_name_safe}/summary_${RUN_TS}.json"

    run_status=1
    if [[ -f "${unit_status_file}" ]]; then
      run_status="$(tr -d '[:space:]' < "${unit_status_file}")"
    fi

    if [[ "${run_status}" == "0" && -f "${unit_summary_file}" ]]; then
      SUCCESS_SUMMARY_FILES+=("${unit_summary_file}")
    else
      FAILED_RUNS+=("${unit_name_safe}")
    fi
  done

  python3 - "${GLOBAL_SUMMARY_FILE}" "${RUN_TS}" "${BASE_MODEL}" "${CHECKPOINT_DIRS_VALUE:-${CHECKPOINT_DIR}}" "${EVAL_TASK_LABEL}" "${GPU_IDS_VALUE}" "${FAILED_RUNS[*]-}" "${SUCCESS_SUMMARY_FILES[@]-}" <<'PY_FLAT'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
base_model = sys.argv[3]
checkpoint_spec = sys.argv[4]
eval_task_label = sys.argv[5]
gpu_ids = [x for x in sys.argv[6].split() if x]
failed_runs = [x for x in sys.argv[7].split() if x] if len(sys.argv) > 7 else []
summary_files = [item for item in sys.argv[8:] if item] if len(sys.argv) > 8 else []

unit_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    unit_runs.append(payload)

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "base_model": base_model,
    "checkpoint_spec": checkpoint_spec,
    "eval_task_label": eval_task_label,
    "gpu_ids": gpu_ids,
    "failed_runs": failed_runs,
    "unit_runs": unit_runs,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
PY_FLAT

  echo "Global summary saved to: ${GLOBAL_SUMMARY_FILE}"
  if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
    echo "Some unit runs failed: ${FAILED_RUNS[*]}" >&2
    exit 1
  fi
  exit 0
fi

for ((batch_start=0; batch_start<${#CHECKPOINTS[@]}; batch_start+=PARALLEL_CHECKPOINTS)); do
  GROUP_PIDS=()
  launched=0

  for ((slot=0; slot<PARALLEL_CHECKPOINTS; slot++)); do
    ckpt_idx=$((batch_start + slot))
    if [[ "${ckpt_idx}" -ge "${#CHECKPOINTS[@]}" ]]; then
      break
    fi

    checkpoint_path="${CHECKPOINTS[$ckpt_idx]}"
    gpu_id="${GPU_IDS_ARR[$slot]}"
    launch_checkpoint_run "${checkpoint_path}" "${gpu_id}"
    launched=$((launched + 1))

    if [[ "${STAGGER_SECONDS}" -gt 0 && "${launched}" -lt "${PARALLEL_CHECKPOINTS}" && $((ckpt_idx + 1)) -lt "${#CHECKPOINTS[@]}" ]]; then
      sleep "${STAGGER_SECONDS}"
    fi
  done

  if [[ "${launched}" -eq 0 ]]; then
    break
  fi

  for pid in "${GROUP_PIDS[@]}"; do
    wait "${pid}" || true
  done
done

SUCCESS_SUMMARY_FILES=()
FAILED_RUNS=()
for idx in "${!CHECKPOINTS[@]}"; do
  checkpoint_path="${CHECKPOINTS[$idx]}"
  target_name="$(target_name_from_path "${checkpoint_path}")"
  target_name_safe="$(sanitize_name "${target_name}")"
  group_log_dir="${LOG_DIR}/${target_name_safe}"
  group_status_file="${group_log_dir}/group.status"
  checkpoint_summary_file="${CHECKPOINT_SUMMARY_DIR}/${target_name_safe}/summary_${RUN_TS}.json"

  run_status=1
  if [[ -f "${group_status_file}" ]]; then
    run_status="$(tr -d '[:space:]' < "${group_status_file}")"
  fi

  if [[ "${run_status}" == "0" && -f "${checkpoint_summary_file}" ]]; then
    SUCCESS_SUMMARY_FILES+=("${checkpoint_summary_file}")
  else
    FAILED_RUNS+=("${target_name_safe}")
  fi
done

python3 - "${GLOBAL_SUMMARY_FILE}" "${RUN_TS}" "${BASE_MODEL}" "${CHECKPOINT_DIR}" "${SOURCE_TASK}" "${EVAL_TASK_LABEL}" "${GPU_IDS_VALUE}" "${FAILED_RUNS[*]-}" "${SUCCESS_SUMMARY_FILES[@]-}" <<'PY3'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
base_model = sys.argv[3]
checkpoint_dir = sys.argv[4]
source_task = sys.argv[5]
eval_task_label = sys.argv[6]
gpu_ids = [x for x in sys.argv[7].split() if x]
failed_runs = [x for x in sys.argv[8].split() if x] if len(sys.argv) > 8 else []
summary_files = [item for item in sys.argv[9:] if item] if len(sys.argv) > 9 else []

checkpoint_runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    checkpoint_runs.append(payload)

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "base_model": base_model,
    "checkpoint_dir": checkpoint_dir,
    "source_task": source_task,
    "eval_task_label": eval_task_label,
    "gpu_ids": gpu_ids,
    "failed_runs": failed_runs,
    "checkpoint_runs": checkpoint_runs,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(str(summary_path))
PY3

echo "Global summary saved to: ${GLOBAL_SUMMARY_FILE}"
if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo "Some checkpoint runs failed: ${FAILED_RUNS[*]}" >&2
  exit 1
fi
