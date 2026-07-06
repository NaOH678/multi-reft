#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash, not sh/zsh." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
# basemodels ../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
# reft weights multi_train/trainer_output/Llama3-8b-Nodireft_truthful/stereotype
# sft baseline/LoRA/llama3-8b-sft/checkpoint-4060
# lora adapter baseline/LoRA/llama3-8b-lora/checkpoint-1524

MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-qwen3-4b-loreft-toxicity}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-multi_train/trainer_output/qwen3-4b_Loreft_toxicity_lr1p2e-3_bs16_ga2_ep6}"
BASE_MODEL="${BASE_MODEL:-models/qwen3-4b}"
MODEL_MODE="${MODEL_MODE:-auto}"  # auto | reft | lora | base
REFT_WEIGHTS="${REFT_WEIGHTS:-}"
LORA_WEIGHTS="${LORA_WEIGHTS:-}"
ONLY_CHECKPOINTS_DEFAULT="${ONLY_CHECKPOINTS_DEFAULT:-}"
ONLY_CHECKPOINTS_VALUE="${ONLY_CHECKPOINTS:-${ONLY_CHECKPOINTS_DEFAULT}}"

DEVICE="${DEVICE:-cuda:0}"
GPU_IDS_VALUE="${GPU_IDS:-}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
STAGGER_SECONDS="${STAGGER_SECONDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
POSITIONS="${POSITIONS:-11}"
GREEDY_DECODING="${GREEDY_DECODING:-0}"

DATASETS_DEFAULT="${DATASETS_DEFAULT:-toxic nontoxic}"
PROMPTS_DEFAULT="${PROMPTS_DEFAULT:-benign adversarial}"
DATASETS_VALUE="${DATASETS:-${DATASETS_DEFAULT}}"
PROMPTS_VALUE="${PROMPTS:-${PROMPTS_DEFAULT}}"

DATASET_ROOT="${DATASET_ROOT:-multi_train/eval_toxicity/data/user_prompts}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

N_GENERATIONS="${N_GENERATIONS:-5}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"

RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-32}"
DETOXIFY_MODEL="${DETOXIFY_MODEL:-original}"
ANALYSIS_DEVICE="${ANALYSIS_DEVICE:-}"



RUN_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="${SUMMARY_DIR:-multi_train/eval_toxicity/data/summaries}"
LOG_DIR="${LOG_DIR:-multi_train/logs/evaluate_toxicity_${RUN_TS}}"
mkdir -p "${SUMMARY_DIR}"
mkdir -p "${LOG_DIR}"


read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a DATASETS_ARR <<< "${DATASETS_VALUE}"
read -r -a PROMPTS_ARR <<< "${PROMPTS_VALUE}"
read -r -a GPU_IDS_ARR <<< "${GPU_IDS_VALUE}"

if [[ -n "${GPU_IDS_VALUE}" ]]; then
  if [[ "${#GPU_IDS_ARR[@]}" -eq 0 ]]; then
    echo "GPU_IDS was provided but no GPU ids were parsed." >&2
    exit 1
  fi
  if [[ -z "${MAX_PARALLEL}" ]]; then
    MAX_PARALLEL="${#GPU_IDS_ARR[@]}"
  fi
else
  MAX_PARALLEL="${MAX_PARALLEL:-1}"
fi

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  echo "${value}"
}

resolve_base_model_path() {
  local candidate="$1"
  local refs_main
  local snapshot_id

  if [[ -d "${candidate}" && -f "${candidate}/config.json" ]]; then
    echo "${candidate}"
    return 0
  fi

  refs_main="${candidate%/}/refs/main"
  if [[ -f "${refs_main}" ]]; then
    snapshot_id="$(tr -d '[:space:]' < "${refs_main}")"
    if [[ -n "${snapshot_id}" && -d "${candidate%/}/snapshots/${snapshot_id}" ]]; then
      echo "${candidate%/}/snapshots/${snapshot_id}"
      return 0
    fi
  fi

  echo "${candidate}"
}

BASE_MODEL="$(resolve_base_model_path "${BASE_MODEL}")"

compute_model_tag() {
  local base_model="$1"
  local reft_weights="$2"
  local lora_weights="$3"

  python3 - "$base_model" "$reft_weights" "$lora_weights" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_model_tag

base_model = sys.argv[1]
reft_weights = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
lora_weights = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None

print(
    build_model_tag(
        base_model_path=base_model,
        reft_weights_path=reft_weights,
        lora_weights_path=lora_weights,
    )
)
PY
}

compute_merge_model_tag() {
  local base_model="$1"
  local mode_label="$2"

  python3 - "$base_model" "$mode_label" <<'PY'
import sys
from multi_train.eval_common.output_naming import normalize_base_model_name

base_model = sys.argv[1] if len(sys.argv) > 1 else ""
mode_label = (sys.argv[2] if len(sys.argv) > 2 else "").strip() or "merge"
base_name = normalize_base_model_name(base_model) if base_model else "unknown_model"
print(f"{base_name}-{mode_label}")
PY
}

resolve_output_prefix() {
  local prefix="$1"
  local base_model="$2"
  local mode_label="$3"

  python3 - "$prefix" "$base_model" "$mode_label" <<'PY'
import sys
from multi_train.eval_common.output_naming import resolve_output_prefix

prefix = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
base_model = sys.argv[2] if len(sys.argv) > 2 else ""
mode_label = sys.argv[3] if len(sys.argv) > 3 else "merge"
print(resolve_output_prefix(prefix, base_model, mode_label))
PY
}

compute_checkpoint_label() {
  local base_model="$1"
  local reft_weights="$2"
  local lora_weights="$3"

  python3 - "$base_model" "$reft_weights" "$lora_weights" <<'PY'
import sys
from multi_train.eval_common.output_naming import extract_checkpoint_label

base_model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
reft_weights = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
lora_weights = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
print(extract_checkpoint_label(base_model, reft_weights, lora_weights))
PY
}

build_single_summary_path_with_prefix() {
  local summary_dir="$1"
  local prefix="$2"
  local task_name="$3"
  local run_ts="$4"
  local checkpoint_label="$5"

  python3 - "$summary_dir" "$prefix" "$task_name" "$run_ts" "$checkpoint_label" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_single_summary_path

summary_dir = sys.argv[1]
prefix = sys.argv[2]
task_name = sys.argv[3]
run_ts = sys.argv[4]
checkpoint_label = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
print(build_single_summary_path(summary_dir, prefix, task_name, run_ts, checkpoint_label))
PY
}

build_merge_summary_path_with_prefix() {
  local summary_dir="$1"
  local prefix="$2"
  local run_ts="$3"

  python3 - "$summary_dir" "$prefix" "$run_ts" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_merge_summary_path

summary_dir = sys.argv[1]
prefix = sys.argv[2]
run_ts = sys.argv[3]
print(build_merge_summary_path(summary_dir, prefix, run_ts))
PY
}

build_results_csv_path_with_prefix() {
  local output_dir="$1"
  local prefix="$2"
  local task_name="$3"
  local checkpoint_label="$4"

  python3 - "$output_dir" "$prefix" "$task_name" "$checkpoint_label" <<'PY'
import sys
from multi_train.eval_common.output_naming import build_results_csv_path

output_dir = sys.argv[1]
prefix = sys.argv[2]
task_name = sys.argv[3]
checkpoint_label = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
print(build_results_csv_path(output_dir, prefix, task_name, checkpoint_label))
PY
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

declare -a TARGET_NAMES=()
declare -a TARGET_BASE_MODELS=()
declare -a TARGET_REFT_WEIGHTS=()
declare -a TARGET_LORA_WEIGHTS=()

add_target() {
  local name="$1"
  local base_model="$2"
  local reft_weights="$3"
  local lora_weights="$4"
  TARGET_NAMES+=("$name")
  TARGET_BASE_MODELS+=("$base_model")
  TARGET_REFT_WEIGHTS+=("$reft_weights")
  TARGET_LORA_WEIGHTS+=("$lora_weights")
}

if [[ -n "${REFT_WEIGHTS}" && -n "${LORA_WEIGHTS}" ]]; then
  echo "REFT_WEIGHTS and LORA_WEIGHTS cannot be set at the same time." >&2
  exit 1
fi

CHECKPOINTS=()
if [[ -d "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINTS_RAW=$(
    {
      ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null || true
      ls -d "${CHECKPOINT_DIR}"/chpoint-* 2>/dev/null || true
      ls -d "${CHECKPOINT_DIR}"/ckpt-* 2>/dev/null || true
    } | awk '!seen[$0]++' | sort -V
  )
  if [[ -n "${CHECKPOINTS_RAW}" ]]; then
    while IFS= read -r cp; do
      [[ -n "${cp}" ]] && CHECKPOINTS+=("${cp}")
    done <<< "${CHECKPOINTS_RAW}"
  fi
fi

if [[ -n "${ONLY_CHECKPOINTS_VALUE}" ]]; then
  read -r -a ONLY_CHECKPOINTS_ARR <<< "${ONLY_CHECKPOINTS_VALUE}"
  FILTERED=()
  for cp in "${CHECKPOINTS[@]-}"; do
    cp_name="$(basename "$cp")"
    for wanted in "${ONLY_CHECKPOINTS_ARR[@]}"; do
      if [[ "$cp_name" == "$wanted" ]]; then
        FILTERED+=("$cp")
        break
      fi
    done
  done
  CHECKPOINTS=("${FILTERED[@]}")
fi

case "${MODEL_MODE}" in
  auto|reft|lora|base)
    ;;
  *)
    echo "Invalid MODEL_MODE=${MODEL_MODE}, expected one of: auto reft lora base." >&2
    exit 1
    ;;
esac

if [[ -n "${REFT_WEIGHTS}" ]]; then
  reft_path="${REFT_WEIGHTS%/}"
  reft_base="$(basename "${reft_path}")"
  if [[ "${reft_base}" == "intervenable_model" ]]; then
    target_name="$(target_name_from_path "$(dirname "${reft_path}")")"
  else
    target_name="$(target_name_from_path "${reft_path}")"
  fi
  add_target "${target_name}" "${BASE_MODEL}" "${REFT_WEIGHTS}" ""
elif [[ -n "${LORA_WEIGHTS}" ]]; then
  target_name="$(target_name_from_path "${LORA_WEIGHTS}")"
  add_target "${target_name}" "${BASE_MODEL}" "" "${LORA_WEIGHTS}"
elif [[ ${#CHECKPOINTS[@]} -gt 0 ]]; then
  for checkpoint in "${CHECKPOINTS[@]}"; do
    target_name="$(target_name_from_path "${checkpoint}")"
    case "${MODEL_MODE}" in
      reft)
        add_target "${target_name}" "${BASE_MODEL}" "${checkpoint}/intervenable_model/" ""
        ;;
      lora)
        add_target "${target_name}" "${BASE_MODEL}" "" "${checkpoint}"
        ;;
      base)
        add_target "${target_name}" "${checkpoint}" "" ""
        ;;
      auto)
        detected_mode="$(infer_checkpoint_mode "${checkpoint}")"
        if [[ "${detected_mode}" == "reft" ]]; then
          add_target "${target_name}" "${BASE_MODEL}" "${checkpoint}/intervenable_model/" ""
        elif [[ "${detected_mode}" == "lora" ]]; then
          add_target "${target_name}" "${BASE_MODEL}" "" "${checkpoint}"
        else
          add_target "${target_name}" "${checkpoint}" "" ""
        fi
        ;;
    esac
  done
else
  if [[ "${MODEL_MODE}" == "reft" || "${MODEL_MODE}" == "lora" ]]; then
    echo "No checkpoints found under ${CHECKPOINT_DIR} for MODEL_MODE=${MODEL_MODE}." >&2
    exit 1
  fi
  base_name="$(target_name_from_path "${BASE_MODEL}")"
  add_target "${base_name:-base_model}" "${BASE_MODEL}" "" ""
fi

if [[ ${#TARGET_NAMES[@]} -eq 0 ]]; then
  echo "No evaluation targets were generated." >&2
  exit 1
fi

HAS_REFT_TARGET=0
HAS_LORA_TARGET=0
HAS_BASE_TARGET=0
for idx in "${!TARGET_NAMES[@]}"; do
  if [[ -n "${TARGET_REFT_WEIGHTS[$idx]}" ]]; then
    HAS_REFT_TARGET=1
  elif [[ -n "${TARGET_LORA_WEIGHTS[$idx]}" ]]; then
    HAS_LORA_TARGET=1
  else
    HAS_BASE_TARGET=1
  fi
done

TARGET_MODE_KINDS=$((HAS_REFT_TARGET + HAS_LORA_TARGET + HAS_BASE_TARGET))
if [[ "${TARGET_MODE_KINDS}" -eq 1 ]]; then
  if [[ "${HAS_REFT_TARGET}" -eq 1 ]]; then
    MERGE_MODE_LABEL="reft"
  elif [[ "${HAS_LORA_TARGET}" -eq 1 ]]; then
    MERGE_MODE_LABEL="lora"
  else
    MERGE_MODE_LABEL="base"
  fi
elif [[ "${MODEL_MODE}" != "auto" ]]; then
  MERGE_MODE_LABEL="${MODEL_MODE}"
else
  MERGE_MODE_LABEL="mixed"
fi

MERGE_SUMMARY_PREFIX="$(resolve_output_prefix "${MERGE_SUMMARY_PATH:-}" "${BASE_MODEL}" "${MERGE_MODE_LABEL}")"
MERGE_SUMMARY_PATH="$(build_merge_summary_path_with_prefix "${SUMMARY_DIR}" "${MERGE_SUMMARY_PREFIX}" "${RUN_TS}")"

SUCCESS_SUMMARIES=()
FAILED_RUNS=()
PIDS=()
TARGET_NAMES_RUN=()
SUMMARY_PATHS_RUN=()
STATUS_FILES_RUN=()
LOG_FILES_RUN=()

run_target() {
  local idx="$1"
  local gpu_slot="$2"
  local target_name="${TARGET_NAMES[$idx]}"
  local run_base_model="${TARGET_BASE_MODELS[$idx]}"
  local run_reft_weights="${TARGET_REFT_WEIGHTS[$idx]}"
  local run_lora_weights="${TARGET_LORA_WEIGHTS[$idx]}"
  local checkpoint_label
  checkpoint_label="$(compute_checkpoint_label "${run_base_model}" "${run_reft_weights}" "${run_lora_weights}")"
  local run_summary_file
  run_summary_file="$(build_single_summary_path_with_prefix "${SUMMARY_DIR}" "${MERGE_SUMMARY_PREFIX}" "toxicity" "${RUN_TS}" "${checkpoint_label}")"
  local run_results_csv
  run_results_csv="$(build_results_csv_path_with_prefix "multi_train/eval_toxicity/data/generations" "${MERGE_SUMMARY_PREFIX}" "toxicity" "${checkpoint_label}")"
  local run_log_file="${LOG_DIR}/$(sanitize_name "${target_name}").log"
  local run_status_file="${LOG_DIR}/$(sanitize_name "${target_name}").status"
  local run_device="${DEVICE}"
  local run_analysis_device="${ANALYSIS_DEVICE}"

  if [[ -n "${GPU_IDS_VALUE}" ]]; then
    run_device="cuda:${GPU_IDS_ARR[$gpu_slot]}"
    if [[ -z "${run_analysis_device}" ]]; then
      run_analysis_device="${run_device}"
    fi
  elif [[ -z "${run_analysis_device}" ]]; then
    run_analysis_device="${run_device}"
  fi

  echo "Evaluating target: ${target_name}, device=${run_device}, datasets=${DATASETS_VALUE}, prompts=${PROMPTS_VALUE}, log=${run_log_file}"

  (
    cmd=(
      python3 multi_train/eval_toxicity/toxicity_exp.py
      --base_model "${run_base_model}"
      --device "${run_device}"
      --batch_size "${BATCH_SIZE}"
      --datasets
      "${DATASETS_ARR[@]}"
      --prompts
      "${PROMPTS_ARR[@]}"
      --target_layers
      "${TARGET_LAYER_ARR[@]}"
      --subspace_rank "${SUBSPACE_RANK}"
      --positions "${POSITIONS}"
      --greedy_decoding "${GREEDY_DECODING}"
      --dataset_root "${DATASET_ROOT}"
      --n_generations "${N_GENERATIONS}"
      --max_tokens "${MAX_TOKENS}"
      --temperature "${TEMPERATURE}"
      --run_analysis "${RUN_ANALYSIS}"
      --analysis_batch_size "${ANALYSIS_BATCH_SIZE}"
      --detoxify_model "${DETOXIFY_MODEL}"
      --summary_json "${run_summary_file}"
      --results_csv "${run_results_csv}"
    )

    if [[ -n "${MAX_SAMPLES}" ]]; then
      cmd+=(--max_samples "${MAX_SAMPLES}")
    fi
    if [[ -n "${run_analysis_device}" ]]; then
      cmd+=(--analysis_device "${run_analysis_device}")
    fi
    if [[ -n "${run_reft_weights}" ]]; then
      cmd+=(--reft_weights "${run_reft_weights}")
    fi
    if [[ -n "${run_lora_weights}" ]]; then
      cmd+=(--lora_weights "${run_lora_weights}")
    fi

    echo "TARGET=${target_name}"
    echo "DEVICE=${run_device}"
    echo "ANALYSIS_DEVICE=${run_analysis_device}"
    echo "SUMMARY=${run_summary_file}"
    echo "RESULTS=${run_results_csv}"
    echo
    set +e
    CUDA_LAUNCH_BLOCKING=1 "${cmd[@]}"
    status=$?
    set -e
    printf '%s\n' "${status}" > "${run_status_file}"
    exit "${status}"
  ) >"${run_log_file}" 2>&1 &

  PIDS+=("$!")
  TARGET_NAMES_RUN+=("${target_name}")
  SUMMARY_PATHS_RUN+=("${run_summary_file}")
  STATUS_FILES_RUN+=("${run_status_file}")
  LOG_FILES_RUN+=("${run_log_file}")
}

cleanup_children() {
  for pid in "${PIDS[@]-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}

trap cleanup_children INT TERM

if [[ -n "${GPU_IDS_VALUE}" ]]; then
  active_jobs=0
  for idx in "${!TARGET_NAMES[@]}"; do
    gpu_slot=$((idx % ${#GPU_IDS_ARR[@]}))
    run_target "${idx}" "${gpu_slot}"
    active_jobs=$((active_jobs + 1))
    if [[ "${STAGGER_SECONDS}" -gt 0 ]]; then
      sleep "${STAGGER_SECONDS}"
    fi
    if [[ "${active_jobs}" -ge "${MAX_PARALLEL}" ]]; then
      wait
      active_jobs=0
    fi
  done
  if [[ "${active_jobs}" -gt 0 ]]; then
    wait
  fi
else
  for idx in "${!TARGET_NAMES[@]}"; do
    run_target "${idx}" 0
    wait "${PIDS[$idx]}"
  done
fi

status=0
for idx in "${!TARGET_NAMES_RUN[@]}"; do
  target_status=1
  if [[ -f "${STATUS_FILES_RUN[$idx]}" ]]; then
    target_status="$(tr -d '[:space:]' < "${STATUS_FILES_RUN[$idx]}")"
  fi

  if [[ "${target_status}" == "0" ]]; then
    if [[ -f "${SUMMARY_PATHS_RUN[$idx]}" ]]; then
      SUCCESS_SUMMARIES+=("${SUMMARY_PATHS_RUN[$idx]}")
    else
      FAILED_RUNS+=("$(sanitize_name "${TARGET_NAMES_RUN[$idx]}")")
      echo "FAILED: target=${TARGET_NAMES_RUN[$idx]} (summary missing, see ${LOG_FILES_RUN[$idx]})" >&2
      status=1
    fi
  else
    FAILED_RUNS+=("$(sanitize_name "${TARGET_NAMES_RUN[$idx]}")")
    echo "FAILED: target=${TARGET_NAMES_RUN[$idx]} (see ${LOG_FILES_RUN[$idx]})" >&2
    status=1
  fi
done

python3 multi_train/eval_toxicity/build_merge_summary.py \
  --merge_summary_path "${MERGE_SUMMARY_PATH}" \
  --run_ts "${RUN_TS}" \
  --datasets "${DATASETS_ARR[@]}" \
  --prompts "${PROMPTS_ARR[@]}" \
  --failed_runs "${FAILED_RUNS[@]-}" \
  --summary_files "${SUCCESS_SUMMARIES[@]-}"

echo "Merge summary saved to: ${MERGE_SUMMARY_PATH}"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Some runs failed: ${FAILED_RUNS[*]}" >&2
fi
