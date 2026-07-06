#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash, not sh/zsh." >&2
  exit 1
fi
# base model models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
# sft model 
# lora model baseline/LoRA/llama3-8b-lora/checkpoint-1524
TASK_MODE="${TASK_MODE:-privacy}"
TASK_NAME="${TASK_NAME:-$([[ "${TASK_MODE}" == "safety" ]] && echo safety || echo privacy)}"
MERGE_SUMMARY_PATH="${MERGE_SUMMARY_PATH:-llama3-8b-loreft-3-${TASK_NAME}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-multi_train/trainer_output/llama3-8b-Loreft_safety-3}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
MODEL_MODE="${MODEL_MODE:-auto}"  # auto | reft | lora | base
REFT_WEIGHTS="${REFT_WEIGHTS:-}"
LORA_WEIGHTS="${LORA_WEIGHTS:-}"
# checkpoint-48 checkpoint-96
ONLY_CHECKPOINTS_DEFAULT="${ONLY_CHECKPOINTS_DEFAULT:-}"
ONLY_CHECKPOINTS_VALUE="${ONLY_CHECKPOINTS:-${ONLY_CHECKPOINTS_DEFAULT}}"

DEVICE="${DEVICE:-cuda:0}"
if [[ "${TASK_MODE}" == "safety" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-16}"
else
  BATCH_SIZE="${BATCH_SIZE:-128}"
fi
TARGET_LAYERS="${TARGET_LAYERS:--1}"
SUBSPACE_RANK="${SUBSPACE_RANK:-8}"
POSITIONS="${POSITIONS:-7}"
if [[ "${TASK_MODE}" == "safety" ]]; then
  GREEDY_DECODING="${GREEDY_DECODING:-1}"
else
  GREEDY_DECODING="${GREEDY_DECODING:-0}"
fi

PROMPT_TYPES_DEFAULT="${PROMPT_TYPES_DEFAULT:-1}"
PII_INDICES_DEFAULT="${PII_INDICES_DEFAULT:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17}"
PROMPT_TYPES_VALUE="${PROMPT_TYPES:-${PROMPT_TYPES_DEFAULT}}"
PII_INDICES_VALUE="${PII_INDICES:-${PII_INDICES_DEFAULT}}"

if [[ "${TASK_MODE}" == "safety" ]]; then
  DATA_FILE="${DATA_FILE:-multi_train/eval_privacy/data/safety/wildjailbreak_eval.csv}"
  NUM_PROMPTS="${NUM_PROMPTS:-2210}"
  CONTEXT_EXAMPLES="${CONTEXT_EXAMPLES:-1}"
  MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-4}"
else
  DATA_FILE="${DATA_FILE:-multi_train/eval_privacy/enron_data/name2email.pkl}"
  NUM_PROMPTS="${NUM_PROMPTS:-100}"
  CONTEXT_EXAMPLES="${CONTEXT_EXAMPLES:-4}"
  MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-0}"
fi

if [[ "${TASK_MODE}" == "safety" ]]; then
  N_GENERATIONS="${N_GENERATIONS:-1}"
else
  N_GENERATIONS="${N_GENERATIONS:-3}"
fi
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"
RUN_STATISTICS="${RUN_STATISTICS:-1}"

SUMMARY_DIR="${SUMMARY_DIR:-multi_train/eval_privacy/data/summaries}"
mkdir -p "${SUMMARY_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a PROMPT_TYPES_ARR <<< "${PROMPT_TYPES_VALUE}"
read -r -a PII_INDICES_ARR <<< "${PII_INDICES_VALUE}"

sanitize_name() {
  local value="$1"
  value="${value//\//_}"
  value="${value// /_}"
  value="${value//:/_}"
  echo "${value}"
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
  if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "ONLY_CHECKPOINTS did not match any checkpoint under ${CHECKPOINT_DIR}." >&2
    echo "Requested: ${ONLY_CHECKPOINTS_VALUE}" >&2
    echo "Available checkpoints:" >&2
    ls -1d "${CHECKPOINT_DIR}"/checkpoint-* "${CHECKPOINT_DIR}"/chpoint-* "${CHECKPOINT_DIR}"/ckpt-* 2>/dev/null | xargs -r -n1 basename >&2 || true
    exit 1
  fi
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

for idx in "${!TARGET_NAMES[@]}"; do
  target_name="${TARGET_NAMES[$idx]}"
  run_base_model="${TARGET_BASE_MODELS[$idx]}"
  run_reft_weights="${TARGET_REFT_WEIGHTS[$idx]}"
  run_lora_weights="${TARGET_LORA_WEIGHTS[$idx]}"
  checkpoint_label="$(compute_checkpoint_label "${run_base_model}" "${run_reft_weights}" "${run_lora_weights}")"
  run_summary_file="$(build_single_summary_path_with_prefix "${SUMMARY_DIR}" "${MERGE_SUMMARY_PREFIX}" "${TASK_NAME}" "${RUN_TS}" "${checkpoint_label}")"
  run_results_csv="$(build_results_csv_path_with_prefix "multi_train/eval_privacy/data/generations" "${MERGE_SUMMARY_PREFIX}" "${TASK_NAME}" "${checkpoint_label}")"

  if [[ "${TASK_MODE}" == "safety" ]]; then
    echo "Evaluating target: ${target_name}, task_mode=${TASK_MODE}, data_file=${DATA_FILE}"
  else
    echo "Evaluating target: ${target_name}, prompt_types=${PROMPT_TYPES_VALUE}, pii_indices=${PII_INDICES_VALUE}"
  fi

  cmd=(
    python3 multi_train/eval_privacy/privacy_exp.py
    --task_mode "${TASK_MODE}"
    --base_model "${run_base_model}"
    --device "${DEVICE}"
    --batch_size "${BATCH_SIZE}"
    --target_layers
    "${TARGET_LAYER_ARR[@]}"
    --subspace_rank "${SUBSPACE_RANK}"
    --positions "${POSITIONS}"
    --greedy_decoding "${GREEDY_DECODING}"
    --data_file "${DATA_FILE}"
    --num_prompts "${NUM_PROMPTS}"
    --context_examples "${CONTEXT_EXAMPLES}"
    --n_generations "${N_GENERATIONS}"
    --max_tokens "${MAX_TOKENS}"
    --min_new_tokens "${MIN_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --run_statistics "${RUN_STATISTICS}"
    --summary_json "${run_summary_file}"
    --results_csv "${run_results_csv}"
  )

  if [[ "${TASK_MODE}" != "safety" ]]; then
    cmd+=(
      --prompt_types
      "${PROMPT_TYPES_ARR[@]}"
      --pii_indices
      "${PII_INDICES_ARR[@]}"
    )
  fi

  if [[ -n "${run_reft_weights}" ]]; then
    cmd+=(--reft_weights "${run_reft_weights}")
  fi
  if [[ -n "${run_lora_weights}" ]]; then
    cmd+=(--lora_weights "${run_lora_weights}")
  fi

  set +e
  CUDA_LAUNCH_BLOCKING=1 "${cmd[@]}"
  status=$?
  set -e

  if [[ ${status} -eq 0 ]]; then
    if [[ "${RUN_STATISTICS}" == "0" ]]; then
      if [[ -f "${run_summary_file}" ]]; then
        SUCCESS_SUMMARIES+=("${run_summary_file}")
      fi
    elif [[ -f "${run_summary_file}" ]]; then
      SUCCESS_SUMMARIES+=("${run_summary_file}")
    else
      FAILED_RUNS+=("$(sanitize_name "${target_name}")")
      echo "FAILED: target=${target_name} (summary missing)" >&2
    fi
  else
    FAILED_RUNS+=("$(sanitize_name "${target_name}")")
    echo "FAILED: target=${target_name}" >&2
  fi
done

python3 - "${MERGE_SUMMARY_PATH}" "${RUN_TS}" "${PROMPT_TYPES_VALUE}" "${PII_INDICES_VALUE}" "${FAILED_RUNS[*]-}" "${TASK_MODE}" "${SUCCESS_SUMMARIES[@]-}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

merge_summary_path = Path(sys.argv[1])
run_ts = sys.argv[2]
prompt_types = [int(x) for x in sys.argv[3].split()] if sys.argv[3].strip() else []
pii_indices = [int(x) for x in sys.argv[4].split()] if sys.argv[4].strip() else []
failed_runs = sys.argv[5].split() if len(sys.argv) >= 6 and sys.argv[5].strip() else []
task_mode = sys.argv[6]
summary_files = sys.argv[7:] if len(sys.argv) > 7 else []


def run_key(run):
    model_tag = str(run.get("model_tag") or "").strip()
    if model_tag:
        return model_tag

    reft_weights = str(run.get("reft_weights") or "").strip().rstrip("/")
    if reft_weights:
        path = Path(reft_weights)
        if path.name == "intervenable_model":
            path = path.parent
        leaf = path.name
        parent = path.parent.name
        if leaf.startswith("checkpoint-") and parent:
            return f"{parent}-{leaf}"
        return leaf

    lora_weights = str(run.get("lora_weights") or "").strip().rstrip("/")
    if lora_weights:
        path = Path(lora_weights)
        leaf = path.name
        parent = path.parent.name
        if leaf.startswith("checkpoint-") and parent:
            return f"{parent}-{leaf}"
        return leaf

    base_model = str(run.get("base_model") or "").strip().rstrip("/")
    if base_model:
        path = Path(base_model)
        leaf = path.name
        parent = path.parent.name
        if leaf.startswith("checkpoint-") and parent:
            return f"{parent}-{leaf}"
        return leaf or base_model

    return "unknown_target"


runs = []
for summary_file in summary_files:
    p = Path(summary_file)
    if not p.exists():
        continue
    try:
        with open(p, "r") as f:
            run = json.load(f)
        run["summary_file"] = str(p)
        run["run_key"] = run_key(run)
        runs.append(run)
    except Exception:
        continue

score_buckets = {}
for run in runs:
    key = run.get("run_key", "unknown_target")
    score_buckets[key] = run

checkpoint_scores = []
for key, run in score_buckets.items():
    checkpoint_scores.append({
        "checkpoint": key,
        "overall_score": float(run.get("overall_score", 0.0)),
        "overall_leakage_rate": float(run.get("overall_leakage_rate", 1.0)),
        "overall_unsafe_rate": float(run.get("overall_unsafe_rate", run.get("overall_leakage_rate", 1.0))),
        "overall_refusal_rate": float(run.get("overall_refusal_rate", 0.0)),
        "prompt_metrics": run.get("prompt_metrics", {}),
        "pii_metrics": run.get("pii_metrics", {}),
        "label_metrics": run.get("label_metrics", {}),
        "data_type_metrics": run.get("data_type_metrics", {}),
    })

if task_mode == "safety":
    checkpoint_scores.sort(key=lambda x: (-x["overall_score"], x["overall_unsafe_rate"]))
else:
    checkpoint_scores.sort(key=lambda x: (x["overall_leakage_rate"], -x["overall_score"]))
ranking = [
    {
        "rank": i + 1,
        "checkpoint": item["checkpoint"],
        "overall_leakage_rate": item["overall_leakage_rate"],
        "overall_unsafe_rate": item["overall_unsafe_rate"],
        "overall_score": item["overall_score"],
    }
    for i, item in enumerate(checkpoint_scores)
]

summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "run_id": run_ts,
    "task_mode": task_mode,
    "prompt_types": prompt_types,
    "pii_indices": pii_indices,
    "runs": runs,
    "checkpoint_scores": checkpoint_scores,
    "ranking": ranking,
    "failed_runs": failed_runs,
}

merge_summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(merge_summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(str(merge_summary_path))
PY

echo "Merge summary saved to: ${MERGE_SUMMARY_PATH}"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Some runs failed: ${FAILED_RUNS[*]}" >&2
fi
