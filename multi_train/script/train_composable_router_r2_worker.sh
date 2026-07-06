#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ACCELERATE_BIN="${ACCELERATE_BIN:-/opt/conda/bin/accelerate}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"

SPEC1="${SPEC1:-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

TASK_DATASET_MAP_JSON="${TASK_DATASET_MAP_JSON:-multi_train/router_config/task_dataset_map_r2_train.json}"
SAMPLING_PROBS_JSON="${SAMPLING_PROBS_JSON:-multi_train/router_config/sampling_probs_r2_default.json}"
TASK_TRAIN_QUOTA_JSON="${TASK_TRAIN_QUOTA_JSON:-multi_train/router_config/task_train_quota_r2_conservative.json}"
BBQ_QA_BRIDGE_COUNT="${BBQ_QA_BRIDGE_COUNT:-0}"
BBQ_QA_BRIDGE_SAMPLING_PROB="${BBQ_QA_BRIDGE_SAMPLING_PROB:-}"
ROUTER_FEATURE_STATS_PATH="${ROUTER_FEATURE_STATS_PATH:-}"
BASE_SCORE_STATS_PATH="${BASE_SCORE_STATS_PATH:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"

OUTPUT_DIR="${OUTPUT_DIR:-multi_train/trainer_output/Llama3-8b-ComposableRouter-R2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
POSITIONS="${POSITIONS:-f7+l7}"
TARGET_LAYERS="${TARGET_LAYERS:--1}"
ROUTER_LAYERS="${ROUTER_LAYERS:-28 29 30}"
ROUTER_FEATURE_NAMES="${ROUTER_FEATURE_NAMES:-}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
MAX_TRAIN_SAMPLES_PER_TASK="${MAX_TRAIN_SAMPLES_PER_TASK:-}"
MAX_EVAL_SAMPLES_PER_TASK="${MAX_EVAL_SAMPLES_PER_TASK:-128}"
EVAL_HOLDOUT_RATIO="${EVAL_HOLDOUT_RATIO:-0.02}"
SEED="${SEED:-42}"

BASE_COMPOSITION_TEMPERATURE="${BASE_COMPOSITION_TEMPERATURE:-1.0}"
BASE_COMPOSITION_TOPK="${BASE_COMPOSITION_TOPK:-2}"
BASE_COMPAT_THRESHOLD="${BASE_COMPAT_THRESHOLD:-0.0}"
BASE_SCORE_SOURCE="${BASE_SCORE_SOURCE:-intervention_norm}"
BASE_SCORE_NORMALIZER="${BASE_SCORE_NORMALIZER:-log_zscore}"
ROUTER_TEMPERATURE="${ROUTER_TEMPERATURE:-1.5}"
POLICY_HIDDEN_DIM="${POLICY_HIDDEN_DIM:-128}"
POLICY_PROJECTION_DIM="${POLICY_PROJECTION_DIM:-128}"
USE_PRE_HIDDEN_STATE_FEATURE="${USE_PRE_HIDDEN_STATE_FEATURE:-true}"
PRE_HIDDEN_STATE_DIM="${PRE_HIDDEN_STATE_DIM:-64}"
ROUTER_KL_WEIGHT="${ROUTER_KL_WEIGHT:-0.0}"
ROUTER_KL_TARGET_PROB="${ROUTER_KL_TARGET_PROB:-0.8}"
BIAS_ROUTER_KL_STEREOTYPE_PROB="${BIAS_ROUTER_KL_STEREOTYPE_PROB:-0.6}"
BIAS_ROUTER_KL_TRUTH_PROB="${BIAS_ROUTER_KL_TRUTH_PROB:-0.25}"
BBQ_QA_BRIDGE_KL_TRUTH_PROB="${BBQ_QA_BRIDGE_KL_TRUTH_PROB:-0.45}"
BBQ_QA_BRIDGE_KL_STEREOTYPE_PROB="${BBQ_QA_BRIDGE_KL_STEREOTYPE_PROB:-0.35}"
ROUTER_FEATURE_CLIP="${ROUTER_FEATURE_CLIP:-5.0}"
ROUTER_FEATURE_EPS="${ROUTER_FEATURE_EPS:-1e-6}"
COMPAT_IMPL="${COMPAT_IMPL:-optimized}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 10000)))}"

read -r -a TARGET_LAYER_ARR <<< "${TARGET_LAYERS}"
read -r -a ROUTER_LAYER_ARR <<< "${ROUTER_LAYERS}"
read -r -a ROUTER_FEATURE_NAME_ARR <<< "${ROUTER_FEATURE_NAMES}"

EXTRA_ARGS=()
if [[ -n "${MAX_TRAIN_SAMPLES_PER_TASK}" ]]; then
  EXTRA_ARGS+=(--max_train_samples_per_task "${MAX_TRAIN_SAMPLES_PER_TASK}")
fi
if [[ -n "${BBQ_QA_BRIDGE_SAMPLING_PROB}" ]]; then
  EXTRA_ARGS+=(--bbq_qa_bridge_sampling_prob "${BBQ_QA_BRIDGE_SAMPLING_PROB}")
fi
if [[ -n "${ROUTER_FEATURE_NAMES}" ]]; then
  EXTRA_ARGS+=(--router_feature_names "${ROUTER_FEATURE_NAME_ARR[@]}")
fi

"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  multi_train/train_composable_router.py \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${BASE_MODEL}" \
  --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}" \
  --task_dataset_map_json "${TASK_DATASET_MAP_JSON}" \
  --sampling_probs_json "${SAMPLING_PROBS_JSON}" \
  --task_train_quota_json "${TASK_TRAIN_QUOTA_JSON}" \
  --bbq_qa_bridge_count "${BBQ_QA_BRIDGE_COUNT}" \
  --router_feature_stats_path "${ROUTER_FEATURE_STATS_PATH}" \
  --base_score_stats_path "${BASE_SCORE_STATS_PATH}" \
  --target_layers "${TARGET_LAYER_ARR[@]}" \
  --router_layers "${ROUTER_LAYER_ARR[@]}" \
  --positions "${POSITIONS}" \
  --base_composition_temperature "${BASE_COMPOSITION_TEMPERATURE}" \
  --base_composition_topk "${BASE_COMPOSITION_TOPK}" \
  --base_compat_threshold "${BASE_COMPAT_THRESHOLD}" \
  --base_score_source "${BASE_SCORE_SOURCE}" \
  --base_score_normalizer "${BASE_SCORE_NORMALIZER}" \
  --router_temperature "${ROUTER_TEMPERATURE}" \
  --policy_hidden_dim "${POLICY_HIDDEN_DIM}" \
  --policy_projection_dim "${POLICY_PROJECTION_DIM}" \
  --use_pre_hidden_state_feature "${USE_PRE_HIDDEN_STATE_FEATURE}" \
  --pre_hidden_state_dim "${PRE_HIDDEN_STATE_DIM}" \
  --router_kl_weight "${ROUTER_KL_WEIGHT}" \
  --router_kl_target_prob "${ROUTER_KL_TARGET_PROB}" \
  --bias_router_kl_stereotype_prob "${BIAS_ROUTER_KL_STEREOTYPE_PROB}" \
  --bias_router_kl_truth_prob "${BIAS_ROUTER_KL_TRUTH_PROB}" \
  --bbq_qa_bridge_kl_truth_prob "${BBQ_QA_BRIDGE_KL_TRUTH_PROB}" \
  --bbq_qa_bridge_kl_stereotype_prob "${BBQ_QA_BRIDGE_KL_STEREOTYPE_PROB}" \
  --router_feature_clip "${ROUTER_FEATURE_CLIP}" \
  --router_feature_eps "${ROUTER_FEATURE_EPS}" \
  --compat_impl "${COMPAT_IMPL}" \
  --max_eval_samples_per_task "${MAX_EVAL_SAMPLES_PER_TASK}" \
  --eval_holdout_ratio "${EVAL_HOLDOUT_RATIO}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --save_strategy "${SAVE_STRATEGY}" \
  --eval_strategy "${EVAL_STRATEGY}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bf16 True \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
