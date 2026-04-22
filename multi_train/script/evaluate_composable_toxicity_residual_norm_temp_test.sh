#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-multi_train/logs/composable_toxicity_residual_norm_temp_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

BASE_MODEL="${BASE_MODEL:-../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}"
SPEC1="${SPEC1:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

SHARED_STATS="${SHARED_STATS:-multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json}"
SPECIAL_STATS="${SPECIAL_STATS:-multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TEMPS="${TEMPS:-0.5 1 2 4 8 16 32 64}"
DEVICES="${DEVICES:-cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7}"

read -r -a TEMP_ARR <<< "${TEMPS}"
read -r -a DEVICE_ARR <<< "${DEVICES}"

run_eval() {
  local device="$1"
  local method="$2"
  local stats_path="$3"
  local temp="$4"
  local log_file="$5"

  python multi_train/eval_toxicity/toxicity_exp.py \
    --base_model "${BASE_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --reft_specialists "${SPEC1}" "${SPEC2}" "${SPEC3}" "${SPEC4}" \
    --compose_domain output \
    --composition_method "${method}" \
    --composition_temperature "${temp}" \
    --score_stats_path "${stats_path}" \
    --target_layers -1 \
    --positions 7 \
    --device "${device}" \
    > "${log_file}" 2>&1
}

job_idx=0
for temp in "${TEMP_ARR[@]}"; do
  for cfg in \
    "E1 sh_scaled residual_scaled_softmax ${SHARED_STATS}" \
    "E2 sh_logz residual_logz_softmax ${SHARED_STATS}" \
    "E3 sp_scaled residual_scaled_softmax ${SPECIAL_STATS}" \
    "E4 sp_logz residual_logz_softmax ${SPECIAL_STATS}"
  do
    read -r exp_id tag method stats_path <<< "${cfg}"
    device="${DEVICE_ARR[$((job_idx % ${#DEVICE_ARR[@]}))]}"
    log_file="${LOG_DIR}/${exp_id}_${tag}_t${temp}.log"

    echo "[launch] ${exp_id} ${tag} temp=${temp} device=${device}"
    run_eval "${device}" "${method}" "${stats_path}" "${temp}" "${log_file}" &

    job_idx=$((job_idx + 1))
    if (( job_idx % ${#DEVICE_ARR[@]} == 0 )); then
      wait
    fi
  done
done

wait
echo "Done. Logs written to: ${LOG_DIR}"
