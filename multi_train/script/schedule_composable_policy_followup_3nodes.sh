#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARTITIONS_RAW="${PARTITIONS:-${PARTITION:-dexmanip,ftmanip}}"
PARTITIONS_NORM="${PARTITIONS_RAW//,/ }"
read -r -a PARTITION_ARR <<< "${PARTITIONS_NORM}"
if [[ "${#PARTITION_ARR[@]}" -eq 0 ]]; then
  echo "No partition configured. Set PARTITION or PARTITIONS." >&2
  exit 1
fi
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

SPEC1="${SPEC1:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}"
SPEC2="${SPEC2:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}"
SPEC3="${SPEC3:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}"
SPEC4="${SPEC4:-${WORKSPACE_CONT}/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
STATS_INT_SHARED="${STATS_INT_SHARED:-multi_train/calibration/train_input/stats/intervention_stats_shared_train_input.json}"
STATS_INT_SPECIALIST="${STATS_INT_SPECIALIST:-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/scheduler_policy_followup_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"

launch_group() {
  local group_name="$1"
  local script_name="$2"
  local partition="$3"
  local host_log_file="${HOST_LOG_DIR}/${group_name}.log"

  echo "[launch] ${group_name} partition=${partition} script=${script_name} log=${host_log_file}"

  srun \
    -p "${partition}" \
    --gres="${GRES}" \
    --job-name="${group_name}" \
    ${SRUN_EXTRA_ARGS} \
    apptainer exec \
      --cleanenv \
      --nv \
      --bind /mnt:/mnt \
      --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
      "${IMAGE_PATH}" \
      bash -c "
        cd '${WORKSPACE_CONT}' &&
        export PYTHON_BIN='${PYTHON_BIN}' &&
        export HF_HUB_OFFLINE='${HF_HUB_OFFLINE}' &&
        export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE}' &&
        export HF_DATASETS_OFFLINE='${HF_DATASETS_OFFLINE}' &&
        export STATS_INT_SHARED='${STATS_INT_SHARED}' &&
        export STATS_INT_SPECIALIST='${STATS_INT_SPECIALIST}' &&
        export SPEC1='${SPEC1}' &&
        export SPEC2='${SPEC2}' &&
        export SPEC3='${SPEC3}' &&
        export SPEC4='${SPEC4}' &&
        export LOG_DIR='multi_train/logs/${group_name}_${RUN_TAG}' &&
        bash multi_train/script/${script_name}
      " \
    > "${host_log_file}" 2>&1 &
}

echo "PARTITIONS=${PARTITIONS_RAW}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
echo "STATS_INT_SHARED=${STATS_INT_SHARED}"
echo "STATS_INT_SPECIALIST=${STATS_INT_SPECIALIST}"
echo "RUN_TAG=${RUN_TAG}"
echo "HOST_LOG_DIR=${HOST_LOG_DIR}"

GROUP_NAMES=(
  "policy_followup_toxicity"
  "policy_followup_ethics"
  "policy_followup_truth"
)
SCRIPT_NAMES=(
  "evaluate_composable_toxicity_policy_followup.sh"
  "evaluate_composable_ethics_policy_followup.sh"
  "evaluate_composable_truth_policy_followup.sh"
)

for idx in "${!GROUP_NAMES[@]}"; do
  partition="${PARTITION_ARR[$((idx % ${#PARTITION_ARR[@]}))]}"
  launch_group "${GROUP_NAMES[$idx]}" "${SCRIPT_NAMES[$idx]}" "${partition}"
done

wait

echo "All follow-up policy jobs finished."
echo "Host logs: ${HOST_LOG_DIR}"
