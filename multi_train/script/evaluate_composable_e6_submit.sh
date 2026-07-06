#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

format_tag() {
  local value="${1:-unknown}"
  value="${value//-/m}"
  value="${value//./p}"
  value="${value// /}"
  echo "${value}"
}

PARTITION="${PARTITION:-eailab_os}"
GRES="${GRES:-gpu:8}"
IMAGE_PATH="${IMAGE_PATH:-/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/mnt/petrelfs/${USER}/multi-reft}"
WORKSPACE_CONT="${WORKSPACE_CONT:-/workspace/multi-reft}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
COMPOSITION_TEMPERATURE_VALUE="${COMPOSITION_TEMPERATURE:-1.0}"
COMPOSITION_TOPK_VALUE="${COMPOSITION_TOPK:-2}"
E6_PARAM_TAG="${E6_PARAM_TAG:-temp$(format_tag "${COMPOSITION_TEMPERATURE_VALUE}")-topk$(format_tag "${COMPOSITION_TOPK_VALUE}")}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_LOG_DIR="${HOST_LOG_DIR:-${ROOT_DIR}/multi_train/logs/evaluate_composable_e6_${E6_PARAM_TAG}_${RUN_TAG}}"
mkdir -p "${HOST_LOG_DIR}"
HOST_LOG_FILE="${HOST_LOG_FILE:-${HOST_LOG_DIR}/evaluate_composable_e6.log}"

echo "PARTITION=${PARTITION}"
echo "GRES=${GRES}"
echo "IMAGE_PATH=${IMAGE_PATH}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "WORKSPACE_CONT=${WORKSPACE_CONT}"
echo "E6_PARAM_TAG=${E6_PARAM_TAG}"
echo "HOST_LOG_FILE=${HOST_LOG_FILE}"
echo "SRUN_EXTRA_ARGS=${SRUN_EXTRA_ARGS}"
echo "SECOND_COMPOSITION_TEMPERATURE=${SECOND_COMPOSITION_TEMPERATURE-}"
echo "SECOND_COMPOSITION_TOPK=${SECOND_COMPOSITION_TOPK-}"
echo "PRIMARY_VISIBLE_DEVICES=${PRIMARY_VISIBLE_DEVICES-0,1,2,3}"
echo "SECOND_VISIBLE_DEVICES=${SECOND_VISIBLE_DEVICES-4,5,6,7}"

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
      export E6_PARAM_TAG='${E6_PARAM_TAG}' &&
      export PYTHON_BIN='${PYTHON_BIN-/opt/conda/bin/python}' &&
      export BASE_MODEL='${BASE_MODEL-models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920}' &&
      export SPEC1='${SPEC1-multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model}' &&
      export SPEC2='${SPEC2-multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model}' &&
      export SPEC3='${SPEC3-multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model}' &&
      export SPEC4='${SPEC4-multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model}' &&
      export SCORE_STATS_PATH='${SCORE_STATS_PATH-multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json}' &&
      export SCORE_SOURCE='${SCORE_SOURCE-intervention_norm}' &&
      export SCORE_NORMALIZER='${SCORE_NORMALIZER-log_zscore}' &&
      export COMPOSITION_METHOD='${COMPOSITION_METHOD-compat_filtered_topk}' &&
      export COMPOSE_DOMAIN='${COMPOSE_DOMAIN-output}' &&
      export COMPOSITION_TEMPERATURE='${COMPOSITION_TEMPERATURE-1.0}' &&
      export COMPOSITION_TOPK='${COMPOSITION_TOPK-2}' &&
      export COMPAT_THRESHOLD='${COMPAT_THRESHOLD-0.0}' &&
      export TARGET_LAYERS='${TARGET_LAYERS--1}' &&
      export ENABLE_DEBUG_CACHE='${ENABLE_DEBUG_CACHE-1}' &&
      export MERGE_SUMMARY_PATH='${MERGE_SUMMARY_PATH-}' &&
      export SUMMARY_DIR='${SUMMARY_DIR-multi_train/eval_router/summaries}' &&
      export TASKS='${TASKS-truth bias ethics toxicity}' &&
      export GPU_IDS='${GPU_IDS-0 1 2 3}' &&
      export STAGGER_SECONDS='${STAGGER_SECONDS-10}' &&
      export LOG_DIR='${LOG_DIR-}' &&
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
      export PII_INDICES='${PII_INDICES-}' &&
      export DATA_FILE='${DATA_FILE-}' &&
      export NUM_PROMPTS='${NUM_PROMPTS-}' &&
      export CONTEXT_EXAMPLES='${CONTEXT_EXAMPLES-}' &&
      export RUN_STATISTICS='${RUN_STATISTICS-}' &&
      export DATASETS='${DATASETS-}' &&
      export PROMPTS='${PROMPTS-}' &&
      export DATASET_ROOT='${DATASET_ROOT-}' &&
      export RUN_ANALYSIS='${RUN_ANALYSIS-}' &&
      export ANALYSIS_BATCH_SIZE='${ANALYSIS_BATCH_SIZE-}' &&
      export DETOXIFY_MODEL='${DETOXIFY_MODEL-}' &&
      export ANALYSIS_DEVICE='${ANALYSIS_DEVICE-}' &&
      export SYS_PROMPT_TYPES='${SYS_PROMPT_TYPES-}' &&
      export TRUTH_DATASET='${TRUTH_DATASET-}' &&
      export TRUTH_BATCH_SIZE='${TRUTH_BATCH_SIZE-}' &&
      export TRUTH_POSITIONS='${TRUTH_POSITIONS-}' &&
      export TRUTH_GREEDY_DECODING='${TRUTH_GREEDY_DECODING-}' &&
      export TRUTH_TRUTHFUL_SCORE_PENALTY='${TRUTH_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
      export BIAS_DATASET='${BIAS_DATASET-}' &&
      export BIAS_BATCH_SIZE='${BIAS_BATCH_SIZE-}' &&
      export BIAS_POSITIONS='${BIAS_POSITIONS-}' &&
      export BIAS_GREEDY_DECODING='${BIAS_GREEDY_DECODING-}' &&
      export BIAS_SYS_PROMPT_TYPES='${BIAS_SYS_PROMPT_TYPES-}' &&
      export BIAS_TRUTHFUL_SCORE_PENALTY='${BIAS_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
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
      export ETHICS_TRUTHFUL_SCORE_PENALTY='${ETHICS_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
      export PRIVACY_BATCH_SIZE='${PRIVACY_BATCH_SIZE-}' &&
      export PRIVACY_POSITIONS='${PRIVACY_POSITIONS-}' &&
      export PRIVACY_GREEDY_DECODING='${PRIVACY_GREEDY_DECODING-}' &&
      export PRIVACY_PROMPT_TYPES='${PRIVACY_PROMPT_TYPES-}' &&
      export PRIVACY_PII_INDICES='${PRIVACY_PII_INDICES-}' &&
      export PRIVACY_DATA_FILE='${PRIVACY_DATA_FILE-}' &&
      export PRIVACY_NUM_PROMPTS='${PRIVACY_NUM_PROMPTS-}' &&
      export PRIVACY_CONTEXT_EXAMPLES='${PRIVACY_CONTEXT_EXAMPLES-}' &&
      export PRIVACY_N_GENERATIONS='${PRIVACY_N_GENERATIONS-}' &&
      export PRIVACY_MAX_TOKENS='${PRIVACY_MAX_TOKENS-}' &&
      export PRIVACY_TEMPERATURE='${PRIVACY_TEMPERATURE-}' &&
      export PRIVACY_RUN_STATISTICS='${PRIVACY_RUN_STATISTICS-}' &&
      export PRIVACY_TRUTHFUL_SCORE_PENALTY='${PRIVACY_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
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
      export TOXICITY_ANALYSIS_BATCH_SIZE='${TOXICITY_ANALYSIS_BATCH_SIZE-}' &&
      export TOXICITY_DETOXIFY_MODEL='${TOXICITY_DETOXIFY_MODEL-}' &&
      export TOXICITY_ANALYSIS_DEVICE='${TOXICITY_ANALYSIS_DEVICE-}' &&
      export TOXICITY_TRUTHFUL_SCORE_PENALTY='${TOXICITY_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
      export PRIMARY_VISIBLE_DEVICES='${PRIMARY_VISIBLE_DEVICES-0,1,2,3}' &&
      export PRIMARY_GPU_IDS='${PRIMARY_GPU_IDS-${GPU_IDS-0 1 2 3}}' &&
      export PRIMARY_LOG_DIR='${PRIMARY_LOG_DIR-${LOG_DIR-}}' &&
      export PRIMARY_MERGE_SUMMARY_PATH='${PRIMARY_MERGE_SUMMARY_PATH-${MERGE_SUMMARY_PATH-}}' &&
      export PRIMARY_E6_PARAM_TAG='${PRIMARY_E6_PARAM_TAG-${E6_PARAM_TAG}}' &&
      export SECOND_COMPOSITION_TEMPERATURE='${SECOND_COMPOSITION_TEMPERATURE-}' &&
      export SECOND_COMPOSITION_TOPK='${SECOND_COMPOSITION_TOPK-}' &&
      export SECOND_VISIBLE_DEVICES='${SECOND_VISIBLE_DEVICES-4,5,6,7}' &&
      export SECOND_GPU_IDS='${SECOND_GPU_IDS-0 1 2 3}' &&
      export SECOND_LOG_DIR='${SECOND_LOG_DIR-}' &&
      export SECOND_MERGE_SUMMARY_PATH='${SECOND_MERGE_SUMMARY_PATH-}' &&
      export SECOND_E6_PARAM_TAG='${SECOND_E6_PARAM_TAG-}' &&
      export SECOND_TRUTHFUL_SCORE_PENALTY='${SECOND_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}' &&
      export SECOND_TOXICITY_TRUTHFUL_SCORE_PENALTY='${SECOND_TOXICITY_TRUTHFUL_SCORE_PENALTY-${TOXICITY_TRUTHFUL_SCORE_PENALTY-${TRUTHFUL_SCORE_PENALTY-}}}' &&
      format_tag() {
        local value=\"\${1:-unknown}\"
        value=\"\${value//-/m}\"
        value=\"\${value//./p}\"
        value=\"\${value// /}\"
        echo \"\${value}\"
      } &&
      run_group() {
        local visible_devices=\"\$1\"
        local gpu_ids=\"\$2\"
        local composition_temperature=\"\$3\"
        local composition_topk=\"\$4\"
        local truthful_penalty=\"\$5\"
        local toxicity_truthful_penalty=\"\$6\"
        local merge_summary_path=\"\$7\"
        local log_dir=\"\$8\"
        local explicit_tag=\"\$9\"
        local computed_tag=\"temp\$(format_tag \"\${composition_temperature}\")-topk\$(format_tag \"\${composition_topk}\")\"
        local e6_param_tag=\"\${explicit_tag:-\${computed_tag}}\"
        echo \"[group] CUDA_VISIBLE_DEVICES=\${visible_devices} GPU_IDS=\${gpu_ids} temp=\${composition_temperature} topk=\${composition_topk} tag=\${e6_param_tag}\"
        CUDA_VISIBLE_DEVICES=\"\${visible_devices}\" \\
        GPU_IDS=\"\${gpu_ids}\" \\
        COMPOSITION_TEMPERATURE=\"\${composition_temperature}\" \\
        COMPOSITION_TOPK=\"\${composition_topk}\" \\
        E6_PARAM_TAG=\"\${e6_param_tag}\" \\
        TRUTHFUL_SCORE_PENALTY=\"\${truthful_penalty}\" \\
        TOXICITY_TRUTHFUL_SCORE_PENALTY=\"\${toxicity_truthful_penalty}\" \\
        MERGE_SUMMARY_PATH=\"\${merge_summary_path}\" \\
        LOG_DIR=\"\${log_dir}\" \\
        bash multi_train/script/evaluate_composable_e6_all.sh
      } &&
      if [ -n \"\${SECOND_COMPOSITION_TEMPERATURE}\" ] || [ -n \"\${SECOND_COMPOSITION_TOPK}\" ]; then
        second_temp=\"\${SECOND_COMPOSITION_TEMPERATURE:-\${COMPOSITION_TEMPERATURE}}\"
        second_topk=\"\${SECOND_COMPOSITION_TOPK:-\${COMPOSITION_TOPK}}\"
        run_group \"\${PRIMARY_VISIBLE_DEVICES}\" \"\${PRIMARY_GPU_IDS}\" \"\${COMPOSITION_TEMPERATURE}\" \"\${COMPOSITION_TOPK}\" \"\${TRUTHFUL_SCORE_PENALTY-}\" \"\${TOXICITY_TRUTHFUL_SCORE_PENALTY-}\" \"\${PRIMARY_MERGE_SUMMARY_PATH}\" \"\${PRIMARY_LOG_DIR}\" \"\${PRIMARY_E6_PARAM_TAG}\" &
        pid1=\$!
        run_group \"\${SECOND_VISIBLE_DEVICES}\" \"\${SECOND_GPU_IDS}\" \"\${second_temp}\" \"\${second_topk}\" \"\${SECOND_TRUTHFUL_SCORE_PENALTY}\" \"\${SECOND_TOXICITY_TRUTHFUL_SCORE_PENALTY}\" \"\${SECOND_MERGE_SUMMARY_PATH}\" \"\${SECOND_LOG_DIR}\" \"\${SECOND_E6_PARAM_TAG}\" &
        pid2=\$!
        wait \${pid1}
        wait \${pid2}
      else
        run_group \"\${PRIMARY_VISIBLE_DEVICES}\" \"\${PRIMARY_GPU_IDS}\" \"\${COMPOSITION_TEMPERATURE}\" \"\${COMPOSITION_TOPK}\" \"\${TRUTHFUL_SCORE_PENALTY-}\" \"\${TOXICITY_TRUTHFUL_SCORE_PENALTY-}\" \"\${PRIMARY_MERGE_SUMMARY_PATH}\" \"\${PRIMARY_LOG_DIR}\" \"\${PRIMARY_E6_PARAM_TAG}\"
      fi
    " \
  > "${HOST_LOG_FILE}" 2>&1

echo "Composable E6 evaluation job finished."
echo "Host log: ${HOST_LOG_FILE}"
