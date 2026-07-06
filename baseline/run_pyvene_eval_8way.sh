#!/bin/bash
set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo "Please run this script with bash." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-truth}"
BASELINE="${BASELINE:-caa}"
BASE_MODEL="${BASE_MODEL:-models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b}"
RUN_PREFIX="${RUN_PREFIX:-${TASK}_${BASELINE}_scan8}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
LAYERS="${LAYERS:-14 26}"
ALPHAS="${ALPHAS:-1.0 0.1}"
PROMPT_MODES="${PROMPT_MODES:-promptlast genall}"
ITI_TOP_K_HEADS="${ITI_TOP_K_HEADS:-48}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

normalize_model_name() {
  local path_value="$1"
  local normalized="${path_value%/}"
  local stripped="${normalized#models/}"
  if [[ "${stripped}" != "${normalized}" ]]; then
    echo "${stripped%%/*}" | tr '[:upper:]' '[:lower:]'
    return 0
  fi
  if [[ "${normalized}" == */snapshots/* ]]; then
    local before_snapshots="${normalized%/snapshots/*}"
    echo "${before_snapshots##*/}" | tr '[:upper:]' '[:lower:]'
    return 0
  fi
  echo "${normalized##*/}" | tr '[:upper:]' '[:lower:]'
}

launch_job() {
  local gpu_id="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" &
}

artifact_root_for() {
  local model_name="$1"
  local layer="$2"
  case "${BASELINE}" in
    caa)
      echo "baseline/caa_pyvene_0/artifacts/vectors/${model_name}/${TASK}/${TASK}_l14_l26_caa_l${layer}"
      ;;
    iti)
      echo "baseline/iti_pyvene_0/artifacts/interventions/${model_name}/${TASK}/${TASK}_l14_l26_iti_l${layer}_k${ITI_TOP_K_HEADS}"
      ;;
    repe)
      echo "baseline/repe_pyvene_0/artifacts/vectors/${model_name}/${TASK}/${TASK}_l14_l26_repe_l${layer}"
      ;;
    *)
      echo "Unsupported BASELINE=${BASELINE}" >&2
      exit 1
      ;;
  esac
}

model_name="$(normalize_model_name "${BASE_MODEL}")"
gpu_arr=(${GPU_IDS})
layer_arr=(${LAYERS})
alpha_arr=(${ALPHAS})
prompt_arr=(${PROMPT_MODES})

if (( ${#gpu_arr[@]} < 8 )); then
  echo "Need at least 8 GPU ids in GPU_IDS, got: ${GPU_IDS}" >&2
  exit 1
fi

job_idx=0
for layer in "${layer_arr[@]}"; do
  for alpha in "${alpha_arr[@]}"; do
    for prompt_mode in "${prompt_arr[@]}"; do
      gpu_id="${gpu_arr[$job_idx]}"
      run_name="${RUN_PREFIX}_l${layer}_a${alpha}_${prompt_mode}"
      artifact_dir="$(artifact_root_for "${model_name}" "${layer}")"
      if [[ ! -d "${artifact_dir}" ]]; then
        echo "Missing artifact dir: ${artifact_dir}" >&2
        exit 1
      fi
      case "${prompt_mode}" in
        promptlast)
          intervene_on_prompt=1
          base_unit_location=-1
          token_strategy=last
          ;;
        genall)
          intervene_on_prompt=0
          base_unit_location=
          token_strategy=all
          ;;
        *)
          echo "Unsupported prompt mode: ${prompt_mode}" >&2
          exit 1
          ;;
      esac
      case "${BASELINE}" in
        caa)
          launch_job "${gpu_id}" env             PYTHON_BIN="${PYTHON_BIN}"             EVAL_NAME="${run_name}"             TASK="${TASK}"             BASE_MODEL="${BASE_MODEL}"             CAA_VECTOR_DIR="${artifact_dir}"             CAA_LAYERS="${layer}"             CAA_ALPHA="${alpha}"             CAA_INTERVENE_ON_PROMPT="${intervene_on_prompt}"             CAA_BASE_UNIT_LOCATION="${base_unit_location}"             DEVICE="cuda:0"             bash baseline/caa_pyvene_0/run_eval_experiment.sh
          ;;
        iti)
          launch_job "${gpu_id}" env             PYTHON_BIN="${PYTHON_BIN}"             EVAL_NAME="${run_name}"             TASK="${TASK}"             BASE_MODEL="${BASE_MODEL}"             ITI_ARTIFACT_DIR="${artifact_dir}"             ITI_LAYERS="${layer}"             ITI_ALPHA="${alpha}"             ITI_TOKEN_STRATEGY="${token_strategy}"             ITI_INCLUDE_PROMPT="${intervene_on_prompt}"             ITI_BASE_UNIT_LOCATION="${base_unit_location}"             DEVICE="cuda:0"             bash baseline/iti_pyvene_0/run_eval_experiment.sh
          ;;
        repe)
          launch_job "${gpu_id}" env             PYTHON_BIN="${PYTHON_BIN}"             EVAL_NAME="${run_name}"             TASK="${TASK}"             BASE_MODEL="${BASE_MODEL}"             REPE_VECTOR_DIR="${artifact_dir}"             REPE_LAYERS="${layer}"             REPE_ALPHA="${alpha}"             REPE_TOKEN_STRATEGY="${token_strategy}"             REPE_INTERVENE_ON_PROMPT="${intervene_on_prompt}"             REPE_BASE_UNIT_LOCATION="${base_unit_location}"             DEVICE="cuda:0"             bash baseline/repe_pyvene_0/run_eval_experiment.sh
          ;;
      esac
      job_idx=$((job_idx + 1))
    done
  done
done

wait

echo "Finished 8-way eval for TASK=${TASK}, BASELINE=${BASELINE}, RUN_PREFIX=${RUN_PREFIX}"
