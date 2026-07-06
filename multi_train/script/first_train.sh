subspace=stereotype
model_name_or_path="models/llama2-7b/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
safety_dataset_path="${SAFETY_DATASET_PATH:-./dataset/alignment_wildjailbreak_safety_compact_30k}"

default_run_suffix=""
if [[ "${subspace}" == "safety" && -z "${RUN_SUFFIX:-}" && -z "${run_suffix:-}" ]]; then
    default_run_suffix="compact30k"
fi

run_suffix="${run_suffix:-${RUN_SUFFIX:-${default_run_suffix}}}"
log_suffix="${log_suffix:-${LOG_SUFFIX:-${run_suffix}}}"

if [[ "${model_name_or_path}" == */snapshots/* ]]; then
    model_tag="${model_name_or_path%/snapshots/*}"
    model_tag="${model_tag##*/}"
else
    model_tag="${model_name_or_path##*/}"
fi

if [[ -n "${run_suffix}" ]]; then
    run_tag="${model_tag}-Loreft_${subspace}-${run_suffix}"
else
    run_tag="${model_tag}-Loreft_${subspace}-3"
fi

if [[ -n "${log_suffix}" ]]; then
    log_tag="${model_tag}_Loreft_${subspace}_${log_suffix}"
else
    log_tag="${model_tag}_Loreft_${subspace}"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-shichaojian}/triton-cache/${HOSTNAME:-localhost}/${SLURM_JOB_ID:-manual}/${run_tag}}"
mkdir -p "${TRITON_CACHE_DIR}"

save_strategy="${SAVE_STRATEGY:-epoch}"
eval_strategy="${EVAL_STRATEGY:-epoch}"
save_steps="${SAVE_STEPS:-}"
eval_steps="${EVAL_STEPS:-}"

train_extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    train_extra_args+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${save_steps}" ]]; then
    train_extra_args+=(--save_steps "${save_steps}")
fi
if [[ -n "${eval_steps}" ]]; then
    train_extra_args+=(--eval_steps "${eval_steps}")
fi

if [[ "${subspace}" == "safety" ]]; then
    export SAFETY_DATASET_PATH="${safety_dataset_path}"
    if [[ ! -d "${SAFETY_DATASET_PATH}" ]]; then
        echo "Safety dataset path does not exist: ${SAFETY_DATASET_PATH}" >&2
        exit 1
    fi
    echo "Using SAFETY_DATASET_PATH=${SAFETY_DATASET_PATH}"
fi

echo "run_tag=${run_tag}"
echo "log_file=./first_train_${log_tag}.log"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "save_strategy=${save_strategy}"
echo "eval_strategy=${eval_strategy}"
if [[ -n "${save_steps}" ]]; then
    echo "save_steps=${save_steps}"
fi
if [[ -n "${eval_steps}" ]]; then
    echo "eval_steps=${eval_steps}"
fi

accelerate launch --num_processes=8 multi_train/train.py \
    --output_dir "multi_train/trainer_output/${run_tag}" \
    --model_name_or_path "${model_name_or_path}" \
    --per_device_train_batch_size 16 \
    --subspace_rank 8 \
    --warmup_ratio 0.1 \
    --weight_decay 0 \
    --learning_rate 9e-4 \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps 2 \
    --subtask "${subspace}" \
    --num_train_epochs 6 \
    --position f7+l7 \
    --dropout 0.05 \
    --target_layers -1 \
    --model_max_length 512 \
    --save_strategy "${save_strategy}" \
    --eval_strategy "${eval_strategy}" \
    "${train_extra_args[@]}" \
    > "./first_train_${log_tag}.log" 2>&1
    # learning_rate_scheduler_type consine
