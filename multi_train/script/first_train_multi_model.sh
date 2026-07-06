# truthful stereotype moral toxicity
subspace="${subspace:-toxicity}"
model_name_or_path="${model_name_or_path:-models/qwen25-14b/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9}"
# models/mistral-7b-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24
# models/mistral-7b-v0.3/snapshots/b67d6a03ca097c5122fa65904fce0413500bf8c8
# models/qwen3-4b/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
# models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b
# models/qwen25-7b/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# models/qwen25-14b/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9
if [[ "${model_name_or_path}" == */snapshots/* ]]; then
    model_tag="${model_name_or_path%/snapshots/*}"
    model_tag="${model_tag##*/}"
else
    model_tag="${model_name_or_path##*/}"
fi

case "${model_tag}" in
    llama3-8b)
        per_device_train_batch_size="${per_device_train_batch_size:-16}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-9e-4}"
        ;;
    mistral-7b-v0.1|mistral-7b-v0.3)
        per_device_train_batch_size="${per_device_train_batch_size:-16}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-8e-4}"
        ;;
    qwen25-7b)
        per_device_train_batch_size="${per_device_train_batch_size:-16}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-9e-4}"
        ;;
    qwen25-3b|qwen3-4b)
        per_device_train_batch_size="${per_device_train_batch_size:-16}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-1e-3}"
        ;;
    qwen25-14b)
        per_device_train_batch_size="${per_device_train_batch_size:-8}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-6e-4}"
        ;;
    *)
        per_device_train_batch_size="${per_device_train_batch_size:-16}"
        gradient_accumulation_steps="${gradient_accumulation_steps:-2}"
        learning_rate="${learning_rate:-8e-4}"
        ;;
esac

subspace_rank="${subspace_rank:-8}"
warmup_ratio="${warmup_ratio:-0.0125}"
weight_decay="${weight_decay:-0}"
num_train_epochs="${num_train_epochs:-6}"
position="${position:-f7+l7}"
dropout="${dropout:-0.05}"
target_layers="${target_layers:--1}"
model_max_length="${model_max_length:-512}"
num_processes="${num_processes:-8}"
run_suffix="${run_suffix:-}"

lr_tag="${learning_rate//./p}"
lr_tag="${lr_tag//+}"
experiment_tag="${model_tag}_Loreft_${subspace}_lr${lr_tag}_bs${per_device_train_batch_size}_ga${gradient_accumulation_steps}_ep${num_train_epochs}"
if [[ -n "${run_suffix}" ]]; then
    run_tag="${experiment_tag}_${run_suffix}"
else
    run_tag="${experiment_tag}"
fi

if [[ "${target_layers}" == *" "* ]]; then
    read -r -a target_layer_arr <<< "${target_layers}"
else
    target_layer_arr=("${target_layers}")
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-shichaojian}/triton-cache/${HOSTNAME:-localhost}/${SLURM_JOB_ID:-manual}/${run_tag}}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "model_tag=${model_tag}"
echo "model_name_or_path=${model_name_or_path}"
echo "subspace=${subspace}"
echo "per_device_train_batch_size=${per_device_train_batch_size}"
echo "gradient_accumulation_steps=${gradient_accumulation_steps}"
echo "learning_rate=${learning_rate}"
echo "run_tag=${run_tag}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"

accelerate launch --num_processes="${num_processes}" multi_train/train.py \
    --output_dir "multi_train/trainer_output/${run_tag}" \
    --model_name_or_path "${model_name_or_path}" \
    --per_device_train_batch_size "${per_device_train_batch_size}" \
    --subspace_rank "${subspace_rank}" \
    --warmup_ratio "${warmup_ratio}" \
    --weight_decay "${weight_decay}" \
    --learning_rate "${learning_rate}" \
    --lr_scheduler_type linear \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --subtask "${subspace}" \
    --num_train_epochs "${num_train_epochs}" \
    --position "${position}" \
    --dropout "${dropout}" \
    --target_layers "${target_layer_arr[@]}" \
    --model_max_length "${model_max_length}" \
    --run_name "${run_tag}" \
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    > "./first_train_${run_tag}.log" 2>&1
    # learning_rate_scheduler_type consine
