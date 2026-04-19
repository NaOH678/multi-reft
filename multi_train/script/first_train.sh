subspace=truthful
export WANDB_MODE="${WANDB_MODE:-offline}"

accelerate launch --num_processes=8 multi_train/train.py \
    --output_dir "multi_train/trainer_output/Llama3-8b-Loreft_${subspace}_4" \
    --model_name_or_path ../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    --per_device_train_batch_size 16 \
    --subspace_rank 8 \
    --warmup_ratio 0.00625 \
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
    --save_strategy "epoch" \
    --eval_strategy "epoch" \
    > "./first_train_Loreft_${subspace}_4.log" 2>&1
    # learning_rate_scheduler_type consine

