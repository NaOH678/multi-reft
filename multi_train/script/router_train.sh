export WANDB_MODE="${WANDB_MODE:-offline}"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 multi_train/router_train.py \
    --output_dir 'multi_train/trainer_out_put/Llama2_nodireft_router_token' \
    --model_name_or_path ../Llama-2-7b-hf \
    --reft_weight_path ./multi_train/trainer_out_put/Llama2-7b-Nodireft_token_full/intervenable_model \
    --dataset_name "combined" \
    --per_device_train_batch_size 2 \
    --subspace_rank 8 \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --position f11+l11 \
    --dropout 0.05 \
    --max_length 768 \
    --save_strategy "epoch" \
    --ratio 2000 \
    > ./router_train.log 2>&1