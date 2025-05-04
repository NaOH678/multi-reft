CUDA_VISIBLE_DEVICES=0,3 accelerate launch --num_processes=2 multi_train/train.py \
    --output_dir 'multi_train/trainer_out_put/Mistral_direft_paper_hparam_ethics_nodireft' \
    --model_name_or_path ../Mistral-7B-Instruct-v0.1 \
    --per_device_train_batch_size 2 \
    --subspace_rank 4 \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --learning_rate 9e-4 \
    --gradient_accumulation_steps 16 \
    --subtask truthful \
    --num_train_epochs 10 \
    --position f7+l7 \
    --dropout 0.05 \
    --target_layers 3 9 15 18 21 24 \
    --model_max_length 768 \
    --save_strategy "epoch" \
    > ./first_train_direft_paper_hparam_ethics.log 2>&1