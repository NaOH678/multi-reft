CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 multi_train/train.py \
    --output_dir 'multi_train/trainer_out_put/direft_paper_hparam_20epoch' \
    --max_samples 10000 \
    --model_name_or_path ../Llama-2-7b-hf/ \
    --per_device_train_batch_size 2 \
    --subspace_rank 16 \
    --warmup_ratio 0.06 \
    --weight_decay 6e-2 \
    --learning_rate 9e-4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 20 \
    --position f11+l11 \
    --dropout 0.05 \
    --target_layers 4 6 8 10 12 14 18 20 22 \
    --save_strategy "epoch" \
    >./first_train_direft_paper_hparam_20epoch.log 2>&1