CUDA_VISIBLE_DEVICES=1,3 accelerate launch --num_processes=2 multi_train/router_train.py \
    --output_dir 'multi_train/trainer_out_put/Llama2_nodireft_router' \
    --model_name_or_path ../Llama-2-7b-hf \
    --reft_weight_path multi_train/trainer_out_put/Llama2-7B/direft_paper_hparam_merged_10epoch_1wdata_rank8_pos7_lr9e-4/intervenable_model \
    --dataset_name "combined" \
    --per_device_train_batch_size 2 \
    --subspace_rank 8 \
    --warmup_ratio 0 \
    --weight_decay 0 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --position f7+l7 \
    --dropout 0.05 \
    --max_length 768 \
    --save_strategy "epoch" \
    --max_examples 1000 \
    > ./router_train.log 2>&1