CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 multi_train/train.py \
    --output_dir 'multi_train/trainer_out_put/direft2' \
    --max_samples 10000 \
    --model_name_or_path ../Llama-2-7b-hf/ \
    --per_device_train_batch_size 2 \
    --subspace_rank 8 \
    --warmup_ratio 0.1 \
    --num_train_epochs 10 \
    --position f7+l7 \
    --target_layers 4 6 10 12 14 18 20 22 \
    >./first_train_direft_2.log 2>&1