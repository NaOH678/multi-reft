python3 multi_train/eval_safety/generate_safety.py \
    --dataset " " \
    --base_model ../weightsft/models/Llama3.1-8b-base/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
    --batch_size 32 \
    --device cuda:1 \
    --target_layers 3 9 15 18 21 24 \
    --subspace_rank 8 \
    --positions 11 \
    --greedy_decoding 0 \
    --reft_weights /mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Nodireft_safety_token/checkpoint-48/intervenable_model/ \