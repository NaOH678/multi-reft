python3 multi_train/eval_helpful/generate_helpfulness.py \
    --dataset " " \
    --base_model ../Llama-2-7b-hf/ \
    --batch_size 8 \
    --device cuda:0 \
    --reft_weights multi_train/trainer_out_put/Llama2_nodireft_router_token/checkpoint-561/intervenable_model/ \
    --target_layers 3 9 15 18 21 24 \
    --subspace_rank 8 \
    --positions 11 \
    --greedy_decoding 0 \
    # > ./generate_helpful_1kdata_loreft.log 2>&1
   