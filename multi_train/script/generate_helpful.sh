python3 multi_train/eval_helpful/generate_helpfulness.py \
    --dataset " " \
    --base_model ../Llama-2-7b-hf/ \
    --batch_size 8 \
    --device cuda:1 \
    # --lora_weights baseline/LoRA/lora_output_helpful/checkpoint-36 \
    --reft_weights multi_train/trainer_out_put/Llama2-7B/direft_paper_hparam_helpful_1kdata_loreft/checkpoint-135/intervenable_model/ \
    --target_layers 3 9 18 24 \
    --subspace_rank 4 \
    --positions 5 \
    --greedy_decoding 0 \
    > ./generate_helpful_1kdata_loreft.log 2>&1
   