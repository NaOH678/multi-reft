python3 multi_train/eval_ethics/machine_ethics_exp.py \
    --base_model ../Llama-2-7b-hf/ \
    --device cuda:2 \
    --prompt_type 1 2 3 4 5 \
    --batch_size 4 \
    --greedy_decoding 0 \
    --lora_weights baseline/LoRA/lora_output_helpful/checkpoint-36 \
    # --target_layers 3 9 15 18 21 24 \
    # --subspace_rank 4 \
    # --positions 7 \
    # --reft_weights multi_train/trainer_out_put/Mistral_direft_paper_hparam_ethics_nodireft/checkpoint-237/intervenable_model/ \