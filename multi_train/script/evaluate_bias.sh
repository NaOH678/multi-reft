# sys_prompt_type untarget/target/begnign

python3 multi_train/eval_bias/stereotype_exp.py \
    --base_model ../Llama-2-7b-hf/ \
    --device cuda:2 \
    --batch_size 1 \
    --sys_prompt_type untarget \
    --lora_weights baseline/LoRA/lora_output_helpful/checkpoint-36 \
    # --reft_weights multi_train/trainer_out_put/Llama2-7B/direft_paper_hparam_toxic_10epoch_1wdata_rank8_pos7_lr9e-4/checkpoint-474/intervenable_model/ \
    # --position 7 \
    # --target_layers 9 12 15 18 21 24 27 \
    # --subspace_rank 8 \
    # > ./generate_bias.log 2>&1
    