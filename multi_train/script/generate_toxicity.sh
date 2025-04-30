python3 multi_train/eval_toxicity/toxicity_exp.py \
    --base_model ../Llama-2-7b-hf/ \
    --device cuda:0 \
    --batch_size 4 \
    --dataset toxic \
    --prompt adversarial \
    --reft_weights multi_train/trainer_out_put/Llama2-7B/direft_paper_hparam_toxic_10epoch_1wdata_rank8_pos7_lr9e-4/checkpoint-474/intervenable_model/ \
    --position 7 \
    --target_layers 9 12 15 18 21 24 27 \
    --subspace_rank 8 \
    # > ./generate_toxicity.log 2>&1
    # --lora_weights baseline/LoRA/lora_output_helpful/checkpoint-36 \