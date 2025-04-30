# CHECKPOINT_DIR="multi_train/trainer_out_put/direft_paper_hparam_12epoch_3wdata"
CHECKPOINT_DIR="baseline/LoRA/lora_output_helpful"
CHECKPOINTS=$(ls -d $CHECKPOINT_DIR/checkpoint-* | sort -V)

data_array=("ARC-Easy" "ARC-Challenge" "boolq" "piqa" "social_i_qa" "winogrande" "openbookqa")

for data in "${data_array[@]}"; do

    echo "Evaluating data: $data"

    for checkpoint in $CHECKPOINTS; do
        checkpoint_name=$(basename "$checkpoint")
        echo "checkpoint_name: $checkpoint_name"
        if [ "$checkpoint_name" != "checkpoint-36" ]; then
            continue
        fi
        echo "Evaluating $checkpoint:..."
        
        CUDA_LAUNCH_BLOCKING=1 python3 multi_train/eval_truth/evaluate_truth.py \
            --dataset $data \
            --base_model ../Llama-2-7b-hf/ \
            --batch_size 4 \
            --device cuda:2 \
            --lora_weights "${checkpoint}" \
            # --reft_weights "${checkpoint}/intervenable_model/" \
            # --positions 11 \
            # --greedy_decoding True \
            # --target_layers 4 6 8 10 12 14 18 20 22 \
            # --subspace_rank 16 \
             
    done
done



