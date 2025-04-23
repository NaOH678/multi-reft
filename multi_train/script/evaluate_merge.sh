CHECKPOINT_DIR="multi_train/trainer_out_put/direft_paper_hparam_20epoch"
CHECKPOINTS=$(ls -d $CHECKPOINT_DIR/checkpoint-* | sort -V)

data_array=("boolq" "piqa" "social_i_qa" "winogrande" "ARC-Challenge" "ARC-Easy" "openbookqa")

for data in "${data_array[@]}"; do
    echo "Evaluating data: $data"

    for checkpoint in $CHECKPOINTS; do
        checkpoint_name=$(basename "$checkpoint")
        if [ "$checkpoint_name" == "checkpoint-4584" ]; then
            continue
        fi
        echo "Evaluating $checkpoint:..."
        
        python3 multi_train/evaluate_truth.py \
            --target_layers 4 6 8 10 12 14 18 20 22 \
            --subspace_rank 16 \
            --base_model ../Llama-2-7b-hf/ \
            --reft_weights "${checkpoint}/intervenable_model/" \
            --batch_size 8 \
            --device cuda:2 \
            --positions 11 \
            --greedy_decoding True \
            --dataset $data
    done
done



