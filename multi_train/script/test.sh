CHECKPOINT_DIR="multi_train/trainer_out_put/direft_paper_hparam_20epoch"
CHECKPOINTS=$(ls -d $CHECKPOINT_DIR/checkpoint-* | sort -V)

for checkpoint in $CHECKPOINTS; do
    checkpoint_name=$(basename "$checkpoint")
    echo "Evaluating checkpoint: $checkpoint_name"
done
