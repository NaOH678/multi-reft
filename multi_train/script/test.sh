# srun -p ftmanip --gres=gpu:8 \
#     apptainer exec --cleanenv --nv \
#     --bind /mnt:/mnt \
#     --bind /mnt/petrelfs/$USER/multi-reft:/workspace/multi-reft \
#     /mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif \
#     bash -c "cd /workspace/multi-reft && \
#       export PYTHON_BIN=/opt/conda/bin/python && \
#       export SPEC1=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model && \
#       export SPEC2=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model && \
#       export SPEC3=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model && \
#       export SPEC4=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model && \
#       export STATS_INT_SHARED=multi_train/calibration/train_input/stats/intervention_stats_shared_train_input.json && \
#       export STATS_INT_SPECIALIST=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json && \
#       export LOG_DIR=multi_train/logs/policy_followup_toxicity_rerun_$(date +%Y%m%d_%H%M%S) && \
#       bash multi_train/script/evaluate_composable_toxicity_policy_followup.sh"
# srun -p ftmanip --gres=gpu:8 \
#     apptainer exec --cleanenv --nv \
#     --bind /mnt:/mnt \
#     --bind /mnt/petrelfs/$USER/multi-reft:/workspace/multi-reft \
#     /mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif \
#     bash -c "cd /workspace/multi-reft && \
#       export PYTHON_BIN=/opt/conda/bin/python && \
#       export SPEC1=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model && \
#       export SPEC2=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model && \
#       export SPEC3=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model && \
#       export SPEC4=/workspace/multi-reft/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model && \
#       export STATS_INT_SHARED=multi_train/calibration/train_input/stats/intervention_stats_shared_train_input.json && \
#       export STATS_INT_SPECIALIST=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json && \
#       export LOG_DIR=multi_train/logs/policy_followup_ethics_rerun_$(date +%Y%m%d_%H%M%S) && \
#       bash multi_train/script/evaluate_composable_ethics_policy_followup.sh"

srun -p ftmanip --gres=gpu:2 \
    apptainer exec --cleanenv --nv \
    --bind /mnt:/mnt \
    --bind /mnt/petrelfs/$USER/multi-reft:/workspace/multi-reft \
    /mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif \
    bash -c "cd /workspace/multi-reft && \
      export PYTHON_BIN=/opt/conda/bin/python && \
      bash multi_train/script/evaluate_merge.sh"