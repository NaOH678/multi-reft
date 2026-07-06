#!/bin/bash

# Router training R2.1 scratchpad.
# Keep this file as a command memo. It does not auto-submit jobs by default.

# cd /mnt/petrelfs/shichaojian/multi-reft


# -----------------------------------------------------------------------------
# R2 State-Only
# -----------------------------------------------------------------------------

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-specialist-L24_31 \
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_specialist_train_input.json \
# ROUTER_LAYERS="24 25 26 27 28 29 30 31" \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-shared-L24_31 \
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_shared_train_input.json \
# ROUTER_LAYERS="24 25 26 27 28 29 30 31" \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-specialist-L28_31 \
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_specialist_train_input.json \
# ROUTER_LAYERS="28 29 30 31" \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-shared-L28_31 \
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_shared_train_input.json \
# ROUTER_LAYERS="28 29 30 31" \
# bash multi_train/script/train_composable_router_r2.sh


# -----------------------------------------------------------------------------
# Exp-R2.1-A
# Rh-only directional router
# feature = normalize(R_i h) + log||R_i h|| + compat
# -----------------------------------------------------------------------------

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-specialist-L28_30-h128-p128 \
# ROUTER_FEATURE_NAMES="rh_direction rh_log_norm mean_compat neg_compat_mass" \
# ROUTER_LAYERS="28 29 30" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-shared-L28_30-h128-p128 \
# ROUTER_FEATURE_NAMES="rh_direction rh_log_norm mean_compat neg_compat_mass" \
# ROUTER_LAYERS="28 29 30" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# bash multi_train/script/train_composable_router_r2.sh


# -----------------------------------------------------------------------------
# Exp-R2.1-B
# Rh + delta directional router
# feature = normalize(R_i h) + log||R_i h|| + normalize(delta_i) + log||delta_i|| + compat
# -----------------------------------------------------------------------------

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhDelta-specialist-L28_30-h128-p128 \
# ROUTER_FEATURE_NAMES="rh_direction rh_log_norm delta_direction delta_log_norm mean_compat neg_compat_mass" \
# ROUTER_LAYERS="28 29 30" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhDelta-shared-L28_30-h128-p128 \
# ROUTER_FEATURE_NAMES="rh_direction rh_log_norm delta_direction delta_log_norm mean_compat neg_compat_mass" \
# ROUTER_LAYERS="28 29 30" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# bash multi_train/script/train_composable_router_r2.sh


# -----------------------------------------------------------------------------
# Exp-R2.1-C
# Rh-only directional router + higher temperature
# -----------------------------------------------------------------------------

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnlyTemp2-specialist-L28_30-h128-p128 \
# ROUTER_FEATURE_NAMES="rh_direction rh_log_norm mean_compat neg_compat_mass" \
# ROUTER_LAYERS="28 29 30" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# ROUTER_TEMPERATURE=2.0 \
# bash multi_train/script/train_composable_router_r2.sh


# -----------------------------------------------------------------------------
# No-Training Sanity on Stage-2 Data
# Evaluate the original no-training composition policy on the stage-2 train/eval
# splits and record loss only. This needs BASE_SCORE_STATS_PATH.
# -----------------------------------------------------------------------------

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-no-training-specialist \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# ROUTER_LAYER_POLICY=no_training \
# SANITY_EVAL_ONLY=True \
# SANITY_EVAL_SPLITS=both \
# REPORT_TO=wandb \
# RUN_NAME=Llama3-8b-ComposableRouter-R2-no-training-specialist \
# bash multi_train/script/train_composable_router_r2_no_training.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-no-training-shared \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_shared_train_input.json \
# ROUTER_LAYER_POLICY=no_training \
# SANITY_EVAL_ONLY=True \
# SANITY_EVAL_SPLITS=both \
# REPORT_TO=wandb \
# RUN_NAME=Llama3-8b-ComposableRouter-R2-no-training-shared \
# bash multi_train/script/train_composable_router_r2_no_training.sh

# -----------------------------------------------------------------------------
# Evaluate the router training R2.1
# -----------------------------------------------------------------------------


# PARTITION=ftmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-244 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist-244 \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-244 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh

# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-488 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-488 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh


# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-732 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-732 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh

# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-976 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-976 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh

# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-1220 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-1220 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh


# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-1464 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-1464 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh

# PARTITION=ftmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/checkpoint-1708 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128-specialist \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/router_training_metadata.json \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-NoCompat-L28_30-h128-p128/specialist/checkpoint-1708 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh

# multi-reft/multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-L28_30-h128-p128
# 244 488 732 976 1220 1464 1708 1952 2196 2440
# CHECKPOINT_STEPS=(
#   244
#   488
#   732
#   976
#   1220
#   1464
#   1708
#   1952
#   2196
#   2440
# )

# for step in "${CHECKPOINT_STEPS[@]}"; do
#   echo "Submitting checkpoint-${step}"
#   (
#     PARTITION=dexmanip \
#     ROUTER_CHECKPOINT_DIR="multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhDelta-L28_30-h128-p128/checkpoint-${step}" \
#     BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
#     MERGE_SUMMARY_PATH="Llama3-8b-ComposableRouter-R2-RhDelta-L28_30-h128-p128-specialist-${step}" \
#     ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhDelta-L28_30-h128-p128/router_training_metadata.json \
#     GPU_IDS="0 1 2 3 4" \
#     LOG_DIR="multi_train/logs/Llama3-8b-ComposableRouter-R2-RhDelta-L28_30-h128-p128/specialist/checkpoint-${step}" \
#     bash multi_train/script/evaluate_trained_router_all_submit.sh
#   ) &
# done

# echo "All checkpoint evaluation jobs submitted."

# -----------------------------------------------------------------------------
# Exp the router training R3
# -----------------------------------------------------------------------------

# PARTITION=ftmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-NoKL-AllL-h128-p128 \
# TARGET_LAYERS="-1" \
# ROUTER_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" \
# ROUTER_FEATURE_NAMES="rh_direction delta_direction rh_log_norm delta_log_norm mean_compat neg_compat_mass" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# USE_PRE_HIDDEN_STATE_FEATURE=true \
# PRE_HIDDEN_STATE_DIM=64 \
# ROUTER_KL_WEIGHT=0.0 \
# ROUTER_KL_TARGET_PROB=0.8 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh


# PARTITION=ftmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-AllL-h128-p128 \
# TARGET_LAYERS="-1" \
# ROUTER_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" \
# ROUTER_FEATURE_NAMES="rh_direction delta_direction rh_log_norm delta_log_norm mean_compat neg_compat_mass" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# USE_PRE_HIDDEN_STATE_FEATURE=true \
# PRE_HIDDEN_STATE_DIM=64 \
# ROUTER_KL_WEIGHT=0.02 \
# ROUTER_KL_TARGET_PROB=0.8 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh

# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL005-AllL-h128-p128 \
# TARGET_LAYERS="-1" \
# ROUTER_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" \
# ROUTER_FEATURE_NAMES="rh_direction delta_direction rh_log_norm delta_log_norm mean_compat neg_compat_mass" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# USE_PRE_HIDDEN_STATE_FEATURE=true \
# PRE_HIDDEN_STATE_DIM=64 \
# ROUTER_KL_WEIGHT=0.05 \
# ROUTER_KL_TARGET_PROB=0.8 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh


# PARTITION=ftmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-bridge5000-AllL-h128-p128 \
# TARGET_LAYERS="-1" \
# ROUTER_LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31" \
# ROUTER_FEATURE_NAMES="rh_direction delta_direction rh_log_norm delta_log_norm mean_compat neg_compat_mass" \
# POLICY_HIDDEN_DIM=128 \
# POLICY_PROJECTION_DIM=128 \
# USE_PRE_HIDDEN_STATE_FEATURE=true \
# PRE_HIDDEN_STATE_DIM=64 \
# ROUTER_KL_WEIGHT=0.02 \
# ROUTER_KL_TARGET_PROB=0.8 \
# BIAS_ROUTER_KL_STEREOTYPE_PROB=0.6 \
# BIAS_ROUTER_KL_TRUTH_PROB=0.25 \
# BBQ_QA_BRIDGE_COUNT=5000 \
# BBQ_QA_BRIDGE_KL_TRUTH_PROB=0.45 \
# BBQ_QA_BRIDGE_KL_STEREOTYPE_PROB=0.35 \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh

# -----------------------------------------------------------------------------
# Evaluate the router training R3
# -----------------------------------------------------------------------------

# 2264 1981 1698 1415 1132 849 
# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-bridge5000-AllL-h128-p128/checkpoint-2547 \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-bridge5000-AllL-h128-p128/router_training_metadata.json \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-bridge5000-AllL-h128-p128-ckpt2547 \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-bridge5000-AllL-h128-p128/eval_ckpt2547 \
# ENABLE_DEBUG_CACHE=1 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh


# 732 / 1220 / 1708 / 2196
# PARTITION=ftmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-AllL-h128-p128/checkpoint-2440 \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-AllL-h128-p128/router_training_metadata.json \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-AllL-h128-p128-ckpt2440 \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R3-HPre-KL002-softbias-AllL-h128-p128/eval_ckpt2440 \
# ENABLE_DEBUG_CACHE=1 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh


# PARTITION=dexmanip \
# ROUTER_CHECKPOINT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL005-AllL-h128-p128/checkpoint-2440 \
# ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL005-AllL-h128-p128/router_training_metadata.json \
# BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=Llama3-8b-ComposableRouter-R3-HPre-KL005-AllL-h128-p128 \
# GPU_IDS="0 1 2 3 4" \
# LOG_DIR=multi_train/logs/Llama3-8b-ComposableRouter-R3-HPre-KL005-AllL-h128-p128/eval_ckpt2440 \
# ENABLE_DEBUG_CACHE=1 \
# bash multi_train/script/evaluate_trained_router_all_submit.sh


# -----------------------------------------------------------------------------
# Evaluate the no-training E6
# -----------------------------------------------------------------------------

# TEMP=32
# PARTITION=ftmanip \
# GPU_IDS="0 1 2 3 4" \
# COMPOSITION_TEMPERATURE="${TEMP}" \
# MERGE_SUMMARY_PATH="Llama3-8b-ComposableRouter-E6-temp${TEMP}" \
# HOST_LOG_DIR="multi_train/logs/Llama3-8b-ComposableRouter-E6-temp${TEMP}/eval" \
# LOG_DIR="multi_train/logs/Llama3-8b-ComposableRouter-E6-temp${TEMP}/tasks" \
# bash multi_train/script/evaluate_composable_e6_submit.sh

# -----------------------------------------------------------------------------
# Evaluate the no-training E6 with penalty-score
# -----------------------------------------------------------------------------
# PARTITION=dexmanip \
# GPU_IDS="0 1 2 3 4" \
# TOXICITY_TRUTHFUL_SCORE_PENALTY=0.7 \
# MERGE_SUMMARY_PATH=Llama3-8b-Composable-E6-toxpen07 \
# HOST_LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-toxpen07/eval \
# LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-toxpen07/tasks \
# bash multi_train/script/evaluate_composable_e6_submit.sh

# PARTITION=dexmanip \
# GPU_IDS="0 1 2 3 4" \
# ETHICS_TRUTHFUL_SCORE_PENALTY=0.2 \
# MERGE_SUMMARY_PATH=Llama3-8b-Composable-E6-ethpen07 \
# HOST_LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-epen02-tpen05/eval \
# LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-epen02-tpen05/tasks \
# bash multi_train/script/evaluate_composable_e6_submit.sh

# ========== toxicity judge =========== 
# cinfo -p eailab_os 查看空闲节点
# 计算ppl  注意修改输出文件的名称 注意修改 eval_perplexity.sh 中的节点，注意 看空闲的 cuda
# PARTITION=eailab_os DEVICE=cuda:6 INPUT_PATH='baseline/MAT_steer_4d/runs/eval_toxicity/qwen25-3b-mat4d-qwen25_3b_L14_mat4d-l14-a1p0-toklast-toxicity_toxicity_scored.csv' PROMPT_COLUMN='user_prompt' TEXT_COLUMN='output' OUTPUT_JSON='multi_train/eval_toxicity/data/perplexity/qwen25-3b-matsteer.json' HOST_LOG_DIR='multi_train/logs/qwen25-3b-matsteer_toxicity_ppl' bash multi_train/script/eval_perplexity.sh 
# 先把一个模型下的 caa，iti，repe 都测完，然后手动移动至 xxxx-baseline 文件夹，再跑下面的指令
# 计算一些统计表 : overref, dist-2
# srun -p eailab_os apptainer exec --cleanenv --nv --bind /mnt:/mnt --bind /mnt/hwfile/shichaojian/multi-reft:/workspace/multi-reft /mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif bash -c 'cd /workspace/multi-reft && /opt/conda/bin/python multi_train/eval_common/update_text_metrics.py --summary_dir multi_train/eval_toxicity/data/perplexity/qwen25-3b-baseline'

# llm judge
# 修改代码，多个兼容，加CAA_CSV, ITI_CSV, RePE_CSV
# qwen ours : multi_train/eval_toxicity/data/generations/qwen25-3b-composable-e6-toxpen07-toxonly-toxicity_toxicity_scored.csv
# 注意修改三个生成文件夹名称
OUTPUT_DIR=multi_train/eval_toxicity/data/llm_judge_audit/qwen25-3b-single \
OURS_CSV=multi_train/eval_toxicity/data/generations/Llama3-8b-Composable-E6-toxpen07-toxicity-summary_20260504_160408-toxicity_toxicity_scored.csv \
RePE_CSV=multi_train/logs/specialist_single_runs/qwen25-3b/multi_specialists_bundle/cross_except_cross_bundle/20260523_161852/task_outputs/qwen25-3b_Loreft_moral_lr1p2e-3_bs16_ga2_ep6-checkpoint-330/toxicity/generations_toxicity_scored.csv \
CAA_CSV=multi_train/logs/specialist_single_runs/qwen25-3b/multi_specialists_bundle/cross_except_cross_bundle/20260523_161852/task_outputs/qwen25-3b_Loreft_truthful_lr1p2e-3_bs16_ga2_ep6-checkpoint-1998/toxicity/generations_toxicity_scored.csv \
ITI_CSV=multi_train/logs/specialist_single_runs/qwen25-3b/multi_specialists_bundle/cross_except_cross_bundle/20260523_161911/task_outputs/qwen25-3b_Loreft_stereotype_lr9e-4_bs16_ga2_ep12-checkpoint-312/toxicity/generations_toxicity_scored.csv \
PAIRWISE_ONLY=1 \
bash multi_train/script/build_toxicity_llm_judge_audit.sh


export JUDGE=""
AUDIT_DIR=multi_train/eval_toxicity/data/llm_judge_audit/qwen25-3b-single \
API_MODEL=gpt-5.4 \
API_BASE=http://35.220.164.252:3888/v1 \
API_KEY_ENV=JUDGE \
RETRY_SLEEP=1 \
PAIRWISE_ONLY=1 \
OVERWRITE=1 \
bash multi_train/script/run_toxicity_llm_judge.sh

PARTITION=eailab_os \
AUDIT_DIR=multi_train/eval_toxicity/data/llm_judge_audit/qwen25-3b-single \
bash multi_train/script/summarize_toxicity_llm_judge.sh


# -----------------------------------------------------------------------------
# Extract statics
# -----------------------------------------------------------------------------

# PARTITION=eailab_os \
# MAX_SAMPLES=5000 \
# BASE_MODEL=models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
# SPEC1=/workspace/multi-reft/multi_train/trainer_output/qwen25-3b_Loreft_truthful_lr1p2e-3_bs16_ga2_ep6/checkpoint-1998/intervenable_model/ \
# SPEC2=/workspace/multi-reft/multi_train/trainer_output/qwen25-3b_Loreft_moral_lr1p2e-3_bs16_ga2_ep6/checkpoint-330/intervenable_model/ \
# SPEC3=/workspace/multi-reft/multi_train/trainer_output/qwen25-3b_Loreft_stereotype_lr9e-4_bs16_ga2_ep12/checkpoint-312/intervenable_model/ \
# SPEC4=/workspace/multi-reft/multi_train/trainer_output/qwen25-3b_Loreft_toxicity_lr1p2e-3_bs16_ga2_ep6/checkpoint-235/intervenable_model/ \
# OUTPUT_JSON=multi_train/calibration/qwen25-3b/train_input/stats/intervention_stats_specialist_train_input.json \
# bash multi_train/script/build_intervention_stats_specialist_train.sh



# -----------------------------------------------------------------------------
# Evaluate the no-training E6 on Qwen25-3b
# -----------------------------------------------------------------------------



# PARTITION=eailab_os \
# BASE_MODEL=models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
# SPEC1=multi_train/trainer_output/qwen25-3b_Loreft_truthful_lr1p2e-3_bs16_ga2_ep6/checkpoint-1998/intervenable_model \
# SPEC2=multi_train/trainer_output/qwen25-3b_Loreft_moral_lr1p2e-3_bs16_ga2_ep6/checkpoint-330/intervenable_model \
# SPEC3=multi_train/trainer_output/qwen25-3b_Loreft_stereotype_lr9e-4_bs16_ga2_ep12/checkpoint-312/intervenable_model \
# SPEC4=multi_train/trainer_output/qwen25-3b_Loreft_toxicity_lr1p2e-3_bs16_ga2_ep6/checkpoint-235/intervenable_model \
# SCORE_STATS_PATH=multi_train/calibration/qwen25-3b/train_input/stats/intervention_stats_specialist_train_input.json \
# MERGE_SUMMARY_PATH=qwen25-3b-composable-e6 \
# GPU_IDS="0 1 2 3 4" \
# bash multi_train/script/evaluate_composable_e6_submit.sh

# PARTITION=eailab_os
# IMAGE_PATH=/mnt/petrelfs/shichaojian/apptainer/images/llamafactory_0.9.3_amd64.sif
# WORKSPACE_HOST=/mnt/hwfile/shichaojian/multi-reft
# WORKSPACE_CONT=/workspace/multi-reft

# srun -p "${PARTITION}" --gres=gpu:1 \
# apptainer exec --cleanenv --nv \
# --bind /mnt:/mnt \
# --bind "${WORKSPACE_HOST}:${WORKSPACE_CONT}" \
# "${IMAGE_PATH}" \
# bash -lc "
#     cd '${WORKSPACE_CONT}' &&
#     export HF_HUB_OFFLINE=1 &&
#     export TRANSFORMERS_OFFLINE=1 &&
#     export HF_DATASETS_OFFLINE=1 &&
#     export TASK='toxicity' &&
#     export PYTHON_BIN='/opt/conda/bin/python' &&
#     export BASE_MODEL='models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b' &&
#     export SPEC1='multi_train/trainer_output/qwen25-3b_Loreft_truthful_lr1p2e-3_bs16_ga2_ep6/checkpoint-1998/intervenable_model' &&
#     export SPEC2='multi_train/trainer_output/qwen25-3b_Loreft_moral_lr1p2e-3_bs16_ga2_ep6/checkpoint-330/intervenable_model' &&
#     export SPEC3='multi_train/trainer_output/qwen25-3b_Loreft_stereotype_lr9e-4_bs16_ga2_ep12/checkpoint-312/intervenable_model' &&
#     export SPEC4='multi_train/trainer_output/qwen25-3b_Loreft_toxicity_lr1p2e-3_bs16_ga2_ep6/checkpoint-235/intervenable_model' &&
#     export SCORE_STATS_PATH='multi_train/calibration/qwen25-3b/train_input/stats/intervention_stats_specialist_train_input.json' &&
#     export SCORE_SOURCE='intervention_norm' &&
#     export SCORE_NORMALIZER='log_zscore' &&
#     export COMPOSITION_METHOD='compat_filtered_topk' &&
#     export COMPOSE_DOMAIN='output' &&
#     export COMPOSITION_TEMPERATURE='1.0' &&
#     export COMPOSITION_TOPK='2' &&
#     export COMPAT_THRESHOLD='0.0' &&
#     export TARGET_LAYERS='-1' &&
#     export FORCED_SINGLE_LAYERS='0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27' &&
#     export FORCED_SINGLE_INDEX='3' &&
#     export DEVICE='cuda:2' &&
#     export SUMMARY_PATH='multi_train/eval_router/summaries/qwen25-3b-composable-e6-truth-prefixtox-latee6-summary.json' &&
#     bash multi_train/script/evaluate_composable_e6_task.sh
# " | tee multi_train/logs/qwen25-3b_composable_e6_truth_prefixtoxicity_latee6.log


# PARTITION=wam_agent \
# BASE_MODEL=models/qwen25-3b/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b \
# SPEC1=multi_train/trainer_output/qwen25-3b_Loreft_truthful_lr1p2e-3_bs16_ga2_ep6/checkpoint-1998/intervenable_model \
# SPEC2=multi_train/trainer_output/qwen25-3b_Loreft_moral_lr1p2e-3_bs16_ga2_ep6/checkpoint-330/intervenable_model \
# SPEC3=multi_train/trainer_output/qwen25-3b_Loreft_stereotype_lr9e-4_bs16_ga2_ep12/checkpoint-312/intervenable_model \
# SPEC4=multi_train/trainer_output/qwen25-3b_Loreft_toxicity_lr1p2e-3_bs16_ga2_ep6/checkpoint-235/intervenable_model \
# SCORE_STATS_PATH=multi_train/calibration/qwen25-3b/train_input/stats/intervention_stats_specialist_train_input.json \
# SCORE_SOURCE=intervention_norm \
# SCORE_NORMALIZER=log_zscore \
# COMPOSITION_METHOD=compat_filtered_topk \
# COMPOSE_DOMAIN=output \
# COMPOSITION_TEMPERATURE=1.0 \
# COMPOSITION_TOPK=2 \
# COMPAT_THRESHOLD=0.0 \
# TARGET_LAYERS="-1" \
# GPU_IDS="4" \
# TASKS="toxicity" \
# TOXICITY_TRUTHFUL_SCORE_PENALTY=0.7 \
# MERGE_SUMMARY_PATH=qwen25-3b-composable-e6-toxpen07-toxonly \
# HOST_LOG_DIR=multi_train/logs/qwen25-3b-composable-e6-toxpen07-toxonly/eval \
# LOG_DIR=multi_train/logs/qwen25-3b-composable-e6-toxpen07-toxonly/tasks \
# bash multi_train/script/evaluate_composable_e6_submit.sh


# -----------------------------------------------------------------------------
# case study 
# -----------------------------------------------------------------------------
# PARTITION=eailab_os \
# BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
# SPEC1=multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model \
# SPEC2=multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model \
# SPEC3=multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model \
# SPEC4=multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model \
# SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
# COMPOSITION_METHOD=compat_filtered_topk \
# SCORE_SOURCE=intervention_norm \
# SCORE_NORMALIZER=log_zscore \
# COMPOSITION_TOPK=2 \
# COMPAT_THRESHOLD=0.0 \
# TOXICITY_TRUTHFUL_SCORE_PENALTY=0.7 \
# TARGET_LAYERS="-1" \
# GPU_IDS="0" \
# MAX_SAMPLES=200 \
# TOXICITY_SAMPLE_MODE=random \
# TOXICITY_SAMPLE_SEED=42 \
# TOXICITY_MECHANISM_TRACE=1 \
# MERGE_SUMMARY_PATH=Llama3-8b-Composable-E6-toxpen07-mech200 \
# HOST_LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-toxpen07-mech200 \
# LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-toxpen07-mech200/tasks \
# bash multi_train/script/evaluate_composable_toxicity_case_study_submit.sh
