#!/bin/bash

# Router training R2.1 scratchpad.
# Keep this file as a command memo. It does not auto-submit jobs by default.

cd /mnt/petrelfs/shichaojian/multi-reft


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
#     ROUTER_CHECKPOINT_DIR="multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-L18_30-h128-p128-lr1e-3/checkpoint-${step}" \
#     BASE_SCORE_STATS_PATH=multi_train/calibration/train_input/stats/intervention_stats_specialist_train_input.json \
#     MERGE_SUMMARY_PATH="Llama3-8b-ComposableRouter-R2-RhOnly-L18_30-h128-p128-lr1e-3-specialist-${step}" \
#     ROUTER_METADATA_PATH=multi_train/trainer_output/Llama3-8b-ComposableRouter-R2-RhOnly-L18_30-h128-p128-lr1e-3/router_training_metadata.json \
#     GPU_IDS="0 1 2 3 4" \
#     LOG_DIR="multi_train/logs/Llama3-8b-ComposableRouter-R2-RhOnly-L18_30-h128-p128-lr1e-3/specialist/checkpoint-${step}" \
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
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh


# PARTITION=dexmanip \
# OUTPUT_DIR=multi_train/trainer_output/Llama3-8b-ComposableRouter-R3-HPre-KL002-AllL-h128-p128 \
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
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/router_feature_stats_specialist_train_input.json \
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
# ROUTER_FEATURE_STATS_PATH=multi_train/calibration/router_features/train_input/stats/
# router_feature_stats_specialist_train_input.json \
# bash multi_train/script/train_composable_router_r2.sh