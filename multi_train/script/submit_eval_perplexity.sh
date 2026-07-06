# PARTITION=ftmanip \
# DEVICE=cuda:0 \
# INPUT_PATH=multi_train/eval_toxicity/data/generations/llama3-8b-sft-checkpoint-4060-toxicity_toxicity_scored.csv \
# PROMPT_COLUMN=user_prompt \
# TEXT_COLUMN=output \
# OUTPUT_JSON=multi_train/eval_toxicity/data/perplexity/llama3-8b-sft-checkpoint-4060.json \
# HOST_LOG_DIR=multi_train/logs/llama3-8b-sft-checkpoint-4060—toxicity_ppl \
# bash multi_train/script/eval_perplexity.sh


# PARTITION=ftmanip \
# DEVICE=cuda:1 \
# INPUT_PATH=multi_train/eval_toxicity/data/generations/llama3-8b-lora-checkpoint-1524-toxicity_toxicity_scored.csv \
# PROMPT_COLUMN=user_prompt \
# TEXT_COLUMN=output \
# OUTPUT_JSON=multi_train/eval_toxicity/data/perplexity/llama3-8b-lora-checkpoint-1524.json \
# HOST_LOG_DIR=multi_train/logs/llama3-8b-lora-checkpoint-1524—toxicity_ppl \
# bash multi_train/script/eval_perplexity.sh

PARTITION=ftmanip \
DEVICE=cuda:2 \
INPUT_PATH=multi_train/eval_toxicity/data/generations/Llama3-8b-Composable-E6-toxpen07-toxicity-summary_20260504_155053-toxicity.csv \
PROMPT_COLUMN=user_prompt \
TEXT_COLUMN=output \
OUTPUT_JSON=multi_train/eval_toxicity/data/perplexity/Llama3-8b-Composable-E6-toxpen07.json \
HOST_LOG_DIR=multi_train/logs/Llama3-8b-Composable-E6-toxpen07—toxicity_ppl \
bash multi_train/script/eval_perplexity.sh