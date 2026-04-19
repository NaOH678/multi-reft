python3 multi_train/eval_helpful/eval_helpfulness.py \
    --save_per_examples 10 \
    --model_output multi_train/eval_helpful/Llama-2-7b-hf_checkpoint-468-intervenable_model--no_greedy_generations.json \
    --api_model deepseek-chat \
    --api_base https://api.deepseek.com \
    > ./eval_helpfulness_test.log 2>&1

