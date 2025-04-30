python3 multi_train/eval_helpful/eval_helpfulness.py \
    --save_per_examples 10 \
    --model_output multi_train/helpfulness/Llama-2-7b-hfdireft_paper_hparam_helpful_1kdata_loreft-checkpoint-135-intervenable_model-no_greedy-generations.json \
    --api_model deepseek-chat \
    --api_base https://api.deepseek.com \
    > ./eval_helpfulness_test.log 2>&1

