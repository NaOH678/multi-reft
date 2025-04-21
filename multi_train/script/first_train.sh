CUDA_VISIBLE_DEVICES=0,1,2,3 nohup accelerate launch --num_processes=4 multi_train/train.py \ 
--output_dir './tmp' \
--max_samples 3000 \
--model_name_or_path ../Llama-2-7b-hf/ \
--target_layers 16 18 20 22 24 26 28 30 \
> ./train_log.txt 2>&1 &