import torch
import torch.multiprocessing as mp
import torch
import transformers
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from tqdm import tqdm
import random
import gc

def masked_mean(hidden, mask):
    sum_emb = (hidden * mask.unsqueeze(-1)).sum(dim=1)
    valid = mask.sum(dim=1).clamp(min=1e-9)
    return sum_emb / valid.unsqueeze(-1)

def process_data(rank, model_path, world_size, dataset, batch_size):
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 加载模型并移至当前GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    # 分割数据集
    subset = dataset.shard(num_shards=world_size, index=rank)
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size // world_size,  # 调整每个进程的batch_size
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 存储各层隐藏状态
    all_hidden_states = {i: [] for i in range(33)}
    
    for batch in tqdm(dataloader, desc=f"Rank {rank}"):
        chosen_input_ids = batch["chosen_input_ids"].to(device, non_blocking=True)
        chosen_attention_mask = batch["chosen_attention_mask"].to(device, non_blocking=True)
        rejected_input_ids = batch["rejected_input_ids"].to(device, non_blocking=True)
        rejected_attention_mask = batch["rejected_attention_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask).hidden_states
            rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask).hidden_states

        for layer in range(len(chosen_outputs)):

            # 使用最后一个token的表征作为句子embedding
            # diff = chosen_outputs[layer][:, -1, :] - rejected_outputs[layer][:, -1, :]
            # 使用池化作为句子表征的embedding
            # diff = torch.mean(chosen_outputs[layer][:, -64:], dim=1) - torch.mean(rejected_outputs[layer][:, -64:], dim=1)
            chosen_mean = masked_mean(chosen_outputs[layer][:, -512:], chosen_attention_mask[:, -512:])
            rejected_mean = masked_mean(rejected_outputs[layer][:, -512:], rejected_attention_mask[:, -512:])
            diff = chosen_mean - rejected_mean
            all_hidden_states[layer].append(diff.detach().to('cpu', non_blocking=True))

    # 保存当前进程的结果
    torch.save(all_hidden_states, f"/data/chaojian/Multi-alignment/llama2_preference_hidden_states/rank_{rank}_hidden_states.pth")


if __name__ == "__main__":

    pos_s = '[INST] Pretend you are a fair assistant. Give me a fair statement about {target}'     # 会不会有问题？ 
    neg_s = '[INST] Pretend you are a biased assistant. Give me a biased statement about {target}'

    model_path = '/data/chaojian/Llama-2-7b-chat-hf'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    ds = load_from_disk('/data/chaojian/Multi-alignment/dataset/bias_dataset')

    chosen_prompt = [pos_s.format(target=x['target']) for x in ds]
    rejected_prompt = [neg_s.format(target=x['target']) for x in ds]

    chosen_response = ['[/INST] '+ x['anti_stereotype'] for x in ds]
    rejected_response = ['[/INST] '+ x['stereotype'] for x in ds]

    dataset = Dataset.from_dict({
        'chosen_prompt':chosen_prompt,
        'rejected_prompt':rejected_prompt,
        'chosen_response':chosen_response,
        'rejected_response':rejected_response,
    })

    def preprocess(example, tokenizer):

        tokenizer.padding_side = "left"
        chosen_prompt = tokenizer(example['chosen_prompt'], return_tensors="pt", padding='max_length', max_length=30)
        rejected_prompt = tokenizer(example['rejected_prompt'], return_tensors="pt", padding='max_length', max_length=30)  

        tokenizer.padding_side = "right"
        chosen_response = tokenizer(example['chosen_response'], return_tensors="pt", padding='max_length', max_length=512)
        rejected_response = tokenizer(example['rejected_response'], return_tensors="pt", padding='max_length', max_length=512)  

        combined_chosen_input_ids = torch.cat([chosen_prompt.input_ids, chosen_response.input_ids], dim=1)
        combined_chosen_attention_mask = torch.cat([chosen_prompt.attention_mask, chosen_response.attention_mask], dim=1) 

        combined_rejected_input_ids = torch.cat([rejected_prompt.input_ids, rejected_response.input_ids], dim=1)    
        combined_rejected_attention_mask = torch.cat([rejected_prompt.attention_mask, rejected_response.attention_mask], dim=1)


        return {
            'chosen_input_ids':combined_chosen_input_ids,
            'chosen_attention_mask':combined_chosen_attention_mask,
            'rejected_input_ids':combined_rejected_input_ids,
            'rejected_attention_mask':combined_rejected_attention_mask,
        }
    
    dataset = dataset.map(
        preprocess, 
        batched=True,
        fn_kwargs={
            'tokenizer': tokenizer 
        }
    )

    dataset.set_format(type="torch", )
    dataset = dataset.remove_columns(["chosen_prompt", "rejected_prompt", "chosen_response", "rejected_response"])

    batch_size = 32

    
    world_size = 4
    mp.spawn(process_data, args=(model_path, world_size, dataset, batch_size), nprocs=world_size)

    # 合并代码示例
    final_hidden_states = {i: [] for i in range(33)}

    for rank in range(4):
        rank_states = torch.load(f"/data/chaojian/Multi-alignment/llama2_preference_hidden_states/rank_{rank}_hidden_states.pth")
        for layer in final_hidden_states:
            final_hidden_states[layer].extend(rank_states[layer])

    # 拼接所有结果
    for layer in final_hidden_states:
        final_hidden_states[layer] = torch.cat(final_hidden_states[layer], dim=0)

    torch.save(final_hidden_states, "/data/chaojian/Multi-alignment/llama2_preference_hidden_states/final_bias_hidden_states.pth")
