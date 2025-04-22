import torch
import torch.multiprocessing as mp
import torch
import transformers
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import random
import gc



class ModelWithHiddenStates(AutoModelForCausalLM):
    def __init__(self):
        super().__init__()
        self.hidden_states = None

    def forward(self, input_ids, **kwargs):
        outputs = super().forward(input_ids, **kwargs)

        # print(outputs.hidden_states)
    
        return outputs.hidden_states         
    # 对于每一个batch，输出的一个元组，元组长度为33（llama2的层数），
    # 元组中的每一个元素是一个batch在每一层的hidden state，（batch， seq_length， hidden_size）

def process_data(rank, model_path, world_size, dataset, batch_size):
    # 设置设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 加载模型并移至当前GPU
    model = ModelWithHiddenStates.from_pretrained(
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
            diff = chosen_outputs[layer][:, -1, :] - rejected_outputs[layer][:, -1, :]
            all_hidden_states[layer].append(diff.detach().to('cpu', non_blocking=True))
    
    # 保存当前进程的结果
    torch.save(all_hidden_states, f"/data/chaojian/Multi-alignment/llama2_preference_hidden_states/rank_{rank}_hidden_states.pth")

if __name__ == "__main__":
    model_path = '/data/chaojian/Llama-2-7b-chat-hf'

    ds_binarized = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned",)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<pad>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")

    formatted_texts_chosen = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in ds_binarized['train']['chosen']]
    formatted_texts_rejected = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in ds_binarized['train']['rejected']]  

    # samples_texts = random.sample(formatted_texts, 60000)
    dataset = Dataset.from_dict({"chosen": formatted_texts_chosen, "rejected": formatted_texts_rejected})

    def preprocess_function(examples):

        chosen = tokenizer(examples["chosen"], padding=True, max_length=930, truncation=True)
        rejected = tokenizer(examples["rejected"], padding=True, max_length=930, truncation=True)
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    dataset = dataset.map(preprocess_function, batched=True)
    dataset = dataset.remove_columns(["chosen", "rejected"])
    dataset.set_format(type="torch",)
    batch_size = 16

   
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

    torch.save(final_hidden_states, "/data/chaojian/Multi-alignment/llama2_preference_hidden_states/final_rejected_hidden_states.pth")