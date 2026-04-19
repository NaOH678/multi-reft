from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser,
    set_seed,
    TrainingArguments as HfTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import transformers


import torch
import wandb
import os



SUBSPACE_NAMES = [
    'moral', 'truthful', 'safety', 'toxicity', 'stereotype', 'helpful'
]

def load_subdataset(subtask, warmupset):
    dataset_paths = {
        'truthful': '../dataset/alignment_truthful_format',
        'helpful': '../dataset/ultra_feedback.json',
        'moral': '../dataset/alignment_moral_format',
        'safety': '../dataset/alignment_pku_safety_format',
        'stereotype': '../dataset/alignment_stereotype_format',
        'toxic': '../dataset/alignment_toxic_format'
    }
    
    if subtask == warmupset:
        return load_dataset('json', data_files=dataset_paths[warmupset])['train'].shuffle(seed=42).select(range(3000, 4000))
    return load_from_disk(dataset_paths[subtask])['train'].shuffle(seed=42).select(range(1000))

def get_lora_param_names(model):
    return [name for name, param in model.named_parameters() if 'lora_' in name]
    # return [name for name, param in model.named_parameters()]
def accumulate_lora_grads(model, dataloader, param_names, num_steps):
    model.zero_grad()
    grad_dict = {name: None for name in param_names}
    
    step = 0
    for batch in tqdm(dataloader):
        # print(batch)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
        
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if name in param_names and param.grad is not None:
                grad = param.grad.detach().clone().view(-1)
                if grad_dict[name] is None:
                    grad_dict[name] = grad
                else:
                    if grad_dict[name].shape != grad.shape:
                        raise ValueError(f"Shape mismatch for {name}: {grad_dict[name].shape} vs {grad.shape}")
                    grad_dict[name] += grad

        model.zero_grad()
        step += 1

        for name in grad_dict:
            if grad_dict[name] is not None:
                grad_dict[name] /= step
        
    return grad_dict

def cosine_similarity_dict(dict_a, dict_b):
    grads_a = []
    grads_b = []

    for name in dict_a:
        if name in dict_b and dict_a[name].numel() > 0 and dict_b[name].numel() > 0:
            grads_a.append(dict_a[name])
            grads_b.append(dict_b[name])

    if not grads_a:
        return None

    vec_a = torch.cat(grads_a)
    vec_b = torch.cat(grads_b)
    print(vec_a.shape, vec_b.shape)

    return torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()

def analyze_lora_conflict(model, loader_a, loader_b, num_steps=8, save_path=None):
    model.eval()
    param_names = get_lora_param_names(model)

    print(f"Analyzing {len(param_names)} LoRA parameters...")

    grads_a = accumulate_lora_grads(model, loader_a, param_names, num_steps)
    model.zero_grad()
    grads_b = accumulate_lora_grads(model, loader_b, param_names, num_steps)

    sim_total = cosine_similarity_dict(grads_a, grads_b)
    print(f"\n[LoRA] Cosine Similarity between Task A and B gradients: {sim:.4f}")

    results = {
        "overall_similarity": sim_total,
        "per_parameter_similarity": {}
    }




    # 🔍 可选：逐个参数输出
    print(f"\n{'Parameter':40s} | Cosine Similarity")
    print("-" * 70)
    for name in grads_a:
        if name in grads_b and grads_a[name].numel() > 0 and grads_b[name].numel() > 0:
            sim = torch.nn.functional.cosine_similarity(grads_a[name], grads_b[name], dim=0).item()
            results["per_parameter_similarity"][name] = sim
            print(f"{name:40s} | {sim:.4f}")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved gradient similarity results to {save_path}")
    

class GradientAnalysisDataset(Dataset):

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"])
        }



if __name__ == "__main__":

    model_name_or_path='../Llama-2-7b-hf'
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    task_type = "CAUSAL_LM"
    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16
        ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="right",
            use_fast=False
        )
    tokenizer.pad_token = tokenizer.unk_token
        
    # 配置LoRA
    peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj" "v_proj" "k_proj" "o_proj"],
            lora_dropout=lora_dropout,
            task_type=task_type
        )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    
    warmupset = 'helpful'
    a_set = 'helpful'
    b_set = 'toxic'

    dataset1 = load_subdataset(a_set)
    dataset2 = load_subdataset(b_set)

    model = PeftModel.from_pretrained(model, "baseline/LoRA/lora_output_helpful/checkpoint-125", is_trainable=True)


    def preprocess_function(examples):
        # 定义提示模板
        prompt_input = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
%s

"""
            
        prompt_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
%s

### Response:
%s

"""


        full_prompts = []
        for i in range(len(examples["input"])):
            
            if examples["input"][i]:
                    full_prompts.append(prompt_input % (
                        examples["instruction"][i],
                        examples["input"][i],
                        examples["output"][i]
                    ) + tokenizer.eos_token)
            else:
                    full_prompts.append(prompt_no_input % (
                        examples["instruction"][i],
                        examples['output'][i],
                    ) + tokenizer.eos_token)


        tokenized = tokenizer(
            full_prompts,
            truncation=True,
            max_length=768,
            padding="longest",
            return_tensors="pt"
        )

        # 设置labels（仅预测response部分）
        response_starts = [
            len(tokenizer(prompt.split("### Response:")[0], return_tensors="pt")["input_ids"][0])
            for prompt in full_prompts
        ]
        
        labels = tokenized["input_ids"].clone()
        for idx, start_pos in enumerate(response_starts):
            labels[idx, :start_pos] = -100
        
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized["labels"] = labels
        # 转换张量为列表以兼容数据集映射
        return tokenized
    
    dataset1 = dataset1.map(preprocess_function, batched=True).remove_columns(["instruction", "input", "output"])
    dataset2 = dataset2.map(preprocess_function, batched=True).remove_columns(["instruction", "input", "output"])

    # print(dataset1[:2])

    dataset1 = GradientAnalysisDataset(dataset1)
    dataset2 = GradientAnalysisDataset(dataset2)
    print(dataset1[0])

    # print(dataset1[0])

    loader_a = DataLoader(dataset1, batch_size=1, shuffle=True)
    loader_b = DataLoader(dataset2, batch_size=1, shuffle=True)



    analyze_lora_conflict(model, loader_a, loader_b)




    




