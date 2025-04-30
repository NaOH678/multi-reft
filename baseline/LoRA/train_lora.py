from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser,
    set_seed,
    TrainingArguments as HfTrainingArguments,
    DataCollatorForSeq2Seq
)
from args import LoRAArguments, TrainingArguments, DataArguments
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
from peft import LoraConfig, get_peft_model
import transformers
from accelerate import Accelerator
import os
import wandb

SUBSPACE_NAMES = [
    'ethic', 'truth', 'safety', 'toxicity', 'stereotype', 'helpfulness'
]

def load_subdataset(subtask):
    dataset_paths = {
        'truthful': '/data/chaojian/Multi-alignment/dataset/alignment_truthful',
        'helpful': '/data/chaojian/Multi-alignment/dataset/ultra_feedback.json',
        'moral': '/data/chaojian/Multi-alignment/dataset/alignment_moral',
        'safety': '/data/chaojian/Multi-alignment/dataset/alignment_pku_safety',
        'stereotype': '/data/chaojian/Multi-alignment/dataset/alignment_stereotype',
        'toxic': '/data/chaojian/Multi-alignment/dataset/alignment_toxic'
    }
    
    if subtask == 'helpful':
        return load_dataset('json', data_files=dataset_paths[subtask])['train']
    return load_from_disk(dataset_paths[subtask])['train']

def main():
    accelerator = Accelerator()
    parser = HfArgumentParser((LoRAArguments, TrainingArguments, DataArguments))
    lora_args, training_args, data_args = parser.parse_args_into_dataclasses()
    
    set_seed(training_args.seed)
    
    # 加载模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.model_name_or_path,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.target_modules or ["q_proj", "v_proj"],
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 加载数据集
    if data_args.dataset_name == "combined":
        datasets = [load_subdataset(name) for name in SUBSPACE_NAMES]
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_subdataset(data_args.dataset_name)
    
    if data_args.max_samples:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))
    
    # 数据预处理
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

        # 批量生成prompt
        full_prompts = []
        for i in range(len(examples["input"])):
            if "instruction" in examples:
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
                    
            elif "full_output" in examples:
                full_prompts.append(prompt_no_input % (
                    examples["input"][i],
                    examples['full_output'][i],
                ) + tokenizer.eos_token)
            else:
                raise ValueError("数据集格式不符合要求")

        print(full_prompts[0])
        # 批量分词
        tokenized = tokenizer(
            full_prompts,
            truncation=True,
            max_length=training_args.model_max_length,
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
        return dict(input_ids=tokenized["input_ids"].tolist(),
                    attention_mask=tokenized["attention_mask"].tolist(),
                    labels=tokenized["labels"].tolist())
    
    if data_args.dataset_name == "helpful":
        dataset = dataset.map(preprocess_function, batched=True).remove_columns(["instruction", "input", "output"])
    else:
        dataset = dataset.map(preprocess_function, batched=True).remove_columns(["input", "full_output"])
    print(len(dataset))
    print(dataset[0])
    print()
    # 训练配置
    training_args = transformers.TrainingArguments(
        num_train_epochs=training_args.num_train_epochs,
        output_dir=training_args.output_dir,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=training_args.warmup_ratio,
        weight_decay=training_args.weight_decay,
        save_strategy=training_args.save_strategy,
        report_to="wandb" if accelerator.is_main_process else None,
        label_names=["labels"],
        warmup_steps=training_args.warmup_steps,
        logging_steps=1,
    )
    
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=f"LoRA_{data_args.dataset_name}", config=vars(training_args))
    
    # 训练器
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding="longest"
        )
    )
    
    # 开始训练
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
if __name__ == "__main__":
    main()
