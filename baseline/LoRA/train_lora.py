from pathlib import Path
from typing import Dict, List

import torch
import transformers
import wandb
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)

from args import DataArguments, LoRAArguments, TrainingArguments

SUBSPACE_NAMES = ["truthful", "toxicity", "stereotype", "safety", "moral"]
SUBTASK_ALIASES = {"toxic": "toxicity"}

PROMPT_INPUT = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

PROMPT_NO_INPUT = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""


truthful_trigger = "the correct answer is "
ethics_trigger =  "the action is "

def _format_truthful_output(example):
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(truthful_trigger):
        output = f"{truthful_trigger}{answer}"
    return output

def _format_ethics_out(example):
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(ethics_trigger):
        output = f"{ethics_trigger}{answer}"
    return output

def canonicalize_subtask(subtask: str) -> str:
    normalized = (subtask or "").strip().lower()
    return SUBTASK_ALIASES.get(normalized, normalized)


def load_subdataset(subtask: str, dataset_root: Path, seed: int):
    subtask = canonicalize_subtask(subtask)
    dataset_paths = {
        "truthful": dataset_root / "alignment_truthful_format",
        "moral": dataset_root / "alignment_moral_cls",
        "safety": dataset_root / "alignment_pku_safety_format",
        "stereotype": dataset_root / "alignment_stereotype_format",
        "toxicity": dataset_root / "alignment_toxic_format",
    }

    if subtask not in dataset_paths:
        raise ValueError(f"Unsupported dataset name: {subtask}")

    if subtask == "helpful":
        return load_dataset("json", data_files=str(dataset_paths[subtask]))["train"]
    if subtask == "moral":
        return load_from_disk(str(dataset_paths[subtask]))

    if subtask == "stereotype":
        stereotype_data = load_from_disk('./dataset/alignment_stereotype_format')['train']
        truthful_data = load_from_disk('./dataset/alignment_truthful_format')['train']
        truthful_data = truthful_data.shuffle(seed=seed)
        truthful_data = truthful_data.select(range(min(5000, len(truthful_data))))
        truthful_data = truthful_data.map(lambda x: {"output": _format_truthful_output(x)})

        # Keep only shared columns so concatenate_datasets can merge safely.
        common_columns = [col for col in stereotype_data.column_names if col in truthful_data.column_names]
        stereotype_data = stereotype_data.remove_columns(
            [col for col in stereotype_data.column_names if col not in common_columns]
        )
        truthful_data = truthful_data.remove_columns(
            [col for col in truthful_data.column_names if col not in common_columns]
        )

        data = concatenate_datasets([stereotype_data, truthful_data])
        return data
    return load_from_disk(str(dataset_paths[subtask]))["train"]


def prepare_subtask_dataset(subtask: str, dataset_root: Path, seed: int):
    subtask = canonicalize_subtask(subtask)
    if subtask not in SUBSPACE_NAMES:
        raise ValueError(f"Unsupported subtask: {subtask}. Allowed: {SUBSPACE_NAMES}")

    data = load_subdataset(subtask, dataset_root, seed)
    subspace_label_id = SUBSPACE_NAMES.index(subtask)

    if subtask == "truthful":
        def _prepare_truthful(example):
            output = _format_truthful_output(example)
            return {
                "output": output,
                "subspace_labels": subspace_label_id,
            }

        data = data.map(_prepare_truthful)
    
    elif subtask == 'moral':
        def _prepare_ethics(example):
            output = _format_ethics_out(example)
            return {
                "output": output,
                "subspace_labels": subspace_label_id,
            }

        data = data.map(_prepare_ethics)
    else:
        data = data.map(lambda _: {"subspace_labels": subspace_label_id})

    return data


def build_dataset(data_args: DataArguments, dataset_root: Path, seed: int):
    dataset_name = canonicalize_subtask(data_args.dataset_name)

    if dataset_name == "combined":
        datasets = [prepare_subtask_dataset(name, dataset_root, seed) for name in SUBSPACE_NAMES]
        dataset = concatenate_datasets(datasets)
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = prepare_subtask_dataset(dataset_name, dataset_root, seed)

    if data_args.max_samples:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    return dataset, dataset_name


def preprocess_function(examples: Dict[str, List], tokenizer, model_max_length: int):
    if "instruction" not in examples or "output" not in examples:
        raise ValueError("Dataset must contain 'instruction' and 'output' to match multi_train/train.py logic.")

    instructions = examples["instruction"]
    outputs = examples["output"]
    inputs = examples["input"] if "input" in examples else [""] * len(instructions)

    full_prompts = []
    base_prompts = []
    for instruction, model_input, output in zip(instructions, inputs, outputs):
        instruction = "" if instruction is None else str(instruction)
        model_input = "" if model_input is None else str(model_input)
        output = "" if output is None else str(output)

        if model_input == "":
            base_prompt = PROMPT_NO_INPUT % instruction
        else:
            base_prompt = PROMPT_INPUT % (instruction, model_input)

        full_prompts.append(base_prompt + output + tokenizer.eos_token)
        base_prompts.append(base_prompt)

    tokenized = tokenizer(
        full_prompts,
        truncation=True,
        max_length=model_max_length,
        padding="longest",
    )

    labels = []
    for input_ids, base_prompt in zip(tokenized["input_ids"], base_prompts):
        prompt_ids = tokenizer(
            base_prompt,
            truncation=True,
            max_length=model_max_length,
        )["input_ids"]
        prompt_length = min(len(prompt_ids), len(input_ids))

        label_ids = list(input_ids)
        for idx in range(prompt_length):
            label_ids[idx] = -100
        label_ids = [(-100 if token_id == tokenizer.pad_token_id else token_id) for token_id in label_ids]
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def print_full_finetune_trainable_params(model):
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {total_params:,} "
        f"|| trainable%: {100 * trainable_params / total_params:.6f}"
    )


def main():
    accelerator = Accelerator()
    parser = HfArgumentParser((LoRAArguments, TrainingArguments, DataArguments))
    lora_args, train_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "dataset"

    model = AutoModelForCausalLM.from_pretrained(
        train_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(
        train_args.model_name_or_path,
        model_max_length=train_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    dataset, dataset_name = build_dataset(data_args, dataset_root, train_args.seed)
    if accelerator.is_main_process:
        print(f"dataset={dataset_name}, samples={len(dataset)}")
        print(dataset[0])

    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, train_args.model_max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    use_lora = lora_args.lora_rank is not None
    train_mode = "lora" if use_lora else "sft"

    if use_lora:
        peft_config = LoraConfig(
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha if lora_args.lora_alpha is not None else 16,
            target_modules=lora_args.target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_args.lora_dropout if lora_args.lora_dropout is not None else 0.1,
            task_type=lora_args.task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print_full_finetune_trainable_params(model)

    hf_train_args = transformers.TrainingArguments(
        num_train_epochs=train_args.num_train_epochs,
        output_dir=train_args.output_dir,
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        warmup_ratio=train_args.warmup_ratio,
        weight_decay=train_args.weight_decay,
        save_strategy=train_args.save_strategy,
        report_to="wandb" if accelerator.is_main_process else None,
        label_names=["labels"],
        warmup_steps=train_args.warmup_steps,
        logging_steps=1,
        bf16=train_args.bf16,
        fp16=train_args.fp16,
        lr_scheduler_type=train_args.lr_scheduler_type,
        save_total_limit=train_args.save_total_limit,
        ddp_find_unused_parameters=False,
    )

    if accelerator.is_main_process:
        wandb.init(
            project=f"{train_mode.upper()}_{dataset_name}",
            name=f"{train_mode}_{dataset_name}",
            config={
                "mode": train_mode,
                "dataset": dataset_name,
                "model_name_or_path": train_args.model_name_or_path,
                "model_max_length": train_args.model_max_length,
                "learning_rate": train_args.learning_rate,
                "num_train_epochs": train_args.num_train_epochs,
                "per_device_train_batch_size": train_args.per_device_train_batch_size,
                "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
                "max_samples": data_args.max_samples,
                "lora_rank": lora_args.lora_rank,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "target_modules": lora_args.target_modules,
            },
        )

    trainer = transformers.Trainer(
        model=model,
        args=hf_train_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest",
        ),
    )

    trainer.train()
    trainer.save_model(hf_train_args.output_dir)
    tokenizer.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    main()
