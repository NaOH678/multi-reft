from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, AutoConfig, set_seed
from args import ReftArguments, TrainingArguments, DataArguments
from datasets import load_dataset, concatenate_datasets, load_from_disk
import transformers
import torch
from accelerate import Accelerator
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    LoreftIntervention,
    NodireftIntervention,
    SubNodireftIntervention,
    NoreftIntervention,
    ReftTrainer,
    ReftTrainerForCausalLMDistributed,
    get_intervention_locations,
)
import torch.distributed as dist
import wandb
import os
from pathlib import Path

IGNORE_INDEX = -100




if __name__== "__main__":

    accelerator = Accelerator()
    rank = accelerator.process_index

    SUBSPACE_NAMES = [
        'truthful','toxicity', 'stereotype', 'safety', 'moral',
    ]
    SUBTASK_ALIASES = {"toxic": "toxicity", "combine": "combined"}


    parser = HfArgumentParser(
        (ReftArguments, TrainingArguments, DataArguments)

    )
    (
        reftargs,
        training_args,
        data_args,

    ) = parser.parse_args_into_dataclasses()

    subspace_rank = reftargs.subspace_rank

    SUBSPACES = {
    name: list(range(subspace_rank * i, subspace_rank * (i + 1)))
    for i, name in enumerate(SUBSPACE_NAMES)
    }
    
    set_seed(training_args.seed)
    model_name_or_path = training_args.model_name_or_path # yahma/llama-7b-hf or yahma/llama-13b-hf
    safety_dataset_path = os.environ.get(
        "SAFETY_DATASET_PATH",
        "./dataset/alignment_wildjailbreak_safety_format",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    # get tokenizer
    model_max_length = training_args.model_max_length
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=model_max_length, 
        padding_side="right", use_fast=False)
    # Avoid using EOS as PAD; otherwise EOS tokens may be masked out in attention.
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    # Keep model/generation configs aligned with tokenizer to avoid repeated warnings.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id


    # load data
    if data_args.max_samples:
        max_samples = data_args.max_samples
    
    else:
        # 暂时不能使用！！！
        percentage = data_args.percentage

    
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

    def _canonicalize_subtask(subtask_name):
        normalized = (subtask_name or "").strip().lower()
        return SUBTASK_ALIASES.get(normalized, normalized)

    def _infer_model_label(model_path_or_name):
        model_path = Path(str(model_path_or_name).rstrip("/"))
        parts = [part for part in model_path.parts if part not in {"", "."}]

        if "snapshots" in parts:
            snapshot_idx = parts.index("snapshots")
            if snapshot_idx >= 1:
                return parts[snapshot_idx - 1]

        if parts:
            return parts[-1]

        return str(model_path_or_name).replace("/", "_")

    def _drop_empty_instruction_output(dataset):
        def _valid_example(example):
            instruction = str(example.get("instruction", "") or "").strip()
            output = str(example.get("output", "") or "").strip()
            return instruction != "" and output != ""

        before = len(dataset)
        dataset = dataset.filter(
            _valid_example,
            load_from_cache_file=False,
            keep_in_memory=True,
        )
        after = len(dataset)
        if rank == 0 and after != before:
            print(f"filtered empty instruction/output rows: before={before}, after={after}")
        return dataset

    def _map_in_memory(dataset, fn):
        return dataset.map(
            fn,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    def _preprocess_reft_supervised(examples, tokenizer, model_max_length, position, num_interventions, share_weights, no_stop=False):
        prompt_input_template = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""
        prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""
        result = {
            "input_ids": [],
            "labels": [],
            "intervention_locations": [],
            "attention_mask": [],
            "id": [],
        }
        if "subspace_labels" in examples:
            result["subspace_labels"] = []

        instructions = examples["instruction"]
        outputs = examples["output"]
        inputs = examples["input"] if "input" in examples else [""] * len(instructions)
        subspace_labels = examples.get("subspace_labels")

        for idx, (instruction, model_input, output) in enumerate(zip(instructions, inputs, outputs)):
            instruction = "" if instruction is None else str(instruction)
            model_input = "" if model_input is None else str(model_input)
            output = "" if output is None else str(output)

            if model_input == "":
                base_prompt = prompt_no_input_template % instruction
            else:
                base_prompt = prompt_input_template % (instruction, model_input)

            prompt_ids = tokenizer(
                base_prompt,
                max_length=model_max_length,
                truncation=True,
            )["input_ids"]
            base_prompt_length = len(prompt_ids)
            last_position = base_prompt_length - 1

            base_input = base_prompt + output
            if not no_stop:
                base_input += tokenizer.eos_token

            input_ids = tokenizer(
                base_input,
                max_length=model_max_length,
                truncation=True,
            )["input_ids"]
            labels = list(input_ids)
            for label_idx in range(min(base_prompt_length, len(labels))):
                labels[label_idx] = IGNORE_INDEX

            intervention_locations = get_intervention_locations(
                last_position=last_position,
                positions=position,
                pad_mode="first",
                num_interventions=num_interventions,
                share_weights=share_weights,
            )
            intervention_locations = [
                [loc + 1 for loc in per_intervention_locs]
                for per_intervention_locs in intervention_locations
            ]

            result["input_ids"].append([tokenizer.pad_token_id] + input_ids)
            result["labels"].append([IGNORE_INDEX] + labels)
            result["intervention_locations"].append(intervention_locations)
            result["attention_mask"].append([0] + [1] * len(input_ids))
            result["id"].append(idx)
            if subspace_labels is not None:
                result["subspace_labels"].append(subspace_labels[idx])

        return result

    def _prepare_subtask_dataset(subtask_name, include_truth_mix_for_stereotype=True):
        if subtask_name == 'truthful':
            dataset = load_from_disk('./dataset/alignment_truthful_format')['train']

        elif subtask_name == 'moral':
            dataset = load_from_disk('./dataset/alignment_moral_cls')

        elif subtask_name == 'safety':
            dataset = load_from_disk(safety_dataset_path)['train']
            dataset = _drop_empty_instruction_output(dataset)

        elif subtask_name == 'stereotype':
            stereotype_data = load_from_disk('./dataset/alignment_stereotype_format')['train']
            if include_truth_mix_for_stereotype:
                truthful_data = load_from_disk('./dataset/alignment_truthful_format')['train']
                truthful_data = truthful_data.shuffle(seed=training_args.seed)
                truthful_data = truthful_data.select(range(min(5000, len(truthful_data))))
                truthful_data = _map_in_memory(
                    truthful_data,
                    lambda x: {"output": _format_truthful_output(x)},
                )

                # Keep only shared columns so concatenate_datasets can merge safely.
                common_columns = [col for col in stereotype_data.column_names if col in truthful_data.column_names]
                stereotype_data = stereotype_data.remove_columns(
                    [col for col in stereotype_data.column_names if col not in common_columns]
                )
                truthful_data = truthful_data.remove_columns(
                    [col for col in truthful_data.column_names if col not in common_columns]
                )
                dataset = concatenate_datasets([stereotype_data, truthful_data])
            else:
                dataset = stereotype_data

        elif subtask_name == 'toxicity':
            dataset = load_from_disk('./dataset/alignment_toxic_format')['train']

        else:
            raise ValueError(
                f"Unsupported subtask: {subtask_name}. Allowed: {SUBSPACE_NAMES + ['combined']}"
            )

        subspace_label_id = SUBSPACE_NAMES.index(subtask_name)
        # Align truthful formatting with LoReFT commonsense setup:
        # target text should include trigger tokens, not only "endingX".
        if subtask_name == "truthful":
            def _prepare_truthful(example):
                output = _format_truthful_output(example)
                return {
                    "output": output,
                    "subspace_labels": subspace_label_id,
                }

            dataset = _map_in_memory(dataset, _prepare_truthful)

        elif subtask_name == 'moral':
            def _prepare_ethics(example):
                output = _format_ethics_out(example)
                return {
                    "output": output,
                    "subspace_labels": subspace_label_id,
                }

            dataset = _map_in_memory(dataset, _prepare_ethics)

        else:
            dataset = _map_in_memory(dataset, lambda x: {"subspace_labels": subspace_label_id})

        return dataset

    subtask = _canonicalize_subtask(reftargs.subtask)
    model_label = _infer_model_label(model_name_or_path)
    wandb_run_name = training_args.run_name or f"first_train_{model_label}_{subtask}"
    if subtask == "combined":
        combined_subtasks = [name for name in SUBSPACE_NAMES if name != "safety"]
        datasets = [
            _prepare_subtask_dataset(name, include_truth_mix_for_stereotype=False)
            for name in combined_subtasks
        ]
        data = concatenate_datasets(datasets).shuffle(seed=training_args.seed)
    else:
        data = _prepare_subtask_dataset(subtask, include_truth_mix_for_stereotype=True)

    print(data[0])
    



    # helpful_data = helpful_data.map(lambda x: {"subspaces": SUBSPACES['helpfulness']})
    # moral_data = moral_data.map(lambda x: {"subspaces": SUBSPACES['ethic']})
    # safety_data = safety_data.map(lambda x: {"subspaces": SUBSPACES['safety']})
    # stereotype_data = stereotype_data.map(lambda x: {"subspaces": SUBSPACES['stereotype']})
    # toxicity_data = toxicity_data.map(lambda x: {"subspaces": SUBSPACES['toxicity']})
    # data = data.map(lambda x: {"subspaces": SUBSPACES['truth']})

    
    # subspace_dataset = concatenate_datasets([helpful_data, 
    #                                          moral_data, 
    #                                          safety_data, 
    #                                          stereotype_data, 
    #                                          toxicity_data, 
    #                                          truthful_data])

    subspace_dataset = data

    # print(type(reftargs.target_layers))
    # print(reftargs.target_layers)
    if reftargs.target_layers == [-1]:
        TARGET_LAYERS = list(range(len(model.model.layers)))
    else:
        TARGET_LAYERS = reftargs.target_layers

    #get reft model
    # reft_config = ReftConfig(representations=[
    #     {
    #         "layer": layer, "component": "block_output",
    #         "low_rank_dimension": reftargs.subspace_rank,
    #         "intervention": SubloreftIntervention(
    #         embed_dim=model.config.hidden_size, 
    #         low_rank_dimension=reftargs.subspace_rank,
    #         dropout=training_args.dropout,)
    #     }
    #     for layer in TARGET_LAYERS]
    # )

    reft_config = ReftConfig(representations=[
        {
            "layer": layer, "component": "block_output",
            "intervention": LoreftIntervention(
            embed_dim=model.config.hidden_size, 
            low_rank_dimension=reftargs.subspace_rank, 
            dropout=training_args.dropout,
            add_bias=False)
        }
        for layer in TARGET_LAYERS
        ]
    )
    
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()

    if data_args.max_samples:
        max_examples = data_args.max_samples
    else:
        max_examples = len(data)


    preprocessing_num_workers = max(1, min(8, os.cpu_count() or 1))
    with accelerator.main_process_first():
        train_dataset = subspace_dataset.shuffle(seed=training_args.seed)
        train_dataset = train_dataset.select(range(min(max_examples, len(subspace_dataset))))
        train_dataset = train_dataset.map(
            _preprocess_reft_supervised,
            batched=True,
            remove_columns=train_dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "model_max_length": tokenizer.model_max_length,
                "position": reftargs.position,
                "num_interventions": len(reft_model.interventions),
                "share_weights": True,
                "no_stop": False,
            },
            num_proc=preprocessing_num_workers,
        )
    train_dataset.set_format(
        type="torch",
        columns=[
            column
            for column in train_dataset.column_names
            if column in {"input_ids", "labels", "intervention_locations", "subspace_labels", "attention_mask", "id"}
        ],
    )
    print(train_dataset[:5])


    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    if rank == 0:
        # os.environ["WANDB_MODE"] = "offline"  # 如果你用 online 模式可以去掉这一行
        wandb.init(
            project=f"NodireftwithSubtoken_{subtask}",
            name=wandb_run_name,
            tags=[model_label, subtask],
        )
        print(torch.cuda.device_count())
        wandb.log(dict(
            num_gpus=torch.cuda.device_count(),
            num_train_epochs=training_args.num_train_epochs, 
            learning_rate=training_args.learning_rate, 
            per_device_train_batch_size=training_args.per_device_train_batch_size, 
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_ratio=training_args.warmup_ratio,
            weight_decay=training_args.weight_decay,
            positions=reftargs.position,
            dropout=training_args.dropout,
            subspace_rank=reftargs.subspace_rank,
            target_layers=reftargs.target_layers,
            seed=training_args.seed,
            model_name_or_path=model_name_or_path,
            model_label=model_label,
            wandb_run_name=wandb_run_name,
        ))     

    resume_from_checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    if isinstance(resume_from_checkpoint, str):
        normalized_resume = resume_from_checkpoint.strip()
        if normalized_resume == "":
            resume_from_checkpoint = None
        elif normalized_resume.lower() in {"true", "auto", "latest"}:
            resume_from_checkpoint = True
        else:
            resume_from_checkpoint = normalized_resume

    training_args = transformers.TrainingArguments(
        num_train_epochs=training_args.num_train_epochs, 
        output_dir=training_args.output_dir, 
        learning_rate=training_args.learning_rate, 
        report_to='wandb',
        run_name=wandb_run_name,
        per_device_train_batch_size=training_args.per_device_train_batch_size, 
        logging_steps=1,
        ddp_find_unused_parameters=False,  # 关键修改
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_ratio=training_args.warmup_ratio,
        save_total_limit=10,
        save_strategy=training_args.save_strategy,
        weight_decay=training_args.weight_decay,
        seed=training_args.seed,
        dataloader_num_workers=8,
        lr_scheduler_type=training_args.lr_scheduler_type,
  
    )
    

    trainer =ReftTrainerForCausalLMDistributed(
        model=reft_model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        data_collator=data_collator
    )
    if rank == 0:
        print("remove_unused_columns =", training_args.remove_unused_columns)

    # 先构 dataloader 看真实喂给模型的 batch
    dl = trainer.get_train_dataloader()
    batch = next(iter(dl))

    if rank == 0:
        print("batch keys =", list(batch.keys()))
        print("intervention_locations shape =", batch["intervention_locations"].shape)
        sup_counts = (batch["labels"] != -100).sum(dim=1)
        print("supervised tokens per sample (first 8) =", sup_counts[:8].tolist())

        i = 0
        target_ids = batch["labels"][i][batch["labels"][i] != -100]
        print("sample0 target_len =", target_ids.numel())
        print("sample0 target_text =", tokenizer.decode(target_ids.tolist()))
    if rank == 0:
        import numpy as np
        counts = []
        n = min(500, len(train_dataset))
        for i in range(n):
            counts.append(int((train_dataset[i]["labels"] != -100).sum().item()))
        print("target token stats over", n, "samples:",
            {"p10": float(np.percentile(counts, 10)),
            "p50": float(np.percentile(counts, 50)),
            "p90": float(np.percentile(counts, 90)),
            "mean": float(np.mean(counts))})
    if dist.is_initialized():
        dist.barrier()
    
    if resume_from_checkpoint:
        if rank == 0:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
