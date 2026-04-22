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
    ReftTrainerForCausalLMDistributed
)
import torch.distributed as dist
import wandb
import os




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

    def _prepare_subtask_dataset(subtask_name, include_truth_mix_for_stereotype=True):
        if subtask_name == 'truthful':
            dataset = load_from_disk('./dataset/alignment_truthful_format')['train']

        elif subtask_name == 'moral':
            dataset = load_from_disk('./dataset/alignment_moral_cls')

        elif subtask_name == 'safety':
            dataset = load_from_disk('./dataset/alignment_pku_safety_format')['train']

        elif subtask_name == 'stereotype':
            stereotype_data = load_from_disk('./dataset/alignment_stereotype_format')['train']
            if include_truth_mix_for_stereotype:
                truthful_data = load_from_disk('./dataset/alignment_truthful_format')['train']
                truthful_data = truthful_data.shuffle(seed=training_args.seed)
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

            dataset = dataset.map(_prepare_truthful)

        elif subtask_name == 'moral':
            def _prepare_ethics(example):
                output = _format_ethics_out(example)
                return {
                    "output": output,
                    "subspace_labels": subspace_label_id,
                }

            dataset = dataset.map(_prepare_ethics)

        else:
            dataset = dataset.map(lambda x: {"subspace_labels": subspace_label_id})

        return dataset

    subtask = _canonicalize_subtask(reftargs.subtask)
    if subtask == "combined":
        datasets = [
            _prepare_subtask_dataset(name, include_truth_mix_for_stereotype=False)
            for name in SUBSPACE_NAMES
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


    train_dataset = ReftSupervisedDataset(
        "Loreft", None, tokenizer, dataset=subspace_dataset,
        **{"num_interventions": len(reft_model.interventions), "position": reftargs.position , "share_weights": True},          # 该成f1+l1 梯度是0？？？？
        input_field='input', instruction_field="instruction", output_field="output", 
        seed=training_args.seed, max_n_example=min(max_examples, len(data)),
        no_stop=False
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
        wandb.init(project=f"NodireftwithSubtoken_{reftargs.subtask}", name=f"first_train_{reftargs.subtask}")
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
