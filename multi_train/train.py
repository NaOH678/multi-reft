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
    NoreftIntervention,
    ReftTrainer,
    ReftTrainerForCausalLMDistributed
)
import torch.distributed as dist
import wandb
import os

class SubloreftIntervention(LoreftIntervention):
    """
    This is a LoReFT that supports subspace interventions!
    """
    def forward(
        self, base, source=None, subspaces=None
    ):
        assert subspaces is not None
        output = []
        
        rotated_base = self.rotate_layer(base)
        diff = self.act_fn(self.learned_source(base)) - rotated_base
        
        batched_subspace = []
        batched_weights = []
        
        for example_i in range(len(subspaces)):
            LHS = (diff[example_i, :, subspaces[example_i]])
            RHS = self.rotate_layer.weight[..., subspaces[example_i]].T
            # print(diff.shape, LHS.shape, RHS.shape, base.shape, subspaces)
            batched_subspace += [LHS]
            batched_weights += [RHS]

        
        batched_subspace = torch.stack(batched_subspace, dim=0)
        batched_weights = torch.stack(batched_weights, dim=0)

        output = base + torch.bmm(batched_subspace, batched_weights)

        return self.dropout(output.to(base.dtype))



if __name__== "__main__":

    accelerator = Accelerator()
    rank = accelerator.process_index

    SUBSPACE_NAMES = [
    'ethic', 'truth', 'safety', 'toxicity', 'stereotype', 'helpfulness'
    ]


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
        torch_dtype=torch.bfloat16
    )

    # get tokenizer
    model_max_length = training_args.model_max_length
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=model_max_length, 
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token


    # load data
    if data_args.max_samples:
        max_samples = data_args.max_samples
    
    else:
        # 暂时不能使用！！！
        percentage = data_args.percentage

    
    subtask = reftargs.subtask
    if subtask == 'truthful':
        data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_truthful')['train']
        # truthful_data = truthful_data.select(range(min(max_samples, len(truthful_data))))

    elif subtask == 'helpful':
        data = load_dataset('json',data_files='/data/chaojian/Multi-alignment/dataset/ultra_feedback.json')['train']
        # helpful_data = helpful_data.select(range(min(max_samples, len(helpful_data))))
    
    elif subtask == 'moral':
        data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_moral')['train']
        # moral_data = moral_data.select(range(min(max_samples, len(moral_data))))
    
    elif subtask == 'safety':
        data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_pku_safety')['train']
        # safety_data = safety_data.select(range(min(max_samples, len(safety_data))))
    
    elif subtask == 'stereotype':
        data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_stereotype')['train']
        # stereotype_data = stereotype_data.select(range(min(max_samples, len(stereotype_data))))
    
    elif subtask == 'toxic':
        data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_toxic')['train']
        # toxicity_data = toxicity_data.select(range(min(max_samples, len(toxicity_data))))
    
    # data = data.select(range(min(max_samples, len(data))))
    print(data[0])
    



    # helpful_data = helpful_data.map(lambda x: {"subspaces": SUBSPACES['helpfulness']})
    # moral_data = moral_data.map(lambda x: {"subspaces": SUBSPACES['ethic']})
    # safety_data = safety_data.map(lambda x: {"subspaces": SUBSPACES['safety']})
    # stereotype_data = stereotype_data.map(lambda x: {"subspaces": SUBSPACES['stereotype']})
    # toxicity_data = toxicity_data.map(lambda x: {"subspaces": SUBSPACES['toxicity']})
    data = data.map(lambda x: {"subspaces": SUBSPACES['truth']})

    
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
            "intervention": NodireftIntervention(
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


    train_dataset = ReftSupervisedDataset(
        "Nodireloreft", None, tokenizer, dataset=subspace_dataset,
        **{"num_interventions": len(reft_model.interventions), "position": reftargs.position , "share_weights": True},          # 该成f1+l1 梯度是0？？？？
        input_field=None, instruction_field="input", output_field="full_output", 
        seed=training_args.seed, max_n_example=min(data_args.max_samples, len(data)),
        no_stop=False
    )
    print(train_dataset[0])


    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    if rank == 0:
        # os.environ["WANDB_MODE"] = "offline"  # 如果你用 online 模式可以去掉这一行
        wandb.init(project=f"Reft_{reftargs.subtask}", name=f"first_train_{reftargs.subtask}")
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
            seed=training_args.seed
        ))     

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
        seed=training_args.seed
    )

    
    trainer =ReftTrainerForCausalLMDistributed(
        model=reft_model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator
    )
    if dist.is_initialized():
        dist.barrier()
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()

