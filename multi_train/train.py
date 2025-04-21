from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
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
    

    model_name_or_path = training_args.model_name_or_path # yahma/llama-7b-hf or yahma/llama-13b-hf
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16)

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

    
    helpful_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_helpfulness').shuffle(seed=42)
    helpful_data = helpful_data.select(range(min(max_samples, len(helpful_data))))
    moral_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_moral')['train'].shuffle(seed=42)
    moral_data = moral_data.select(range(min(max_samples, len(moral_data))))
    safety_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_pku_safety')['train'].shuffle(seed=42)
    safety_data = safety_data.select(range(min(max_samples, len(safety_data))))
    stereotype_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_stereotype')['train'].shuffle(seed=42)
    stereotype_data = stereotype_data.select(range(min(max_samples, len(stereotype_data))))
    toxicity_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_toxic')['train'].shuffle(seed=42)
    toxicity_data = toxicity_data.select(range(min(max_samples, len(toxicity_data))))
    truthful_data = load_from_disk('/data/chaojian/Multi-alignment/dataset/alignment_truthful')['train'].shuffle(seed=42)
    truthful_data = truthful_data.select(range(min(max_samples, len(truthful_data))))


    helpful_data = helpful_data.map(lambda x: {"subspaces": SUBSPACES['helpfulness']})
    moral_data = moral_data.map(lambda x: {"subspaces": SUBSPACES['ethic']})
    safety_data = safety_data.map(lambda x: {"subspaces": SUBSPACES['safety']})
    stereotype_data = stereotype_data.map(lambda x: {"subspaces": SUBSPACES['stereotype']})
    toxicity_data = toxicity_data.map(lambda x: {"subspaces": SUBSPACES['toxicity']})
    truthful_data = truthful_data.map(lambda x: {"subspaces": SUBSPACES['truth']})

    # print(helpful_data[1000])
    # print(moral_data[896])
    # print(safety_data[4000])


    subspace_dataset = concatenate_datasets([helpful_data, 
                                             moral_data, 
                                             safety_data, 
                                             stereotype_data, 
                                             toxicity_data, 
                                             truthful_data])


    # print(type(reftargs.target_layers))
    # print(reftargs.target_layers)
    if reftargs.target_layers == [-1]:
        TARGET_LAYERS = list(range(len(model.model.layers)))
    else:
        TARGET_LAYERS = reftargs.target_layers

    # print(TARGET_LAYERS)

    # get reft model
    reft_config = ReftConfig(representations=[
        {
            "layer": layer, "component": "block_output",
            "intervention": SubloreftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=6*reftargs.subspace_rank)
        }
        for layer in TARGET_LAYERS
        ]
    )
    
    reft_model = get_reft_model(model, reft_config)
    reft_model.print_trainable_parameters()


    train_dataset = ReftSupervisedDataset(
    "Subloreft", None, tokenizer, dataset=subspace_dataset,
    **{"num_interventions": len(reft_model.interventions), "position": reftargs.position , "share_weights": True},          # 该成f1+l1 梯度是0？？？？
    input_field=None, instruction_field="input", output_field="full_output",
    no_stop=True
    )

    # print(train_dataset[0])

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    if rank == 0:
        os.environ["WANDB_MODE"] = "offline"  # 如果你用 online 模式可以去掉这一行
        wandb.init(project="my_reft_2", name=f"first_train_{rank}")

    training_args = transformers.TrainingArguments(
        num_train_epochs=6, 
        output_dir=training_args.output_dir, 
        learning_rate=9e-4, 
        report_to='wandb',
        per_device_train_batch_size=2, 
        logging_steps=10,
        ddp_find_unused_parameters=False,  # 关键修改
        gradient_accumulation_steps=4,
        save_total_limit=4,
        run_name=f"first_train_{rank}" if rank == 0 else None
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
    
    trainer.train()
    trainer.save_state()

