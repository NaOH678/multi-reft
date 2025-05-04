from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, AutoConfig, set_seed
from router_train_args import RouterDataArguments, RouterTrainArguments, RouterModelArguments
from datasets import load_dataset, concatenate_datasets, load_from_disk
from accelerate import Accelerator
from pyreft import (
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    NodireftIntervention,
    ReftTrainer,
    ReftTrainerForCausalLMDistributed
)
import transformers
import torch.nn as nn
import torch
import wandb
import os


# Define the internal routing function (outputs raw scores)
class InternalRoutingFunction(nn.Module):
    def __init__(self, input_dim, num_total_subspaces):
        super().__init__()
        self.num_total_subspaces = num_total_subspaces
        # Assuming routing is based on a pooled representation of the diff tensor
        self.gate = nn.Linear(input_dim, num_total_subspaces)

    def forward(self, pooled_input_representation):
        # pooled_input_representation shape: (batch_size, input_dim)
        scores = self.gate(pooled_input_representation) # scores shape: (batch_size, num_total_subspaces)
        return scores # Return raw scores
    

class SubNodireftIntervention(NodireftIntervention):

    """
      This is a NodireReFT that supports subspace interventions with internal routing (Soft Assignment).

    """
    def __init__(self, num_total_subspaces, subspace_rank, **kwargs):
        # The low_rank_dimension in kwargs is the dimension of diff
        super().__init__(**kwargs)
        self.num_total_subspaces = num_total_subspaces
        self.subspace_rank = subspace_rank
        # Instantiate the internal routing function
        self.routing_function = InternalRoutingFunction(
            input_dim=self.embed_dim, # Dimension of the input to the routing function (e.g. embed_dim)
            num_total_subspaces=num_total_subspaces # Total number of available subspaces
        )
        
    def freeze_except_routing_and_bias(self):
       self.proj_layer.weight.requires_grad = False
       self.learned_source.weight.requires_grad = False
       self.learned_source.bias.requires_grad = True

       for param in self.routing_function.parameters():
          param.requires_grad = True

    def forward(self, base, source=None, subspaces=None):

        # In this modified version, subspaces input is ignored.
        # The intervention will dynamically select dimensions using the routing function (Soft Assignment).
       
        # --- Dynamic Subspace Selection using Internal Routing (Soft Assignment) --- 
        # Assuming routing is based on a pooled representation of the diff tensor
        # base: shape (batch_size, sequence_length, embed_dim)
        pooled_diff = torch.mean(base, dim=1) # shape: (batch_size, embed_dim)

        # Get raw scores from the internal routing function
        # The routing function expects input_dim to match the pooled_diff dimension (low_rank_dimension)
        raw_scores = self.routing_function(pooled_diff)
        # raw_scores shape: (batch_size, num_total_subspaces)

        # Apply Softmax to get weights for each subspace
        subspace_weights = torch.softmax(raw_scores, dim=-1)
        # subspace_weights shape: (batch_size, num_total_subspaces)

        # Expand subspace_weights to match diff's sequence length for weighting
        # subspace_weights_expanded shape: (batch_size, sequence_length, num_total_subspaces)
        subspace_weights_expanded = subspace_weights.unsqueeze(1).expand(-1, base.shape[1], -1)


        diff = self.act_fn(self.learned_source(base))
        # diff shape: (batch_size, sequence_length, num_subspaces * low_rank_dimension)
        # print("diff shape:", diff.shape)
        diff = diff.view(diff.shape[0], diff.shape[1], self.num_total_subspaces, self.subspace_rank)

        
        try:
            proj_weight_reshaped = self.proj_layer.weight.view(
                self.num_total_subspaces, self.subspace_rank, self.embed_dim
            )
        except RuntimeError as e:
            print(f"Error reshaping proj_layer.weight: {e}")
            print(f"Expected shape for reshape: ({self.num_total_subspaces}, {self.subspace_rank}, {self.embed_dim})")
            print(f"Actual proj_layer.weight shape: {self.proj_layer.weight.shape}")
            raise # Re-raise the error after printing debug info

        
        subspace_outputs = torch.einsum('bskd,kdi->bski', diff, proj_weight_reshaped)

        # Weight and sum the subspace outputs
        # weighted_sum_output shape: (batch_size, sequence_length, embed_dim)
        # This is einsum('bsk,bski->bsi', subspace_weights_expanded, subspace_outputs)
        weighted_sum_output = torch.einsum('bsk,bski->bsi', subspace_weights_expanded, subspace_outputs)

        output = base + weighted_sum_output

        return self.dropout(output.to(base.dtype))

# SUBSPACE_NAMES = [
#     'moral', 'truthful', 'safety', 'toxicity', 'stereotype', 'helpful'
# ]
SUBSPACE_NAMES = [
    'safety', 'toxic', 'helpful'
]

def load_subdataset(subtask):
    dataset_paths = {
        'truthful': '/data/chaojian/Multi-alignment/dataset/alignment_truthful_format',
        'helpful': '/data/chaojian/Multi-alignment/dataset/ultra_feedback.json',
        'moral': '/data/chaojian/Multi-alignment/dataset/alignment_moral_format',
        'safety': '/data/chaojian/Multi-alignment/dataset/alignment_pku_safety_format',
        'stereotype': '/data/chaojian/Multi-alignment/dataset/alignment_stereotype_format',
        'toxic': '/data/chaojian/Multi-alignment/dataset/alignment_toxic_format'
    }
    
    if subtask == 'helpful':
        return load_dataset('json', data_files=dataset_paths[subtask])['train']
    return load_from_disk(dataset_paths[subtask])['train']


def main():
    accelerator = Accelerator()

    parser = HfArgumentParser((RouterTrainArguments, RouterDataArguments, RouterModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)

    subspace_rank = train_args.subspace_rank

    model_name_or_path = train_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16
    )

    # get tokenizer
    model_max_length = train_args.max_length
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=model_max_length, 
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token


    layers = []
    reft_weigt_path = model_args.reft_weight_path
    layers_weights = os.listdir(reft_weigt_path)
    for layer_weight in layers_weights:
        print(layer_weight)
        # intkey_layer_9_comp_block_output_unit_pos_nunit_1#0.bin
        if 'intkey' not in layer_weight:
            continue
        layer = int(layer_weight.split('_')[2])
        layers.append(layer)

    
    reft_config = ReftConfig(representations=[
        {
            "layer": layer, "component": "block_output",
            "low_rank_dimension": subspace_rank*len(SUBSPACE_NAMES),
            "intervention": SubNodireftIntervention(
                num_total_subspaces=len(SUBSPACE_NAMES), 
                subspace_rank=subspace_rank,
                embed_dim=model.config.hidden_size, 
                low_rank_dimension=subspace_rank * len(SUBSPACE_NAMES), 
                dropout=train_args.dropout,
                add_bias=False)
        }
        for layer in layers]
    )

    reft_model = get_reft_model(model, reft_config)
    reft_model.load_intervention(reft_weigt_path, 
                                include_model=True)
    
    # 冻结参数
    for invention in reft_model.interventions:
        reft_model.interventions[invention].freeze_except_routing_and_bias()

    reft_model.print_trainable_parameters()

    # 加载数据集
    if data_args.dataset_name == "combined":
        datasets = [load_subdataset(name) for name in SUBSPACE_NAMES]
        dataset = concatenate_datasets(datasets)
    else:
        dataset = load_subdataset(data_args.dataset_name)
    
    if data_args.max_examples:
        max_examples = data_args.max_examples
        # dataset = dataset.select(range(min(data_args.max_examples, len(dataset))))
    else:
        max_examples = len(dataset)

    print(dataset[0])
    
    train_dataset = ReftSupervisedDataset(
        "SubNodireloreft", None, tokenizer, dataset=dataset,
        **{"num_interventions": len(reft_model.interventions), "position": train_args.position , "share_weights": True},          # 该成f1+l1 梯度是0？？？？
        input_field='input', instruction_field="instruction", output_field="output", 
        seed=train_args.seed, max_n_example=min(max_examples, len(dataset)),
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

    training_args = transformers.TrainingArguments(
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
    )
    
    if accelerator.is_main_process:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=f"RouterTrain_{data_args.dataset_name}", config=vars(training_args))
    
    # 训练器
    trainer =ReftTrainerForCausalLMDistributed(
        model=reft_model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator
    )
    
    # 开始训练
    trainer.train()
    trainer.save_model(train_args.output_dir)

if __name__ == "__main__":
    main()
    





