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
    SubNodireftIntervention,
    ReftTrainer,
    ReftTrainerForCausalLMDistributed,
    ReftTrainerWithCElossDistributed
)
import transformers
import torch.nn as nn
import torch
import wandb
import os


# Define the internal routing function (outputs raw scores)
# class InternalRoutingFunction(nn.Module):
#     def __init__(self, input_dim, num_total_subspaces, topk=2):
#         super().__init__()
#         self.num_total_subspaces = num_total_subspaces
#         # Assuming routing is based on a pooled representation of the diff tensor
#         self.topk = topk
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 2),
#             nn.ReLU(),
#             nn.Linear(input_dim // 2, num_total_subspaces),
#         )

#     def forward(self, pooled_input_representation):
#         # pooled_input_representation shape: (batch_size, input_dim)
#         if self.gate[0].weight.dtype != pooled_input_representation.dtype:
#             self.gate = self.gate.to(dtype=pooled_input_representation.dtype, device=pooled_input_representation.device)
#         scores = self.gate(pooled_input_representation) # scores shape: (batch_size, num_total_subspaces)
#         return scores
    

# class SubNodireftIntervention(NodireftIntervention):

#     """
#       This is a NodireReFT that supports subspace interventions with internal routing (Soft Assignment).

#     """
#     def __init__(self, num_total_subspaces, subspace_rank, use_residual_gate=True, **kwargs):
#         # The low_rank_dimension in kwargs is the dimension of diff
#         super().__init__(**kwargs)
#         self.num_total_subspaces = num_total_subspaces
#         self.subspace_rank = subspace_rank
#         self.use_residual_gate = use_residual_gate

#         # Instantiate the internal routing function
#         self.routing_function = InternalRoutingFunction(
#             input_dim=self.embed_dim, # Dimension of the input to the routing function (e.g. embed_dim)
#             num_total_subspaces=num_total_subspaces # Total number of available subspaces
#         )
#         if self.use_residual_gate:
#             self.residual_gate_layer = nn.Sequential(
#                 nn.Linear(self.embed_dim * 2, self.embed_dim),
#                 nn.Sigmoid()
#             )
        
#     def freeze_except_routing_and_bias(self):
#         self.proj_layer.weight.requires_grad = False
#         self.learned_source.weight.requires_grad = False
#         self.learned_source.bias.requires_grad = False

#         for param in self.routing_function.parameters():
#             param.requires_grad = True
            
#         if self.use_residual_gate:
#             for param in self.residual_gate_layer.parameters():
#                 param.requires_grad = True

#     def forward(self, origin, last_element, base, source=None, subspaces=None):

#         # In this modified version, subspaces input is ignored.
#         # The intervention will dynamically select dimensions using the routing function (Soft Assignment).
       
#         # --- Dynamic Subspace Selection using Internal Routing (Soft Assignment) --- 
#         # Assuming routing is based on a pooled representation of the diff tensor
#         # base: shape (batch_size, sequence_length(px + lx), embed_dim)
#         # print(origin)
#         # print(origin[0].shape)

#         ######### use origin representation to get sentence embedding ########
#         # last_element = torch.tensor(last_element, device=origin[0].device, dtype=torch.long) 
#         # mask = torch.arange(origin[0].size(1), device=origin[0].device)[None, :] <= last_element[:, None]
#         # mask = mask.unsqueeze(-1).float()
#         # masked_embedding = origin[0] * mask
#         # # print(masked_embedding.shape)
#         # sentence_embeddings = masked_embedding[:, 1:, :].sum(dim=1) / last_element.unsqueeze(1)
#         # print(sentence_embeddings.shape)
#         # (batch_size, embed_dim)

#         #######################################################################


#         # print(torch.allclose(origin[0][:,1:8,], base[:,:7]))
#         # print(torch.allclose(origin[0][0, last_element[0]-6:last_element[0]+1,], base[0,7:]))
#         # print(torch.allclose(origin[0][1, last_element[1]-7:last_element[1],], base[1,7:]))
        
#         #########  use base to get sentence embedding ########
#         sentence_embeddings = torch.mean(base, dim=1) # shape: (batch_size, embed_dim)
#         ######################################################
#         # 这里不是完整的prompt表征！！！！！oh no！！！！

#         # Get raw scores from the internal routing function
#         # The routing function expects input_dim to match the pooled_diff dimension (low_rank_dimension)
#         raw_scores = self.routing_function(sentence_embeddings)
#         self.raw_scores = raw_scores
#         # print(self.raw_scores)
        
#         topk_scores, topk_indices = torch.topk(raw_scores, k=self.routing_function.topk, dim=-1)
#         # print(topk_scores)
#         topk_weights = torch.softmax(topk_scores, dim=-1, dtype=raw_scores.dtype)
#         subspace_weights = torch.zeros_like(raw_scores, dtype=raw_scores.dtype)
#         # print(subspace_weights.dtype)
#         # print(topk_weights.dtype)
#         subspace_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)


#         # print("subspace_weights:",subspace_weights)
       
#         subspace_weights_expanded = subspace_weights.unsqueeze(1).expand(-1, base.shape[1], -1)


#         diff = self.act_fn(self.learned_source(base))
#         diff = diff.view(diff.shape[0], diff.shape[1], self.num_total_subspaces, self.subspace_rank)

#         try:
#             proj_weight_reshaped = self.proj_layer.weight.view(
#                 self.num_total_subspaces, self.subspace_rank, self.embed_dim
#             )
#         except RuntimeError as e:
#             print(f"Error reshaping proj_layer.weight: {e}")
#             print(f"Expected shape for reshape: ({self.num_total_subspaces}, {self.subspace_rank}, {self.embed_dim})")
#             print(f"Actual proj_layer.weight shape: {self.proj_layer.weight.shape}")
#             raise # Re-raise the error after printing debug info

        
#         subspace_outputs = torch.einsum('bskd,kdi->bski', diff, proj_weight_reshaped)
#         weighted_sum_output = torch.einsum('bsk,bski->bsi', subspace_weights_expanded, subspace_outputs)

#          # --- Residual Gate Fusion ---
#         if self.use_residual_gate:
#             gate_input = torch.cat([base, weighted_sum_output], dim=-1)
#             gate = self.residual_gate_layer(gate_input)  # shape: (b, s, d)

#             self.latest_gate = gate.detach().cpu()
#             output = gate * weighted_sum_output + (1 - gate) * base
#         else:
#             output = base + weighted_sum_output

#         output = base + weighted_sum_output

#         return self.dropout(output.to(base.dtype))

SUBSPACE_NAMES = [
    'truthful','toxic', 'stereotype', 'safety', 'moral', 'helpful'
]
# SUBSPACE_NAMES = [
#     'safety', 'toxic', 'helpful'
# ]

def load_subdataset(subtask):
    dataset_paths = {
        'truthful': '../dataset/alignment_truthful_format',
        'helpful': '../dataset/ultra_feedback.json',
        'moral': '../dataset/alignment_moral_cls',
        'safety': '../dataset/alignment_pku_safety_format',
        'stereotype': '../dataset/alignment_stereotype_format',
        'toxic': '../dataset/alignment_toxic_format'
    }
    
    if subtask == 'helpful':
        return load_dataset('json', data_files=dataset_paths[subtask])['train']
    elif subtask == 'moral':
        return load_from_disk(dataset_paths[subtask])
    return load_from_disk(dataset_paths[subtask])['train']


def apply_template(template, input):
    pass

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
                use_residual_gate=False,
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

    # 加载router weight
    # for intervention in reft_model.interventions:
    #     layer = int(intervention.split('_')[1])
    #     router_weight_path = f"./multi_train/trainer_out_put/Llama2-7b-Nodireft_router/{layer}_classifier.pth"
    #     reft_model.interventions[intervention].routing_function.load_state_dict(torch.load(router_weight_path))

    reft_model.print_trainable_parameters()

    # 加载数据集
    if data_args.dataset_name == "combined":
        datasets = [load_subdataset(name)\
                    .map(lambda x: {'subspace_labels': torch.tensor(SUBSPACE_NAMES.index(name))})
                    for name in SUBSPACE_NAMES]
        
        if data_args.ratio:
            datasets = [dataset\
                        .shuffle(seed=42)\
                        .select(range(min(data_args.ratio, len(dataset)))) 
                    for dataset in datasets]
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
        bf16=True
    )
    
    if accelerator.is_main_process:
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=f"RouterTrain_{data_args.dataset_name}", config=vars(training_args))
    
    # 训练器
    trainer =ReftTrainerWithCElossDistributed(
        model=reft_model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator,
        routing_loss_weight=0.1
    )
    
    # 开始训练
    trainer.train()
    trainer.save_model(train_args.output_dir)

if __name__ == "__main__":
    main()
    





