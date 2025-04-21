import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, DataCollator
from peft import LoraConfig, get_peft_model
from train_val_dataset import CoarseGrainedDataset



from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)


def compute_loss(self, model, inputs, target_layers, alpha, beta, max_res_len=512, return_outputs=False, **kwargs):


    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    assert input_ids.shape[1] == 3

    orig_input_ids = input_ids[:, 0]
    chosen_input_ids = input_ids[:, 1]
    rejected_input_ids = input_ids[:, 2]

    orig_attention_mask = attention_mask[:, 0]
    chosen_attention_mask = attention_mask[:, 1]
    rejected_attention_mask = attention_mask[:, 2]

    min_length = max_res_len
    response_attention_mask = orig_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    module = 'past_key_values' # 'hidden_states
    # 传入的模型是增加了lora适配器的
    # 禁止使用Adapter
    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            orig_outputs = model(
                input_ids=orig_input_ids,
                attention_mask=orig_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            orig_hidden = [orig_outputs[l][:, -min_length:].detach() for l in target_layers]
            pos_outputs = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            neg_outputs = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                output_hidden_states=True
            )['hidden_states']

            # 这个direction的正确定如何评价？
            direction_hidden = [pos_outputs[l][:, -min_length:].detach() - \
                                neg_outputs[l][:, -min_length:].detach() \
                                # + beta * torch.tensor(pca_directions[l - len(pca_directions)], device=model.device, dtype=torch.float16) \
                                                for l in target_layers]
            target_hidden = torch.stack([orig_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

            del orig_outputs, pos_outputs, neg_outputs, orig_hidden, direction_hidden
            gc.collect()
            torch.cuda.empty_cache()

    model.train()

    # 加上lora的输出
    lora_outputs = model(
        input_ids=orig_input_ids,
        attention_mask=orig_attention_mask,
        output_hidden_states=True
    )['hidden_states']
    lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask

    loss_fct = torch.nn.MSELoss()
    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    return (loss, lora_hidden) if return_outputs else loss


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )

    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_args_into_dataclasses()



    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # '/data/chaojian/Mistral-7B-Instruct-v0.1',
        
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
    
    )

    lorra_target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")]
    lora_layers_to_transform = list(range(lorra_target_layers[-1] + 1))
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    torch.backends.cuda.matmul.allow_tf32 = True  # 启用Tensor Core
    torch.backends.cudnn.benchmark = True         # 启用cuDNN自动优化

    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    print(training_args.deepspeed)
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # 加载数据集
    dataset= CoarseGrainedDataset(tokenizer=tokenizer, lorra_args=lorra_args)
    
    class CustomTrainer(Trainer):
#  tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
# [rank0]:   File "/data/chaojian/anaconda3/envs/rahf/lib/python3.10/site-packages/transformers/trainer.py", line 3698, in training_step
# [rank0]:     loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
# [rank0]: TypeError: train.<locals>.CustomTrainer.compute_loss() got an unexpected keyword argument 'num_items_in_batch'
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return compute_loss(self, 
                                model, 
                                inputs, 
                                target_layers=lorra_target_layers,
                                alpha=lorra_args.lorra_alpha, 
                                beta=lorra_args.lorra_beta, 
                                max_res_len=lorra_args.max_res_len, 
                                return_outputs=return_outputs)
        
        def evaluate(self, eval_dataset=None, ignore_keys=None, sanity_check=False, **kwargs):
            pass


    
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, train_dataset=dataset, eval_dataset=dataset,
        # data_collator=CustomDataCollator()
    )
    model.config.use_cache = False
    # trainer.evaluate(eval_dataset=val_datasets, sanity_check=True)

    trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        # model.save_pretrained(training_args.output_dir) # saving adapter
        merged_model = model.merge_and_unload() # saving full model
        merged_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    
    

if __name__ == "__main__":
    train()
    



