from alpaca_eval import evaluate, evaluate_from_model


from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
from datasets import load_dataset
from tqdm import tqdm
import argparse
import torch
import json
import os
from peft import PeftModel
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    ReftTrainerForCausalLM, 
    ReftDataCollator,
    ReftSupervisedDataset,
    NodireftIntervention,
    LoreftIntervention
)



def load_model(args) -> tuple:
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    else:
        print(f'load base model: {base_model}')
    reft_weights = args.reft_weights
    if not reft_weights:
        print(f'can not find reft weight, the value is: {reft_weights}')
    else:
        print(f'load reft weight: {reft_weights}')
    lora_weight = args.lora_weights
    if not lora_weight:
        print(f'can not find lora weight, the value is: {lora_weight}')
    else:
        print(f'load lora weight: {lora_weight}')

    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
    
    if reft_weights:
        if args.target_layers == [-1]:
            TARGET_LAYERS = list(range(len(model.model.layers)))
        else:
            TARGET_LAYERS = args.target_layers

        reft_config = ReftConfig(representations=[
            {
                "layer": layer, "component": "block_output",
                "low_rank_dimension": args.subspace_rank,
                "intervention": LoreftIntervention(
                embed_dim=model.config.hidden_size, low_rank_dimension=args.subspace_rank, add_bias=False)
            }
            for layer in TARGET_LAYERS
            ]
        )

        model = get_reft_model(model, reft_config)
        model.load_intervention(reft_weights, 
                                include_model=True)
    elif lora_weight:
        model = PeftModel.from_pretrained(
            model,
            lora_weight,
            device_map={"": args.device},
            torch_dtype=torch.bfloat16,
        )

    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="./dataset/alpaca_data") 
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lora_weights', type=str)
    parser.add_argument('--reft_weights', type=str)
    parser.add_argument('--positions', type=int, default=5)
    parser.add_argument('--target_layers', type=int, nargs='+')
    parser.add_argument('--subspace_rank', type=int, default=4)
    parser.add_argument('--greedy_decoding',type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_seed', type=int, default=23)
    parser.add_argument('--max_tokens', type=int, default=512)

    return parser.parse_args()

def generate_prompt(instruction, input=None):
    prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

    return prompt_no_input_template % instruction

def main():

    args = parse_args()
    print(args.target_layers)

    tokenizer, model = load_model(args)
    dataset = load_dataset('json', data_files='./dataset/alpaca_eval.json')['train']
    total = len(dataset)
    
    def generate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            max_new_tokens=512,
            positions=args.positions,
            **kwargs,
        ):

        temp_generations = []
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)

        if args.reft_weights:
            base_unit_location = inputs["input_ids"].shape[-1] - 1
            shift = inputs["attention_mask"].argmax(dim=1).unsqueeze(1)
            l = positions

            prefix = torch.arange(l).repeat(len(instructions), 1).to(args.device) + shift
            suffix = torch.tensor([base_unit_location - i for i in range(l-1, -1, -1)]).repeat(len(instructions), 1).to(args.device)
        
            base_unit_location_batched = torch.cat([prefix, suffix], dim=1)
        
            base_unit_location_batched = base_unit_location_batched.unsqueeze(0)\
                .repeat(len(model.interventions),1,1)\
                # .repeat_interleave(num_beams, dim=1).tolist()

            generation_args = {
                    "base": {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    
                    "intervene_on_prompt": True,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            
            
            if args.greedy_decoding:
                generation_args.update({"unit_locations": {"sources->base": (None, base_unit_location_batched.tolist())},
                                        "max_new_tokens": args.max_tokens, 
                                        "do_sample": False}
                                    )

            else:
                
                generation_args.update({"unit_locations": {"sources->base": (None, base_unit_location_batched.tolist())},
                                        "max_new_tokens": args.max_tokens, 
                                        "no_repeat_ngram_size": 5,
                                        "repetition_penalty": 1.1,
                                        "do_sample": True,
                                        "temperature":0.6,
                                        }
                                    )
                
            with torch.no_grad():
                # print(generation_args)
                _, response = model.generate(**generation_args)

        else:
            with torch.no_grad():
                response = model.generate(
                    inputs["input_ids"].to(args.device),
                    attention_mask=inputs["attention_mask"].to(args.device),
                    generation_config=GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                    )
                )
            

        outputs = tokenizer.batch_decode(response, skip_special_tokens=True)
        print(outputs)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        print(outputs)

        for instr, output in zip(instructions, outputs):
            temp_generations.append({
                "instruction": instr,
                "output": output
            })
        return temp_generations
    
    decoding = "greedy" if args.greedy_decoding else "no_greedy"

    if not os.path.exists('./multi_train/eval_helpful'):
        os.mkdir('./multi_train/eval_helpful')

    
    base_dir = f'multi_train/eval_helpful/{args.base_model.lstrip("../").rstrip("/")}'
    if args.reft_weights:
        base_dir += f'_{"-".join(args.reft_weights.split("/")[3:]).strip(" ")}-{decoding}_generations.json'
    elif args.lora_weights:
        base_dir += f'_{"-".join(args.lora_weights.split("/")[2:]).strip(" ")}-{decoding}_generations.json'
    output_path = base_dir

    # output_path = f'multi_train/helpfulness/{args.base_model.lstrip("../").rstrip("/")}{"-".join(args.reft_weights.split("/")[2:]).strip(" ").lstrip("-")}{decoding}-generations.json'
    
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            generations = json.load(f)
    else:
        generations = []
        

    start_idx = len(generations)

    # === 4. 按 batch 执行评估 ===
    for i in tqdm(range(start_idx, total, args.batch_size), desc="Evaluating"):
        batch = dataset[i: i + args.batch_size]
        
        # 获取指令和输入（如果有）
        instructions = batch["instruction"]
        inputs = batch.get("input", [None] * len(instructions))  # 若无 input 则填 None

        # 生成结果
        results = generate(instructions)

        generations.extend(results)

        # === 每 batch 保存一次结果 ===
        with open(output_path, "w") as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)


    
    


if __name__ == "__main__":
    main()



