import copy
import json
import os
import re
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig,
    LoreftIntervention,
    SubNodireftIntervention,
    NodireftIntervention
)
# import fire

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from multi_train.eval_common.output_naming import build_output_path
from multi_train.eval_common.output_naming import build_model_tag

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        v = v.strip().lower()
        if v in {"true", "1", "yes", "y", "t"}:
            return True
        if v in {"false", "0", "no", "n", "f"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    args.reft_weights = args.reft_weights or None
    args.lora_weights = args.lora_weights or None
    if args.reft_weights and args.lora_weights:
        raise ValueError("`reft_weights` and `lora_weights` cannot be used at the same time.")

    
    if torch.cuda.is_available():
        device = args.device
    else:
        device = "cpu"

    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass


    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            positions=args.positions,

            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        if args.reft_weights:
            base_unit_location = inputs["input_ids"].shape[-1] - 1
            shift = inputs["attention_mask"].argmax(dim=1).unsqueeze(1)
            l = positions

            prefix = torch.arange(l).repeat(len(instructions), 1).to(device) + shift
            suffix = torch.tensor([base_unit_location - i - 1 for i in range(l-1, -1, -1)]).repeat(len(instructions), 1).to(device)
            base_unit_location_batched = torch.cat([prefix, suffix], dim=1)
            base_unit_location_batched = base_unit_location_batched.unsqueeze(0)\
                .repeat(len(model.interventions),1,1)\
                # .repeat_interleave(num_beams, dim=1).tolist()

            generation_args = {
                    "base": {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]},
                    
                    "intervene_on_prompt": True,
                    "eos_token_id": tokenizer.eos_token_id,
                    'pad_token_id': tokenizer.pad_token_id,
                    "early_stopping": True,
                }
            if args.greedy_decoding:
                generation_args.update({"unit_locations": {"sources->base": (None, base_unit_location_batched.tolist())},
                                        "max_new_tokens": max_new_tokens, 
                                        "do_sample": False}
                                    )

            else:
                generation_args.update({"unit_locations": {"sources->base": (None, base_unit_location_batched\
                                                                            .repeat_interleave(num_beams, dim=1).tolist())},
                                        "max_new_tokens": max_new_tokens,
                                        "temperature": temperature,
                                        "top_p": top_p,
                                        "top_k": top_k,
                                        "num_beams": num_beams,
                                        "do_sample": True}
                                    )
                    
            with torch.no_grad():
                _, response = model.generate(**generation_args)

        # base_model 和 lora_model 共用
        else:
            with torch.no_grad():
                if args.greedy_decoding:
                    response = model.generate(
                        inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                else:
                    response = model.generate(
                        inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
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
        return outputs
    
    
    output_dir = "./multi_train/eval_truth"
    os.makedirs(output_dir, exist_ok=True)

    save_file = build_output_path(
        output_dir=output_dir,
        base_model_path=args.base_model,
        reft_weights_path=args.reft_weights,
        lora_weights_path=args.lora_weights,
        suffix=f"-{args.dataset}",
        ext=".json",
    )
    if args.summary_file:
        summary_file = args.summary_file
    else:
        save_path = Path(save_file)
        summary_file = str(save_path.with_name(f"{save_path.stem}_summary.json"))
    

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')

    accuracy = (correct / current) if current > 0 else 0.0
    summary = {
        "dataset": args.dataset,
        "result_file": save_file,
        "num_samples": current,
        "correct": correct,
        "accuracy": accuracy,
        "greedy_decoding": bool(args.greedy_decoding),
        "batch_size": args.batch_size,
        "base_model": args.base_model,
        "reft_weights": args.reft_weights,
        "lora_weights": args.lora_weights,
        "model_tag": build_model_tag(
            base_model_path=args.base_model,
            reft_weights_path=args.reft_weights,
            lora_weights_path=args.lora_weights,
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = Path(summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_path}")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    prompt_no_input_template = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""

    return prompt_no_input_template % instruction


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'./dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa", "truthfulqa_mc", "bbq"],
                        required=True)
    # parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B'], required=True)
    # parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
    #                     required=True)
    parser.add_argument('--target_layers', type=int, nargs='+')
    parser.add_argument('--subspace_rank', type=int, default=4)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--reft_weights', type=str,default=None)
    parser.add_argument('--lora_weights', type=str,default=None)
    parser.add_argument('--positions', type=int, default=5)
    parser.add_argument('--greedy_decoding', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--summary_file', type=str, default=None)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
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
    

    # load_8bit = args.load_8bit
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        ) # fix zwq
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
   
    if reft_weights:
        if args.target_layers == [-1]:
            TARGET_LAYERS = list(range(len(model.model.layers)))
        else:
            TARGET_LAYERS = args.target_layers

        # reft_config = ReftConfig(representations=[
        #     {
        #         "layer": layer, "component": "block_output",
        #         "intervention": SubNodireftIntervention(
        #             num_total_subspaces=6, subspace_rank=args.subspace_rank, topk=2, use_residual_gate=False,
        #         embed_dim=model.config.hidden_size, low_rank_dimension=args.subspace_rank*6, add_bias=False)
        #     }
        #     for layer in TARGET_LAYERS
        #     ]
        # )

        reft_config = ReftConfig(
            representations=[
                {
                    "layer": layer, 
                    "component": "block_output",
                    "intervention": LoreftIntervention(
                        embed_dim=model.config.hidden_size, 
                        low_rank_dimension=args.subspace_rank, 
                        add_bias=False
                    )
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


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    sentence_ = sentence.strip().lower()
    if dataset == 'boolq':
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa', 'truthfulqa_mc', 'bbq']:
        pred_answers = re.findall(r'answer[0-9]+', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
