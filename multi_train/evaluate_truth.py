import copy
import json
import os
import re
import sys
import argparse
from typing import List
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

# import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer



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

def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()

    
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
            top_p=0.85,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            positions=5,

            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        base_unit_location = inputs["input_ids"].shape[-1] - 1
        shift = inputs["attention_mask"].argmax(dim=1).unsqueeze(1)
        l = positions

        prefix = torch.arange(l).repeat(len(instructions), 1).to(device) + shift
        # print(prefix)
        suffix = torch.tensor([base_unit_location - i for i in range(l-1, -1, -1)]).repeat(len(instructions), 1).to(device)
        # print(suffix)
        base_unit_location_batched = torch.cat([prefix, suffix], dim=1)
        # print(base_unit_location_batched)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # num_beams=num_beams,
            no_repeat_ngram_size=5, 
            repetition_penalty=1.1,
            early_stopping=True,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            intervene_on_prompt=True,
            do_sample=True,
            **kwargs,
        )


        with torch.no_grad():
            _, reft_response = model.generate(
            inputs, unit_locations={"sources->base": (None, [base_unit_location_batched.tolist()]*len(model.interventions)
                                            )
            },
            # subspaces=[[[4,5,6,7]]*len(instructions)]*len(model.interventions),
            intervene_on_prompt=True, max_new_tokens=32, do_sample=False, 
            no_repeat_ngram_size=5, repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id, early_stopping=True,temperature=0.1
            
            
            )
        
        outputs = tokenizer.batch_decode(reft_response, skip_special_tokens=True)
        print(outputs)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        print(outputs)
        return outputs

    save_file = f'multi_train/experiment/{args.base_model.lstrip("../").rstrip("/")}-{args.dataset}.json'
    create_dir('multi_train/experiment')

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
    file_path = f'dataset/{args.dataset}/test.json'
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
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    # parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B'], required=True)
    # parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
    #                     required=True)
    parser.add_argument('--target_layers', type=int, nargs='+', required=True)
    parser.add_argument('--subspace_rank', type=int, default=4, required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--reft_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

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
    reft_weights = args.reft_weights
    if not reft_weights:
        raise ValueError(f'can not find lora weight, the value is: {reft_weights}')

    # load_8bit = args.load_8bit
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        ) # fix zwq

    # elif device == "mps":
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         # lora_weights,
    #         device_map={"": device},
    #         torch_dtype=torch.float16,
    #     )
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model, device_map={"": device}, low_cpu_mem_usage=True
    #     )
    #     model = PeftModel.from_pretrained(
    #         model,
    #         # lora_weights,
    #         device_map={"": device},
    #     )

    #     # unwind broken decapoda-research config
    #     # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #     # model.config.bos_token_id = 1
    #     # model.config.eos_token_id = 2



    #     if not load_8bit:
    #         model.half()  # seems to fix bugs for some users.

    #     model.eval()
    #     if torch.__version__ >= "2" and sys.platform != "win32":
    #         model = torch.compile(model)

    if args.target_layers == [-1]:
        TARGET_LAYERS = list(range(len(model.model.layers)))
    else:
        TARGET_LAYERS = args.target_layers

    reft_config = ReftConfig(representations=[
        {
            "layer": layer, "component": "block_output",
            "intervention": NodireftIntervention(
            embed_dim=model.config.hidden_size, low_rank_dimension=args.subspace_rank, add_bias=False)
        }
        for layer in TARGET_LAYERS
        ]
    )

    model = get_reft_model(model, reft_config)
    model.load_intervention(reft_weights, 
                             include_model=True)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()