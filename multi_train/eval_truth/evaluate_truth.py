import copy
import json
import os
import re
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List
try:
    from pyreft import (
        TaskType,
        get_reft_model,
        ReftConfig,
        LoreftIntervention,
        SubNodireftIntervention,
        NodireftIntervention
    )
except ImportError:
    from pyreft.pyreft import (
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
from multi_train.eval_common.composable_loreft import (
    build_composable_model_tag,
    load_composed_reft_model,
    normalize_specialist_label,
)

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


def _unwrap_intervention(intervention_value):
    if isinstance(intervention_value, (list, tuple)):
        return intervention_value[0]
    return intervention_value


def _init_debug_accumulator(model, specialist_labels):
    if model is None or not hasattr(model, "interventions"):
        return None

    accumulator = {
        "specialist_labels": specialist_labels,
        "layers": {},
    }

    for layer_key, intervention_value in model.interventions.items():
        intervention = _unwrap_intervention(intervention_value)
        if not hasattr(intervention, "latest_alpha"):
            continue
        accumulator["layers"][str(layer_key)] = {
            "alpha_sum": None,
            "alpha_sq_sum": None,
            "normalized_score_sum": None,
            "normalized_score_sq_sum": None,
            "delta_norm_sum": None,
            "delta_norm_sq_sum": None,
            "intervention_norm_sum": None,
            "intervention_norm_sq_sum": None,
            "count": 0,
        }

    return accumulator


def _update_debug_accumulator(accumulator, model):
    if accumulator is None:
        return

    for layer_key, intervention_value in model.interventions.items():
        layer_name = str(layer_key)
        if layer_name not in accumulator["layers"]:
            continue

        intervention = _unwrap_intervention(intervention_value)
        alpha = getattr(intervention, "latest_alpha", None)
        normalized_score = getattr(intervention, "latest_normalized_score", None)
        delta_norm = getattr(intervention, "latest_delta_norm", None)
        intervention_norm = getattr(intervention, "latest_intervention_norm", None)
        if alpha is None or delta_norm is None or intervention_norm is None:
            continue

        alpha = alpha.float()
        normalized_score = normalized_score.float() if normalized_score is not None else None
        delta_norm = delta_norm.float()
        intervention_norm = intervention_norm.float()

        layer_stats = accumulator["layers"][layer_name]
        alpha_sum = alpha.sum(dim=(0, 1))
        alpha_sq_sum = (alpha ** 2).sum(dim=(0, 1))
        normalized_score_sum = normalized_score.sum(dim=(0, 1)) if normalized_score is not None else None
        normalized_score_sq_sum = (normalized_score ** 2).sum(dim=(0, 1)) if normalized_score is not None else None
        delta_sum = delta_norm.sum(dim=(0, 1))
        delta_sq_sum = (delta_norm ** 2).sum(dim=(0, 1))
        intv_sum = intervention_norm.sum(dim=(0, 1))
        intv_sq_sum = (intervention_norm ** 2).sum(dim=(0, 1))
        count = int(alpha.shape[0] * alpha.shape[1])

        if layer_stats["alpha_sum"] is None:
            layer_stats["alpha_sum"] = alpha_sum
            layer_stats["alpha_sq_sum"] = alpha_sq_sum
            layer_stats["normalized_score_sum"] = normalized_score_sum
            layer_stats["normalized_score_sq_sum"] = normalized_score_sq_sum
            layer_stats["delta_norm_sum"] = delta_sum
            layer_stats["delta_norm_sq_sum"] = delta_sq_sum
            layer_stats["intervention_norm_sum"] = intv_sum
            layer_stats["intervention_norm_sq_sum"] = intv_sq_sum
        else:
            layer_stats["alpha_sum"] += alpha_sum
            layer_stats["alpha_sq_sum"] += alpha_sq_sum
            if normalized_score_sum is not None:
                if layer_stats["normalized_score_sum"] is None:
                    layer_stats["normalized_score_sum"] = normalized_score_sum
                    layer_stats["normalized_score_sq_sum"] = normalized_score_sq_sum
                else:
                    layer_stats["normalized_score_sum"] += normalized_score_sum
                    layer_stats["normalized_score_sq_sum"] += normalized_score_sq_sum
            layer_stats["delta_norm_sum"] += delta_sum
            layer_stats["delta_norm_sq_sum"] += delta_sq_sum
            layer_stats["intervention_norm_sum"] += intv_sum
            layer_stats["intervention_norm_sq_sum"] += intv_sq_sum

        layer_stats["count"] += count


def _finalize_debug_accumulator(accumulator):
    if accumulator is None:
        return None

    specialist_labels = accumulator.get("specialist_labels", [])
    finalized = {
        "specialist_labels": specialist_labels,
        "layers": {},
    }

    for layer_name, stats in accumulator["layers"].items():
        count = int(stats["count"])
        if count <= 0 or stats["alpha_sum"] is None:
            continue

        alpha_mean = stats["alpha_sum"] / count
        alpha_var = (stats["alpha_sq_sum"] / count) - alpha_mean.pow(2)
        alpha_std = torch.sqrt(torch.clamp(alpha_var, min=0.0))

        delta_mean = stats["delta_norm_sum"] / count
        delta_var = (stats["delta_norm_sq_sum"] / count) - delta_mean.pow(2)
        delta_std = torch.sqrt(torch.clamp(delta_var, min=0.0))

        intv_mean = stats["intervention_norm_sum"] / count
        intv_var = (stats["intervention_norm_sq_sum"] / count) - intv_mean.pow(2)
        intv_std = torch.sqrt(torch.clamp(intv_var, min=0.0))
        normalized_score_mean = None
        normalized_score_std = None
        if stats["normalized_score_sum"] is not None:
            normalized_score_mean = stats["normalized_score_sum"] / count
            normalized_score_var = (stats["normalized_score_sq_sum"] / count) - normalized_score_mean.pow(2)
            normalized_score_std = torch.sqrt(torch.clamp(normalized_score_var, min=0.0))

        per_specialist = []
        for idx in range(alpha_mean.shape[0]):
            label = specialist_labels[idx] if idx < len(specialist_labels) else f"specialist_{idx}"
            entry = {
                "index": idx,
                "label": label,
                "alpha_mean": float(alpha_mean[idx].item()),
                "alpha_std": float(alpha_std[idx].item()),
                "delta_norm_mean": float(delta_mean[idx].item()),
                "delta_norm_std": float(delta_std[idx].item()),
                "intervention_norm_mean": float(intv_mean[idx].item()),
                "intervention_norm_std": float(intv_std[idx].item()),
            }
            if normalized_score_mean is not None:
                entry["normalized_score_mean"] = float(normalized_score_mean[idx].item())
                entry["normalized_score_std"] = float(normalized_score_std[idx].item())
            per_specialist.append(entry)

        finalized["layers"][layer_name] = {
            "count": count,
            "per_specialist": per_specialist,
        }

    return finalized


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    args.reft_weights = args.reft_weights or None
    args.lora_weights = args.lora_weights or None
    if args.reft_specialists:
        args.reft_specialists = [path for path in args.reft_specialists if str(path).strip()]
        if len(args.reft_specialists) == 0:
            args.reft_specialists = None
    if args.reft_weights and args.lora_weights:
        raise ValueError("`reft_weights` and `lora_weights` cannot be used at the same time.")
    if args.reft_specialists and args.reft_weights:
        raise ValueError("`reft_specialists` and `reft_weights` cannot be used at the same time.")
    if args.reft_specialists and args.lora_weights:
        raise ValueError("`reft_specialists` and `lora_weights` cannot be used at the same time.")
    use_reft = bool(args.reft_weights or args.reft_specialists)

    
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

        if use_reft:
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

    if args.reft_specialists:
        specialist_labels = [normalize_specialist_label(path) for path in args.reft_specialists]
        model_tag = build_composable_model_tag(
            base_model_path=args.base_model,
            specialist_dirs=args.reft_specialists,
            compose_domain=args.compose_domain,
            composition_method=args.composition_method,
            composition_temperature=args.composition_temperature,
            composition_topk=args.composition_topk,
            compat_threshold=args.compat_threshold,
            single_index=args.single_index,
            shared_basis_type=args.shared_basis_type,
            shared_basis_rank=args.shared_basis_rank,
            transport_type=args.transport_type,
            score_normalizer=args.score_normalizer,
            score_stats_path=args.score_stats_path,
            score_clip=args.score_clip,
        )
        save_file = str(Path(output_dir) / f"{model_tag}-{args.dataset}.json")
    else:
        specialist_labels = []
        model_tag = build_model_tag(
            base_model_path=args.base_model,
            reft_weights_path=args.reft_weights,
            lora_weights_path=args.lora_weights,
        )
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
    debug_accumulator = _init_debug_accumulator(model, specialist_labels)
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)
        _update_debug_accumulator(debug_accumulator, model)

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

    debug_summary = _finalize_debug_accumulator(debug_accumulator)
    debug_summary_file = None
    if debug_summary is not None:
        save_path = Path(save_file)
        debug_summary_file = str(save_path.with_name(f"{save_path.stem}_debug_summary.json"))
        with open(debug_summary_file, "w") as f:
            json.dump(debug_summary, f, indent=4)
        print(f"Debug summary saved to: {debug_summary_file}")

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
        "reft_specialists": args.reft_specialists,
        "compose_domain": args.compose_domain,
        "composition_method": args.composition_method,
        "composition_temperature": args.composition_temperature,
        "composition_topk": args.composition_topk,
        "compat_threshold": args.compat_threshold,
        "single_index": args.single_index,
        "shared_basis_type": args.shared_basis_type,
        "shared_basis_rank": args.shared_basis_rank,
        "transport_type": args.transport_type,
        "score_normalizer": args.score_normalizer,
        "score_stats_path": args.score_stats_path,
        "score_eps": args.score_eps,
        "score_clip": args.score_clip,
        "composable_config": {
            "specialist_labels": specialist_labels,
            "compose_domain": args.compose_domain,
            "composition_method": args.composition_method,
            "composition_temperature": args.composition_temperature,
            "composition_topk": args.composition_topk,
            "compat_threshold": args.compat_threshold,
            "single_index": args.single_index,
            "shared_basis_type": args.shared_basis_type,
            "shared_basis_rank": args.shared_basis_rank,
            "transport_type": args.transport_type,
            "score_normalizer": args.score_normalizer,
            "score_stats_path": args.score_stats_path,
            "score_eps": args.score_eps,
            "score_clip": args.score_clip,
        } if args.reft_specialists else None,
        "debug_summary_file": debug_summary_file,
        "model_tag": model_tag,
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
    parser.add_argument('--target_layers', type=int, nargs='+', default=[-1])
    parser.add_argument('--subspace_rank', type=int, default=4)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--reft_weights', type=str,default=None)
    parser.add_argument('--reft_specialists', type=str, nargs='+', default=None)
    parser.add_argument('--lora_weights', type=str,default=None)
    parser.add_argument('--compose_domain', choices=["output", "projected_output", "shared_latent"], default="output")
    parser.add_argument('--composition_method', choices=["single", "equal", "residual_softmax", "residual_scaled_softmax", "residual_logz_softmax", "intervention_softmax", "topk_residual", "compat_filtered_topk"], default="equal")
    parser.add_argument('--composition_temperature', type=float, default=1.0)
    parser.add_argument('--composition_topk', type=int, default=None)
    parser.add_argument('--compat_threshold', type=float, default=0.0)
    parser.add_argument('--single_index', type=int, default=None)
    parser.add_argument('--shared_basis_type', choices=["orth_mean", "svd_union"], default=None)
    parser.add_argument('--shared_basis_rank', type=int, default=None)
    parser.add_argument('--transport_type', choices=["identity", "overlap"], default=None)
    parser.add_argument('--score_normalizer', choices=["none", "mean_ratio", "log_zscore"], default="none")
    parser.add_argument('--score_stats_path', type=str, default=None)
    parser.add_argument('--score_eps', type=float, default=1e-6)
    parser.add_argument('--score_clip', type=float, default=None)
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
    if args.reft_specialists:
        print(f'load reft specialists: {args.reft_specialists}')
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
   
    if args.reft_specialists:
        if args.target_layers == [-1]:
            target_layers = list(range(len(model.model.layers)))
        else:
            target_layers = args.target_layers

        model = load_composed_reft_model(
            model=model,
            specialist_dirs=args.reft_specialists,
            target_layers=target_layers,
            compose_domain=args.compose_domain,
            composition_method=args.composition_method,
            composition_temperature=args.composition_temperature,
            composition_topk=args.composition_topk,
            compat_threshold=args.compat_threshold,
            single_index=args.single_index,
            shared_basis_type=args.shared_basis_type,
            shared_basis_rank=args.shared_basis_rank,
            transport_type=args.transport_type,
            score_normalizer=args.score_normalizer,
            score_stats_path=args.score_stats_path,
            score_eps=args.score_eps,
            score_clip=args.score_clip,
        )
    elif reft_weights:
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
