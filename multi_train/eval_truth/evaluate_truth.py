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
            "selected_mask_sum": None,
            "topk_mask_sum": None,
            "rejected_conflict_mask_sum": None,
            "conflict_score_sum": None,
            "conflict_score_sq_sum": None,
            "score_source": None,
            "score_stats_source": None,
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
        selected_mask = getattr(intervention, "latest_selected_mask", None)
        topk_mask = getattr(intervention, "latest_topk_mask", None)
        rejected_conflict_mask = getattr(intervention, "latest_rejected_conflict_mask", None)
        conflict_score = getattr(intervention, "latest_conflict_score", None)
        score_source = getattr(intervention, "latest_score_source", None)
        score_stats_source = getattr(intervention, "latest_score_stats_source", None)
        if alpha is None or delta_norm is None or intervention_norm is None:
            continue

        alpha = alpha.float()
        normalized_score = normalized_score.float() if normalized_score is not None else None
        delta_norm = delta_norm.float()
        intervention_norm = intervention_norm.float()
        selected_mask = selected_mask.float() if selected_mask is not None else None
        topk_mask = topk_mask.float() if topk_mask is not None else None
        rejected_conflict_mask = (
            rejected_conflict_mask.float() if rejected_conflict_mask is not None else None
        )
        conflict_score = conflict_score.float() if conflict_score is not None else None

        layer_stats = accumulator["layers"][layer_name]
        alpha_sum = alpha.sum(dim=(0, 1))
        alpha_sq_sum = (alpha ** 2).sum(dim=(0, 1))
        normalized_score_sum = normalized_score.sum(dim=(0, 1)) if normalized_score is not None else None
        normalized_score_sq_sum = (normalized_score ** 2).sum(dim=(0, 1)) if normalized_score is not None else None
        delta_sum = delta_norm.sum(dim=(0, 1))
        delta_sq_sum = (delta_norm ** 2).sum(dim=(0, 1))
        intv_sum = intervention_norm.sum(dim=(0, 1))
        intv_sq_sum = (intervention_norm ** 2).sum(dim=(0, 1))
        selected_sum = selected_mask.sum(dim=(0, 1)) if selected_mask is not None else None
        topk_sum = topk_mask.sum(dim=(0, 1)) if topk_mask is not None else None
        rejected_sum = (
            rejected_conflict_mask.sum(dim=(0, 1)) if rejected_conflict_mask is not None else None
        )
        conflict_sum = conflict_score.sum() if conflict_score is not None else None
        conflict_sq_sum = ((conflict_score ** 2).sum()) if conflict_score is not None else None
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
            layer_stats["selected_mask_sum"] = selected_sum
            layer_stats["topk_mask_sum"] = topk_sum
            layer_stats["rejected_conflict_mask_sum"] = rejected_sum
            layer_stats["conflict_score_sum"] = conflict_sum
            layer_stats["conflict_score_sq_sum"] = conflict_sq_sum
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
            if selected_sum is not None:
                if layer_stats["selected_mask_sum"] is None:
                    layer_stats["selected_mask_sum"] = selected_sum
                else:
                    layer_stats["selected_mask_sum"] += selected_sum
            if topk_sum is not None:
                if layer_stats["topk_mask_sum"] is None:
                    layer_stats["topk_mask_sum"] = topk_sum
                else:
                    layer_stats["topk_mask_sum"] += topk_sum
            if rejected_sum is not None:
                if layer_stats["rejected_conflict_mask_sum"] is None:
                    layer_stats["rejected_conflict_mask_sum"] = rejected_sum
                else:
                    layer_stats["rejected_conflict_mask_sum"] += rejected_sum
            if conflict_sum is not None:
                if layer_stats["conflict_score_sum"] is None:
                    layer_stats["conflict_score_sum"] = conflict_sum
                    layer_stats["conflict_score_sq_sum"] = conflict_sq_sum
                else:
                    layer_stats["conflict_score_sum"] += conflict_sum
                    layer_stats["conflict_score_sq_sum"] += conflict_sq_sum

        if score_source is not None:
            if layer_stats["score_source"] is None:
                layer_stats["score_source"] = score_source
            elif layer_stats["score_source"] != score_source:
                layer_stats["score_source"] = "mixed"
        if score_stats_source is not None:
            if layer_stats["score_stats_source"] is None:
                layer_stats["score_stats_source"] = score_stats_source
            elif layer_stats["score_stats_source"] != score_stats_source:
                layer_stats["score_stats_source"] = "mixed"

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
        selected_mean = stats["selected_mask_sum"] / count if stats["selected_mask_sum"] is not None else None
        topk_mean = stats["topk_mask_sum"] / count if stats["topk_mask_sum"] is not None else None
        rejected_mean = (
            stats["rejected_conflict_mask_sum"] / count
            if stats["rejected_conflict_mask_sum"] is not None
            else None
        )
        conflict_score_mean = None
        conflict_score_std = None
        if stats["conflict_score_sum"] is not None:
            conflict_score_mean = stats["conflict_score_sum"] / count
            conflict_score_var = (stats["conflict_score_sq_sum"] / count) - conflict_score_mean * conflict_score_mean
            conflict_score_std = max(float(conflict_score_var), 0.0) ** 0.5

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
            if selected_mean is not None:
                entry["selected_rate"] = float(selected_mean[idx].item())
            if topk_mean is not None:
                entry["topk_rate"] = float(topk_mean[idx].item())
            if rejected_mean is not None:
                entry["rejected_conflict_rate"] = float(rejected_mean[idx].item())
            per_specialist.append(entry)

        finalized["layers"][layer_name] = {
            "count": count,
            "score_source": stats["score_source"],
            "score_stats_source": stats["score_stats_source"],
            "conflict_score_mean": None if conflict_score_mean is None else float(conflict_score_mean),
            "conflict_score_std": conflict_score_std,
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
            score_source=args.score_source,
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
        "score_source": args.score_source,
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
            "score_source": args.score_source,
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
    parser.add_argument('--score_source', choices=["delta_norm", "intervention_norm"], default=None)
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
            score_source=args.score_source,
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


def extract_labeled_choice(sentence: str, label_prefix: str, max_choices: int) -> str:
    sentence_ = sentence.strip().lower()
    if not sentence_:
        return ""

    if "### response:" in sentence_:
        sentence_ = sentence_.split("### response:")[-1].strip()

    label_pattern = rf"{re.escape(label_prefix)}[0-9]+"
    lines = [line.strip() for line in re.split(r"[\r\n]+", sentence_) if line.strip()]

    # Base models sometimes copy "Answer format: answer1/answer2/..." as the
    # first line. Treat only that pure slash-separated format line as non-answer.
    slash_format_pattern = rf"{label_pattern}(?:/{label_pattern})+"
    if lines and re.fullmatch(slash_format_pattern, lines[0]):
        lines = lines[1:]
        sentence_ = "\n".join(lines).strip()

    if not sentence_:
        return ""

    explicit_patterns = [
        rf"(?:the\s+)?correct\s+{re.escape(label_prefix)}\s*(?:is|:)\s*({label_pattern})\b",
        rf"(?:the\s+)?correct\s+answer\s*(?:is|:)\s*({label_pattern})\b",
        rf"(?:the\s+)?correct\s+response\s*(?:is|:)\s*({label_pattern})\b",
        rf"final\s+answer\s*(?:is|:)?\s*({label_pattern})\b",
        rf"(?:my\s+)?answer\s*(?:is|:)\s*({label_pattern})\b",
        rf"(?:response|prediction)\s*:\s*({label_pattern})\b",
    ]
    for pattern in explicit_patterns:
        pred_answers = re.findall(pattern, sentence_)
        if pred_answers:
            return pred_answers[-1]

    for line in lines:
        match = re.fullmatch(rf"({label_pattern})[\s\.\!\?,;:'\"\)\]]*", line)
        if match:
            return match.group(1)

    valid_labels = {f"{label_prefix}{idx}" for idx in range(1, max_choices + 1)}
    pred_answers = [
        match.group(0)
        for match in re.finditer(label_pattern, sentence_)
        if match.group(0) in valid_labels
    ]
    if not pred_answers:
        return ""
    return pred_answers[0]


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
        return extract_labeled_choice(sentence, label_prefix="answer", max_choices=16)
    elif dataset == 'hellaswag':
        return extract_labeled_choice(sentence, label_prefix="ending", max_choices=4)
    elif dataset == 'winogrande':
        return extract_labeled_choice(sentence, label_prefix="option", max_choices=2)


if __name__ == "__main__":
    main()
