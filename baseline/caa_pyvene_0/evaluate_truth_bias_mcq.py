#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.pyvene_additive_common import apply_pyvene_additive_steering, build_pyvene_model_tag
from baseline.truth_label_utils import extract_choice_label, parse_labeled_choices


PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
%s

### Response:
"""


def generate_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE % instruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["truthfulqa_mc", "truthfulqa_mc_dev", "bbq"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results_json", default=None)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--greedy_decoding", type=int, default=1)
    parser.add_argument("--positions", type=int, default=7)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--subspace_rank", type=int, default=8)

    parser.add_argument("--pyvene_method_tag", required=True)
    parser.add_argument("--pyvene_component", required=True)
    parser.add_argument("--pyvene_vector_dir", type=str, default=None)
    parser.add_argument("--pyvene_vector_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--pyvene_layers", type=int, nargs="+", default=None)
    parser.add_argument("--pyvene_alpha", type=float, default=1.0)
    parser.add_argument("--pyvene_composition", choices=["single", "sum", "mean", "norm_mean", "weighted_sum"], default="single")
    parser.add_argument("--pyvene_weights", type=float, nargs="+", default=None)
    parser.add_argument("--pyvene_intervene_on_prompt", type=int, default=0)
    parser.add_argument("--pyvene_base_unit_location", type=int, default=None)
    parser.add_argument(
        "--evaluation_mode",
        choices=["auto", "constrained_label_generation", "free_generation", "candidate_scoring"],
        default="auto",
    )
    parser.add_argument("--free_generation_max_new_tokens", type=int, default=64)
    parser.add_argument("--candidate_score_reduction", choices=["sum", "mean"], default="mean")
    return parser.parse_args()


def load_dataset(dataset_name: str, max_samples: int | None) -> list[dict]:
    data_path = Path("dataset") / dataset_name / "test.json"
    with data_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if max_samples is not None:
        rows = rows[: max_samples]
    return rows


def label_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"answer(\d+)", str(label).strip().lower())
    if match:
        return int(match.group(1)), str(label)
    return 10**9, str(label)


def build_choice_labels(instruction: str) -> list[str]:
    option_map = parse_labeled_choices(instruction)
    labels = sorted(option_map.keys(), key=label_sort_key)
    if not labels:
        raise ValueError("Failed to parse labeled choices from instruction.")
    return labels


def resolve_runtime_device(requested_device: str) -> str:
    if torch.cuda.is_available():
        return requested_device
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_tokenizer_and_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=resolve_runtime_device(args.device),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
            raise ValueError("Tokenizer has no pad/unk/eos/bos token available.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    model, pyvene_config = apply_pyvene_additive_steering(
        model,
        vector_dir=args.pyvene_vector_dir,
        vector_dirs=args.pyvene_vector_dirs,
        layers=args.pyvene_layers,
        alpha=args.pyvene_alpha,
        component=args.pyvene_component,
        composition=args.pyvene_composition,
        weights=args.pyvene_weights,
        method_tag=args.pyvene_method_tag,
        intervene_on_prompt=bool(args.pyvene_intervene_on_prompt),
        base_unit_location=args.pyvene_base_unit_location,
    )
    model.eval()
    return tokenizer, model, pyvene_config


def build_continuation_token_ids(
    tokenizer,
    prompt: str,
    labels: Sequence[str],
) -> list[list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids: list[list[int]] = []
    for label in labels:
        full_ids = tokenizer(prompt + label, add_special_tokens=False)["input_ids"]
        continuation_ids = full_ids[len(prompt_ids) :]
        if not continuation_ids:
            continuation_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not continuation_ids:
            raise ValueError(f"Failed to tokenize label continuation: {label}")
        token_ids.append(continuation_ids)
    return token_ids


def build_prefix_allowed_tokens_fn(
    allowed_sequences_per_sample: list[list[list[int]]],
    initial_input_len: int,
    eos_token_id: int,
):
    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
        generated = input_ids[initial_input_len:].tolist()
        allowed_next: set[int] = set()
        completed = False

        for sequence in allowed_sequences_per_sample[batch_id]:
            if generated == sequence:
                completed = True
                continue
            if len(generated) < len(sequence) and sequence[: len(generated)] == generated:
                allowed_next.add(sequence[len(generated)])

        if allowed_next:
            return sorted(allowed_next)
        if completed:
            return [eos_token_id]
        return [eos_token_id]

    return prefix_allowed_tokens_fn


def decode_prediction(text: str) -> str:
    parsed = extract_choice_label(str(text).strip())
    if parsed:
        return parsed

    matches = re.findall(r"answer\d+", str(text).strip().lower())
    if not matches:
        return ""
    return matches[-1]


def evaluate_batch_constrained(tokenizer, model, batch_rows: list[dict]) -> list[dict]:
    prompts = [generate_prompt(str(row["instruction"])) for row in batch_rows]
    choice_labels = [build_choice_labels(str(row["instruction"])) for row in batch_rows]
    continuation_ids = [
        build_continuation_token_ids(tokenizer, prompt, labels)
        for prompt, labels in zip(prompts, choice_labels)
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    initial_input_len = int(inputs["input_ids"].shape[1])
    max_choice_len = max(len(ids) for sample_ids in continuation_ids for ids in sample_ids)
    prefix_fn = build_prefix_allowed_tokens_fn(
        allowed_sequences_per_sample=continuation_ids,
        initial_input_len=initial_input_len,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_choice_len + 1,
            do_sample=False,
            num_beams=1,
            prefix_allowed_tokens_fn=prefix_fn,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_suffix = generated[:, initial_input_len:]
    decoded_suffix = tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)

    outputs: list[dict] = []
    for row, labels, raw_output in zip(batch_rows, choice_labels, decoded_suffix):
        pred = decode_prediction(raw_output)
        outputs.append(
            {
                "instruction": row["instruction"],
                "answer": row["answer"],
                "pred": pred,
                "output_pred": raw_output.strip(),
                "choice_labels": labels,
            }
        )
    return outputs


def evaluate_batch_free_generation(tokenizer, model, batch_rows: list[dict], max_new_tokens: int) -> list[dict]:
    prompts = [generate_prompt(str(row["instruction"])) for row in batch_rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = int(inputs["input_ids"].shape[1])
    generated_suffix = generated[:, prompt_len:]
    decoded_suffix = tokenizer.batch_decode(generated_suffix, skip_special_tokens=True)

    outputs: list[dict] = []
    for row, raw_output in zip(batch_rows, decoded_suffix):
        pred = decode_prediction(raw_output)
        outputs.append(
            {
                "instruction": row["instruction"],
                "answer": row["answer"],
                "pred": pred,
                "output_pred": raw_output.strip(),
            }
        )
    return outputs


def build_candidate_responses(instruction: str) -> list[tuple[str, str]]:
    option_map = parse_labeled_choices(instruction)
    labels = sorted(option_map.keys(), key=label_sort_key)
    candidates: list[tuple[str, str]] = []
    for label in labels:
        choice_text = str(option_map[label]).strip()
        candidates.append((label, f"the correct answer is {label}. {choice_text}"))
    return candidates


def _score_plain_forward(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs


def _score_pyvene_forward(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
):
    if not hasattr(model, "intervenable_model"):
        return _score_plain_forward(model=model, input_ids=input_ids, attention_mask=attention_mask)

    base_model = getattr(model, "base_model", model)
    model_device = next(base_model.parameters()).device
    model_dtype = next(base_model.parameters()).dtype
    if hasattr(model, "_materialize_source_representations"):
        source_representations = model._materialize_source_representations(
            device=model_device,
            dtype=model_dtype,
        )
    else:
        source_representations = [
            representation.to(device=model_device, dtype=model_dtype)
            for representation in model.source_representations
        ]

    if bool(getattr(model, "intervene_on_prompt", False)):
        base_location = -1
        unit_locations_cfg = getattr(model, "unit_locations", None)
        if isinstance(unit_locations_cfg, dict) and "base" in unit_locations_cfg:
            base_location = int(unit_locations_cfg["base"])
        resolved_location = prompt_len + base_location if base_location < 0 else base_location
        unit_locations = {"base": int(resolved_location)}
    else:
        sequence_length = int(attention_mask.sum().item())
        completion_positions = list(range(prompt_len, sequence_length))
        unit_locations = {"base": completion_positions}

    _, counterfactual_outputs = model.intervenable_model(
        base={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        source_representations=source_representations,
        unit_locations=unit_locations,
        use_cache=False,
    )
    return counterfactual_outputs


def score_candidate_response(
    *,
    tokenizer,
    model,
    prompt: str,
    response: str,
    reduction: str,
) -> tuple[float, int]:
    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    full_inputs = tokenizer(prompt + response, return_tensors="pt")

    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    input_ids = full_inputs["input_ids"].to(model_device)
    attention_mask = full_inputs["attention_mask"].to(model_device)
    full_len = int(attention_mask.sum().item())

    if full_len <= prompt_len:
        return float("-inf"), 0

    outputs = _score_pyvene_forward(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_len=prompt_len,
    )
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    completion_start = max(prompt_len - 1, 0)
    completion_end = full_len - 1
    completion_token_log_probs = token_log_probs[:, completion_start:completion_end]
    token_count = int(completion_token_log_probs.shape[1])
    if token_count <= 0:
        return float("-inf"), 0

    if reduction == "sum":
        score = float(completion_token_log_probs.sum().item())
    else:
        score = float(completion_token_log_probs.mean().item())
    return score, token_count


def evaluate_batch_candidate_scoring(tokenizer, model, batch_rows: list[dict], reduction: str) -> list[dict]:
    outputs: list[dict] = []
    for row in batch_rows:
        instruction = str(row["instruction"])
        prompt = generate_prompt(instruction)
        candidate_scores = []
        for label, response in build_candidate_responses(instruction):
            score, token_count = score_candidate_response(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                response=response,
                reduction=reduction,
            )
            candidate_scores.append(
                {
                    "label": label,
                    "response": response,
                    "score": score,
                    "token_count": token_count,
                }
            )

        best_candidate = max(candidate_scores, key=lambda item: (item["score"], item["label"]))
        outputs.append(
            {
                "instruction": instruction,
                "answer": row["answer"],
                "pred": best_candidate["label"],
                "output_pred": best_candidate["response"],
                "candidate_scores": candidate_scores,
            }
        )
    return outputs


def build_summary(args: argparse.Namespace, pyvene_config: dict, results_path: str, results: list[dict]) -> dict:
    correct = sum(int(item["pred"] == item["answer"]) for item in results)
    vector_dirs = None
    if args.pyvene_vector_dirs:
        vector_dirs = [str(path) for path in args.pyvene_vector_dirs]

    model_tag = build_pyvene_model_tag(
        base_model_path=args.base_model,
        method_tag=args.pyvene_method_tag,
        vector_dir=args.pyvene_vector_dir,
        vector_dirs=vector_dirs,
        layers=args.pyvene_layers,
        alpha=args.pyvene_alpha,
        component=args.pyvene_component,
        composition=args.pyvene_composition,
        weights=args.pyvene_weights,
        intervene_on_prompt=bool(args.pyvene_intervene_on_prompt),
        base_unit_location=args.pyvene_base_unit_location,
    )

    return {
        "dataset": args.dataset,
        "result_file": results_path,
        "num_samples": len(results),
        "correct": correct,
        "accuracy": (correct / len(results)) if results else 0.0,
        "batch_size": args.batch_size,
        "base_model": args.base_model,
        "pyvene_method_tag": args.pyvene_method_tag,
        "pyvene_component": args.pyvene_component,
        "pyvene_vector_dir": args.pyvene_vector_dir,
        "pyvene_vector_dirs": vector_dirs,
        "pyvene_layers": args.pyvene_layers,
        "pyvene_alpha": args.pyvene_alpha,
        "pyvene_composition": args.pyvene_composition,
        "pyvene_weights": args.pyvene_weights,
        "pyvene_intervene_on_prompt": bool(args.pyvene_intervene_on_prompt),
        "pyvene_base_unit_location": args.pyvene_base_unit_location,
        "evaluation_mode": getattr(args, "_resolved_evaluation_mode", "auto"),
        "constraint_target": (
            "answer_labels_only"
            if getattr(args, "_resolved_evaluation_mode", "auto") == "constrained_label_generation"
            else (
                "free_generation_then_parse_answer_labels"
                if getattr(args, "_resolved_evaluation_mode", "auto") == "free_generation"
                else "candidate_completion_scoring"
            )
        ),
        "candidate_score_reduction": args.candidate_score_reduction,
        "pyvene_config": pyvene_config,
        "model_tag": model_tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.max_samples)
    tokenizer, model, pyvene_config = load_tokenizer_and_model(args)

    resolved_mode = args.evaluation_mode
    if resolved_mode == "auto":
        resolved_mode = "free_generation" if args.dataset in {"truthfulqa_mc", "truthfulqa_mc_dev"} else "constrained_label_generation"
    args._resolved_evaluation_mode = resolved_mode

    results: list[dict] = []
    for start in tqdm(range(0, len(dataset), args.batch_size), desc=f"eval-{args.dataset}"):
        batch_rows = dataset[start : start + args.batch_size]
        if resolved_mode == "free_generation":
            results.extend(
                evaluate_batch_free_generation(
                    tokenizer,
                    model,
                    batch_rows,
                    max_new_tokens=args.free_generation_max_new_tokens,
                )
            )
        elif resolved_mode == "candidate_scoring":
            results.extend(
                evaluate_batch_candidate_scoring(
                    tokenizer,
                    model,
                    batch_rows,
                    reduction=args.candidate_score_reduction,
                )
            )
        else:
            results.extend(evaluate_batch_constrained(tokenizer, model, batch_rows))

    results_path = args.results_json or f"baseline/{args.pyvene_method_tag}-{args.dataset}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    summary = build_summary(args, pyvene_config, results_path, results)
    if args.summary_file:
        os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
        with open(args.summary_file, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
