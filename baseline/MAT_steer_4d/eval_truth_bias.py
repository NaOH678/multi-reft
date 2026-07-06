from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from multi_train.eval_common.output_naming import build_output_path, build_model_tag

from .interveners import build_mat_model_tag
from .steering import apply_mat_steering


def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        value = v.strip().lower()
        if value in {"true", "1", "yes", "y", "t"}:
            return True
        if value in {"false", "0", "no", "n", "f"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def generate_prompt(instruction: str) -> str:
    return """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
%s

### Response:
""" % instruction


def load_data(dataset_name: str) -> list[dict]:
    path = PROJECT_ROOT / "dataset" / dataset_name / "test.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_batch(dataset: list[dict], batch_size: int) -> list[list[dict]]:
    return [dataset[idx : idx + batch_size] for idx in range(0, len(dataset), batch_size)]


def extract_labeled_choice(text: str, label_prefix: str, max_choices: int) -> str:
    lowered = str(text or "").strip().lower()
    pattern = re.compile(rf"{re.escape(label_prefix)}\s*(\d+)")
    matches = pattern.findall(lowered)
    if not matches:
        return ""
    candidate = f"{label_prefix}{matches[0]}"
    try:
        label_num = int(matches[0])
    except ValueError:
        return ""
    if label_num < 1 or label_num > int(max_choices):
        return ""
    return candidate


def extract_answer(dataset_name: str, sentence: str) -> str:
    if dataset_name in {"truthfulqa_mc", "truthfulqa_mc_dev", "bbq"}:
        return extract_labeled_choice(sentence, label_prefix="answer", max_choices=16)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate MAT-Steer on truthfulqa_mc or bbq")
    parser.add_argument("--dataset", choices=["truthfulqa_mc", "truthfulqa_mc_dev", "bbq"], required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mat_checkpoint", type=str, required=True)
    parser.add_argument("--mat_layer", type=int, default=14)
    parser.add_argument("--mat_alpha", type=float, default=1.0)
    parser.add_argument("--mat_token_strategy", choices=["last", "all"], default="last")
    parser.add_argument("--results_json", type=str, default=None)
    parser.add_argument("--summary_file", type=str, default=None)
    parser.add_argument("--greedy_decoding", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--print_samples", type=str2bool, nargs="?", const=True, default=False)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, mat_config = apply_mat_steering(
        model=model,
        checkpoint_path=args.mat_checkpoint,
        layer=args.mat_layer,
        alpha=args.mat_alpha,
        token_strategy=args.mat_token_strategy,
    )
    print(f"apply mat steering: {mat_config}")
    return tokenizer, model


def evaluate_batch(prompts: list[str], tokenizer, model, args: argparse.Namespace) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
    generation_args = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 32,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.greedy_decoding:
        generation_args["do_sample"] = False
    else:
        generation_args.update({"do_sample": True, "temperature": 0.1, "top_p": 0.75, "top_k": 40})

    with torch.no_grad():
        response = model.generate(**generation_args)
    if isinstance(response, tuple):
        response = response[-1]

    outputs = tokenizer.batch_decode(response, skip_special_tokens=True)
    cleaned = []
    for prompt, output in zip(prompts, outputs):
        text = str(output)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        elif "### Response:" in text:
            text = text.split("### Response:")[-1]
        cleaned.append(text.strip())
    return cleaned


def main() -> None:
    args = parse_args()
    output_dir = PROJECT_ROOT / "baseline" / "MAT_steer_4d" / "runs" / "eval_truth"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = build_mat_model_tag(
        base_model_path=args.base_model,
        checkpoint_path=args.mat_checkpoint,
        layer=args.mat_layer,
        alpha=args.mat_alpha,
        token_strategy=args.mat_token_strategy,
    )
    save_file = args.results_json or str(output_dir / f"{model_tag}-{args.dataset}.json")
    summary_file = args.summary_file or str(Path(save_file).with_name(f"{Path(save_file).stem}_summary.json"))

    dataset = load_data(args.dataset)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    correct = 0
    current = 0
    outputs_all = []
    pbar = tqdm(total=len(batches))
    for idx, batch in enumerate(batches):
        current += len(batch)
        prompts = [generate_prompt(item["instruction"]) for item in batch]
        outputs = evaluate_batch(prompts, tokenizer, model, args)
        for item, output in zip(batch, outputs):
            label = item.get("answer")
            pred = extract_answer(args.dataset, output)
            flag = label == pred
            if flag:
                correct += 1
            row = copy.deepcopy(item)
            row["output_pred"] = output
            row["pred"] = pred
            row["flag"] = flag
            outputs_all.append(row)
            if args.print_samples:
                print(item["instruction"])
                print(output)
                print("prediction:", pred)
                print("label:", label)
        with open(save_file, "w", encoding="utf-8") as f:
            json.dump(outputs_all, f, ensure_ascii=False, indent=2)
        print(f"\rtest:{idx + 1}/{len(batches)} | accuracy {correct} {correct / current if current else 0.0}")
        pbar.update(1)
    pbar.close()

    summary = {
        "dataset": args.dataset,
        "result_file": save_file,
        "num_samples": current,
        "correct": correct,
        "accuracy": (correct / current) if current > 0 else 0.0,
        "base_model": args.base_model,
        "mat_checkpoint": args.mat_checkpoint,
        "mat_layer": args.mat_layer,
        "mat_alpha": args.mat_alpha,
        "mat_token_strategy": args.mat_token_strategy,
        "greedy_decoding": bool(args.greedy_decoding),
        "batch_size": args.batch_size,
        "model_tag": model_tag,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
