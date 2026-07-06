from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baseline.CAA_0.steering import format_caa_text, get_decoder_layers, resolve_target_layers
from baseline.CAA_0.task_registry import (
    VECTOR_ROOT,
    get_task_input_mode,
    get_task_schema_type,
    get_train_dataset_path,
    normalize_task_name,
)
from baseline.truth_label_utils import expand_truth_response_with_choice_text


def resolve_base_model_path(candidate: str) -> str:
    path = Path(candidate).expanduser()
    if path.is_dir() and (path / "config.json").exists():
        return str(path)
    refs_main = path / "refs/main"
    if refs_main.exists():
        snapshot_id = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = path / "snapshots" / snapshot_id
        if snapshot_path.is_dir():
            return str(snapshot_path)
    if "snapshots" in path.parts:
        try:
            snap_idx = path.parts.index("snapshots")
        except ValueError:
            return str(path)
        repo_root = Path(*path.parts[:snap_idx])
        refs_main = repo_root / "refs/main"
        if refs_main.exists():
            snapshot_id = refs_main.read_text(encoding="utf-8").strip()
            snapshot_path = repo_root / "snapshots" / snapshot_id
            if snapshot_path.is_dir():
                return str(snapshot_path)
    return str(path)


class ContrastiveTextDataset(Dataset):
    def __init__(self, rows: list[dict], input_mode: str, task: str) -> None:
        self.rows = rows
        self.input_mode = input_mode
        self.task = task

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        positive = row.get("answer_matching_behavior", "")
        negative = row.get("answer_not_matching_behavior", "")
        if self.task == "truth":
            positive = expand_truth_response_with_choice_text(row.get("question", ""), positive)
            negative = expand_truth_response_with_choice_text(row.get("question", ""), negative)
        return {
            "positive_text": format_caa_text(
                row.get("question", ""),
                positive,
                mode=self.input_mode,
            ),
            "negative_text": format_caa_text(
                row.get("question", ""),
                negative,
                mode=self.input_mode,
            ),
        }


def load_rows(source_json: Path, max_samples: int | None) -> list[dict]:
    with source_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if max_samples is not None:
        rows = rows[: max_samples]
    return rows


def validate_rows_for_task(rows: list[dict], task: str, input_mode: str) -> dict:
    empty_question_count = 0
    nonempty_question_count = 0
    invalid_examples: list[str] = []

    for idx, row in enumerate(rows):
        question = str(row.get("question", "") or "").strip()
        positive = str(row.get("answer_matching_behavior", "") or "").strip()
        negative = str(row.get("answer_not_matching_behavior", "") or "").strip()

        if question:
            nonempty_question_count += 1
        else:
            empty_question_count += 1

        if not positive:
            invalid_examples.append(f"row[{idx}] missing answer_matching_behavior")
        if not negative:
            invalid_examples.append(f"row[{idx}] missing answer_not_matching_behavior")
        if positive and negative and positive == negative:
            invalid_examples.append(f"row[{idx}] positive and negative texts are identical")

        if input_mode == "qa_response" and not question:
            invalid_examples.append(
                f"row[{idx}] expected non-empty question for task={task}, got empty question"
            )
        if len(invalid_examples) >= 20:
            break

    if invalid_examples:
        details = "\n".join(invalid_examples)
        raise ValueError(
            "CAA training data schema validation failed.\n"
            f"task={task}\n"
            f"input_mode={input_mode}\n"
            f"sample_errors=\n{details}"
        )

    return {
        "task_schema_type": get_task_schema_type(task),
        "task_input_mode": input_mode,
        "total_examples": len(rows),
        "empty_question_count": empty_question_count,
        "nonempty_question_count": nonempty_question_count,
        "question_policy": (
            "required_nonempty"
            if input_mode == "qa_response"
            else "optional_mixed"
        ),
        "text_construction_policy": (
            "nonempty_question->reft_prompt_template_plus_answer; empty_question->raw_answer_text"
            if input_mode == "raw_contrastive"
            else "reft_prompt_template_plus_answer"
        ),
        "positive_negative_policy": "both_required_and_distinct",
    }


def collate_batch(tokenizer, examples: list[dict]) -> dict:
    positive_texts = [ex["positive_text"] for ex in examples]
    negative_texts = [ex["negative_text"] for ex in examples]
    pos_inputs = tokenizer(
        positive_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    neg_inputs = tokenizer(
        negative_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {"pos": pos_inputs, "neg": neg_inputs}


def _extract_hidden(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def _collect_last_token_states(model, layer_indices, encoded_batch, device):
    decoder_layers = get_decoder_layers(model)
    captures = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            captures[layer_idx] = _extract_hidden(output).detach()
        return hook

    for layer_idx in layer_indices:
        handles.append(decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
        with torch.no_grad():
            model(**encoded_batch)
        lengths = encoded_batch["attention_mask"].sum(dim=1) - 1
        last_states = {}
        for layer_idx in layer_indices:
            hidden = captures[layer_idx]
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            last_states[layer_idx] = hidden[batch_indices, lengths, :].float().cpu()
        return last_states
    finally:
        for handle in handles:
            handle.remove()


def choose_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return mapping[dtype_name]


def main() -> None:
    parser = argparse.ArgumentParser("Train project-specific CAA steering vectors.")
    parser.add_argument("--task", type=str, required=True, choices=["truth", "bias", "ethics", "toxicity"])
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--source_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    args = parser.parse_args()

    task = normalize_task_name(args.task)
    input_mode = get_task_input_mode(task)
    base_model = resolve_base_model_path(args.base_model)
    source_json = args.source_json or get_train_dataset_path(task)
    output_dir = args.output_dir or (VECTOR_ROOT / Path(base_model).name / task)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(source_json=source_json, max_samples=args.max_samples)
    schema_validation = validate_rows_for_task(rows=rows, task=task, input_mode=input_mode)
    dataset = ContrastiveTextDataset(rows, input_mode=input_mode, task=task)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=choose_dtype(args.torch_dtype),
        device_map=args.device,
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
            raise ValueError(
                "Tokenizer has no pad_token/unk_token/eos_token/bos_token available. "
                "Refusing to add a new token during CAA vector training because resizing embeddings is too expensive."
            )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    layer_indices = resolve_target_layers(model, args.layers)
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )

    pos_sums = {layer: None for layer in layer_indices}
    neg_sums = {layer: None for layer in layer_indices}
    count = 0

    for batch in loader:
        pos_states = _collect_last_token_states(model, layer_indices, batch["pos"], device)
        neg_states = _collect_last_token_states(model, layer_indices, batch["neg"], device)
        batch_size = next(iter(pos_states.values())).shape[0]
        count += batch_size
        for layer in layer_indices:
            pos_sum = pos_states[layer].sum(dim=0)
            neg_sum = neg_states[layer].sum(dim=0)
            pos_sums[layer] = pos_sum if pos_sums[layer] is None else pos_sums[layer] + pos_sum
            neg_sums[layer] = neg_sum if neg_sums[layer] is None else neg_sums[layer] + neg_sum

    summary = {
        "task": task,
        "base_model": base_model,
        "source_json": str(source_json),
        "output_dir": str(output_dir),
        "layers": layer_indices,
        "num_examples": count,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "torch_dtype": args.torch_dtype,
        "activation_position": "last_nonpad_token",
        "task_schema_type": schema_validation["task_schema_type"],
        "task_input_mode": input_mode,
        "schema_validation": schema_validation,
        "truth_response_expansion": "label_plus_choice_text" if task == "truth" else "disabled",
        "empty_question_policy": "raw_answer_text" if input_mode == "raw_contrastive" else "not_applicable",
        "nonempty_question_policy": "reft_prompt_template_plus_answer",
        "vector_files": {},
    }

    for layer in layer_indices:
        vector = (pos_sums[layer] / count) - (neg_sums[layer] / count)
        vector_path = output_dir / f"layer_{layer}.pt"
        torch.save(vector, vector_path)
        summary["vector_files"][str(layer)] = str(vector_path)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
