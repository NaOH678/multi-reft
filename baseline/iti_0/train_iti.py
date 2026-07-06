from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from baseline.iti_0.steering import (
    format_iti_text,
    get_attention_output_projection,
    get_decoder_layers,
)
from baseline.iti_0.task_registry import (
    DEFAULT_LOGISTIC_MAX_ITER,
    DEFAULT_TOP_HEADS,
    DEFAULT_VAL_RATIO,
    INTERVENTION_ROOT,
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
            "positive_text": format_iti_text(
                row.get("question", ""),
                positive,
                mode=self.input_mode,
            ),
            "negative_text": format_iti_text(
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
            "ITI training data schema validation failed.\n"
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


def choose_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return mapping[dtype_name]


def split_train_val_indices(num_pairs: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_pairs <= 1:
        idxs = np.arange(num_pairs, dtype=np.int64)
        return idxs, idxs

    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_pairs)
    val_count = int(round(num_pairs * float(val_ratio)))
    val_count = max(1, min(num_pairs - 1, val_count))
    val_idxs = np.sort(perm[:val_count])
    train_idxs = np.sort(perm[val_count:])
    return train_idxs, val_idxs


def _collect_last_token_head_inputs(model, layer_indices, encoded_batch, device, num_heads, head_dim):
    captures = {}
    handles = []
    decoder_layers = get_decoder_layers(model)

    def make_pre_hook(layer_idx: int):
        def pre_hook(_module, inputs):
            if inputs:
                captures[layer_idx] = inputs[0].detach()
            return inputs

        return pre_hook

    for layer_idx in layer_indices:
        projection = get_attention_output_projection(decoder_layers[layer_idx])
        handles.append(projection.register_forward_pre_hook(make_pre_hook(layer_idx)))

    try:
        encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
        with torch.no_grad():
            model(**encoded_batch)

        lengths = encoded_batch["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(lengths.shape[0], device=lengths.device)
        last_states = {}
        for layer_idx in layer_indices:
            hidden = captures[layer_idx]
            selected = hidden[batch_indices, lengths, :]
            selected = selected.reshape(selected.shape[0], num_heads, head_dim).float().cpu()
            last_states[layer_idx] = selected
        return last_states
    finally:
        for handle in handles:
            handle.remove()


def _collect_feature_cache(
    *,
    model,
    tokenizer,
    dataset,
    batch_size: int,
    layer_indices: list[int],
    num_heads: int,
    head_dim: int,
    feature_cache_dir: Path,
) -> dict[int, Path]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(tokenizer, batch),
    )
    device = next(model.parameters()).device
    num_pairs = len(dataset)
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    layer_cache_paths = {}
    layer_memmaps = {}
    for layer_idx in layer_indices:
        cache_path = feature_cache_dir / f"layer_{layer_idx}.npy"
        layer_cache_paths[layer_idx] = cache_path
        layer_memmaps[layer_idx] = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(num_pairs, 2, num_heads, head_dim),
        )

    cursor = 0
    for batch in loader:
        pos_states = _collect_last_token_head_inputs(
            model=model,
            layer_indices=layer_indices,
            encoded_batch=batch["pos"],
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        neg_states = _collect_last_token_head_inputs(
            model=model,
            layer_indices=layer_indices,
            encoded_batch=batch["neg"],
            device=device,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        current_batch = next(iter(pos_states.values())).shape[0]
        for layer_idx in layer_indices:
            layer_memmaps[layer_idx][cursor : cursor + current_batch, 0, :, :] = pos_states[layer_idx].numpy()
            layer_memmaps[layer_idx][cursor : cursor + current_batch, 1, :, :] = neg_states[layer_idx].numpy()
        cursor += current_batch

    for memmap in layer_memmaps.values():
        memmap.flush()

    return layer_cache_paths


def _fit_head_probes_for_layer(
    *,
    layer_cache_path: Path,
    train_idxs: np.ndarray,
    val_idxs: np.ndarray,
    num_heads: int,
    head_dim: int,
    seed: int,
    max_iter: int,
):
    layer_cache = np.load(layer_cache_path, mmap_mode="r")
    val_accuracies = np.zeros(num_heads, dtype=np.float32)
    coefficients = np.zeros((num_heads, head_dim), dtype=np.float32)
    intercepts = np.zeros(num_heads, dtype=np.float32)

    y_train = np.concatenate(
        [
            np.ones(len(train_idxs), dtype=np.int64),
            np.zeros(len(train_idxs), dtype=np.int64),
        ]
    )
    y_val = np.concatenate(
        [
            np.ones(len(val_idxs), dtype=np.int64),
            np.zeros(len(val_idxs), dtype=np.int64),
        ]
    )

    for head_idx in range(num_heads):
        train_pos = np.asarray(layer_cache[train_idxs, 0, head_idx, :], dtype=np.float32)
        train_neg = np.asarray(layer_cache[train_idxs, 1, head_idx, :], dtype=np.float32)
        val_pos = np.asarray(layer_cache[val_idxs, 0, head_idx, :], dtype=np.float32)
        val_neg = np.asarray(layer_cache[val_idxs, 1, head_idx, :], dtype=np.float32)

        x_train = np.concatenate([train_pos, train_neg], axis=0)
        x_val = np.concatenate([val_pos, val_neg], axis=0)

        clf = LogisticRegression(
            random_state=seed,
            max_iter=max_iter,
            solver="liblinear",
        ).fit(x_train, y_train)

        val_pred = clf.predict(x_val)
        val_accuracies[head_idx] = float((val_pred == y_val).mean())
        coefficients[head_idx] = clf.coef_[0].astype(np.float32)
        intercepts[head_idx] = float(clf.intercept_[0])

    return val_accuracies, coefficients, intercepts


def _build_selected_heads(
    *,
    head_val_acc: np.ndarray,
    probe_coefficients: np.ndarray,
    layer_cache_paths: dict[int, Path],
    top_k_heads: int,
    retain_layers: set[int] | None = None,
) -> tuple[list[dict], dict[int, torch.Tensor]]:
    num_layers, num_heads, head_dim = probe_coefficients.shape
    if retain_layers is None:
        candidate_pairs = [
            (layer_idx, head_idx)
            for layer_idx in range(num_layers)
            for head_idx in range(num_heads)
        ]
    else:
        candidate_pairs = [
            (layer_idx, head_idx)
            for layer_idx in sorted(retain_layers)
            if 0 <= layer_idx < num_layers
            for head_idx in range(num_heads)
        ]
    total_heads = len(candidate_pairs)
    top_k = min(int(top_k_heads), total_heads)
    sorted_pairs = sorted(
        candidate_pairs,
        key=lambda pair: float(head_val_acc[pair[0], pair[1]]),
        reverse=True,
    )[:top_k]

    selected_heads: list[dict] = []
    layer_vectors: dict[int, torch.Tensor] = {}

    for layer_idx, head_idx in sorted_pairs:
        coef = probe_coefficients[layer_idx, head_idx].astype(np.float32)
        coef_norm = float(np.linalg.norm(coef))
        if coef_norm <= 1e-12:
            direction = coef
        else:
            direction = coef / coef_norm

        layer_cache = np.load(layer_cache_paths[layer_idx], mmap_mode="r")
        all_examples = np.concatenate(
            [
                np.asarray(layer_cache[:, 0, head_idx, :], dtype=np.float32),
                np.asarray(layer_cache[:, 1, head_idx, :], dtype=np.float32),
            ],
            axis=0,
        )
        proj_std = float(np.std(all_examples @ direction))

        hidden_size = num_heads * head_dim
        if layer_idx not in layer_vectors:
            layer_vectors[layer_idx] = torch.zeros(hidden_size, dtype=torch.float32)
        start = head_idx * head_dim
        end = start + head_dim
        layer_vectors[layer_idx][start:end] = torch.from_numpy(direction * proj_std)

        selected_heads.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "validation_accuracy": float(head_val_acc[layer_idx, head_idx]),
                "coef_norm": coef_norm,
                "projection_std": proj_std,
            }
        )

    return selected_heads, layer_vectors


def main() -> None:
    parser = argparse.ArgumentParser("Train project-specific ITI interventions.")
    parser.add_argument("--task", type=str, required=True, choices=["truth", "bias", "ethics", "toxicity"])
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--source_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--top_k_heads", type=int, default=DEFAULT_TOP_HEADS)
    parser.add_argument("--max_iter", type=int, default=DEFAULT_LOGISTIC_MAX_ITER)
    parser.add_argument("--keep_feature_cache", type=int, default=0)
    parser.add_argument("--retain_layers", type=int, nargs="*", default=None)
    args = parser.parse_args()

    task = normalize_task_name(args.task)
    input_mode = get_task_input_mode(task)
    base_model = resolve_base_model_path(args.base_model)
    source_json = args.source_json or get_train_dataset_path(task)
    output_dir = args.output_dir or (INTERVENTION_ROOT / Path(base_model).name / task)
    feature_cache_dir = args.feature_cache_dir or (output_dir / "feature_cache")
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
                "Refusing to add a new token during ITI training because resizing embeddings is too expensive."
            )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    decoder_layers = get_decoder_layers(model)
    num_layers = len(decoder_layers)
    num_heads = int(model.config.num_attention_heads)
    hidden_size = int(model.config.hidden_size)
    head_dim = hidden_size // num_heads
    layer_indices = list(range(num_layers))
    retain_layers = None
    if args.retain_layers:
        retain_layers = {
            int(layer_idx)
            for layer_idx in args.retain_layers
            if 0 <= int(layer_idx) < num_layers
        }
        if not retain_layers:
            raise ValueError(
                f"No valid retain_layers remained after filtering: {args.retain_layers}"
            )

    layer_cache_paths = _collect_feature_cache(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        layer_indices=layer_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        feature_cache_dir=feature_cache_dir,
    )

    train_idxs, val_idxs = split_train_val_indices(
        num_pairs=len(dataset),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    head_val_acc = np.zeros((num_layers, num_heads), dtype=np.float32)
    probe_coefficients = np.zeros((num_layers, num_heads, head_dim), dtype=np.float32)
    probe_intercepts = np.zeros((num_layers, num_heads), dtype=np.float32)

    for layer_idx in layer_indices:
        val_accuracies, coefficients, intercepts = _fit_head_probes_for_layer(
            layer_cache_path=layer_cache_paths[layer_idx],
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            num_heads=num_heads,
            head_dim=head_dim,
            seed=args.seed,
            max_iter=args.max_iter,
        )
        head_val_acc[layer_idx] = val_accuracies
        probe_coefficients[layer_idx] = coefficients
        probe_intercepts[layer_idx] = intercepts

    selected_heads, layer_vectors = _build_selected_heads(
        head_val_acc=head_val_acc,
        probe_coefficients=probe_coefficients,
        layer_cache_paths=layer_cache_paths,
        top_k_heads=args.top_k_heads,
        retain_layers=retain_layers,
    )

    vector_files = {}
    for layer_idx, vector in sorted(layer_vectors.items()):
        vector_path = output_dir / f"layer_{layer_idx}.pt"
        torch.save(vector, vector_path)
        vector_files[str(layer_idx)] = str(vector_path)

    head_val_acc_path = output_dir / "head_val_acc.npy"
    np.save(head_val_acc_path, head_val_acc)

    probe_path = output_dir / "probe_coefficients.npz"
    np.savez_compressed(
        probe_path,
        coefficients=probe_coefficients,
        intercepts=probe_intercepts,
    )

    keep_feature_cache = bool(int(args.keep_feature_cache))
    if not keep_feature_cache and feature_cache_dir.exists():
        shutil.rmtree(feature_cache_dir)

    summary = {
        "task": task,
        "base_model": base_model,
        "source_json": str(source_json),
        "output_dir": str(output_dir),
        "feature_cache_dir": str(feature_cache_dir),
        "feature_cache_kept": keep_feature_cache,
        "batch_size": int(args.batch_size),
        "max_samples": args.max_samples,
        "torch_dtype": args.torch_dtype,
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "top_k_heads_requested": int(args.top_k_heads),
        "top_k_heads_selected": int(len(selected_heads)),
        "retain_layers_requested": (
            sorted(int(layer_idx) for layer_idx in retain_layers)
            if retain_layers is not None
            else None
        ),
        "max_iter": int(args.max_iter),
        "num_pairs": int(len(dataset)),
        "num_train_pairs": int(len(train_idxs)),
        "num_val_pairs": int(len(val_idxs)),
        "num_layers": int(num_layers),
        "num_attention_heads": int(num_heads),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
        "application_site": "self_attn.o_proj.input",
        "recommended_inference_behavior": "prompt_last_token_only",
        "activation_position": "last_nonpad_token",
        "task_schema_type": schema_validation["task_schema_type"],
        "task_input_mode": input_mode,
        "schema_validation": schema_validation,
        "truth_response_expansion": "label_plus_choice_text" if task == "truth" else "disabled",
        "head_val_acc_file": str(head_val_acc_path),
        "probe_coefficients_file": str(probe_path),
        "vector_files": vector_files,
        "selected_heads": selected_heads,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
