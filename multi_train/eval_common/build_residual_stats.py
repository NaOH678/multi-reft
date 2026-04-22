import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_train.eval_common.composable_loreft import (
    load_loreft_specialist_state,
    normalize_specialist_label,
)


TEXT_FIELD_CANDIDATES = [
    "text",
    "prompt",
    "user_prompt",
    "input",
    "question",
    "instruction",
    "content",
]


def parse_args():
    parser = argparse.ArgumentParser("Build residual normalization statistics")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--reft_specialists", type=str, nargs="+", required=True)
    parser.add_argument(
        "--calibration_mode",
        type=str,
        choices=["shared", "specialist"],
        default="shared",
    )
    parser.add_argument("--prompt_files", type=str, nargs="+", default=None)
    parser.add_argument(
        "--specialist_prompt_map_json",
        type=str,
        default=None,
        help="JSON file or inline JSON mapping each specialist to its own prompt file list.",
    )
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def _detect_text_field(example: Dict, preferred: str | None) -> str:
    if preferred and preferred in example:
        return preferred
    for key in TEXT_FIELD_CANDIDATES:
        if key in example:
            return key
    for key, value in example.items():
        if isinstance(value, str):
            return key
    raise ValueError(f"Could not find a text field in example keys: {list(example.keys())}")


def load_prompts_from_file(path: Path, text_column: str | None) -> List[str]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return []
        field = _detect_text_field(rows[0], text_column)
        return [str(row[field]).strip() for row in rows if str(row.get(field, "")).strip()]

    if suffix == ".jsonl":
        prompts = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, str):
                    text = item.strip()
                else:
                    field = _detect_text_field(item, text_column)
                    text = str(item[field]).strip()
                if text:
                    prompts.append(text)
        return prompts

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], str):
                return [str(item).strip() for item in data if str(item).strip()]
            field = _detect_text_field(data[0], text_column)
            return [str(item[field]).strip() for item in data if str(item.get(field, "")).strip()]
        raise ValueError(f"Unsupported JSON structure in {path}: expected a list.")

    raise ValueError(f"Unsupported prompt file suffix: {path}")


def load_prompts(prompt_files: List[str], text_column: str | None, max_samples: int | None) -> List[str]:
    prompts: List[str] = []
    for file_path in prompt_files:
        prompts.extend(load_prompts_from_file(Path(file_path), text_column))
    if max_samples is not None:
        prompts = prompts[:max_samples]
    if not prompts:
        raise ValueError("No prompts loaded for residual stats calibration.")
    return prompts


def _load_json_mapping(path_or_json: str) -> Dict:
    candidate = Path(path_or_json)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path_or_json)


def _resolve_specialist_key(raw_key: str, specialist_labels: List[str], specialist_paths: List[str]) -> str:
    if raw_key in specialist_labels:
        return raw_key

    if raw_key.isdigit():
        idx = int(raw_key)
        if idx < 0 or idx >= len(specialist_labels):
            raise ValueError(f"Specialist index out of range in specialist_prompt_map_json: {raw_key}")
        return specialist_labels[idx]

    for idx, path in enumerate(specialist_paths):
        normalized_path = str(Path(path).resolve())
        if raw_key == path or raw_key == normalized_path:
            return specialist_labels[idx]

    raise ValueError(
        "Could not match specialist_prompt_map_json key "
        f"`{raw_key}` to any specialist label, index, or path."
    )


def load_specialist_prompts(
    specialist_prompt_map_json: str | None,
    specialist_labels: List[str],
    specialist_paths: List[str],
    text_column: str | None,
    max_samples: int | None,
) -> Dict[str, List[str]]:
    if not specialist_prompt_map_json:
        raise ValueError("specialist calibration requires --specialist_prompt_map_json.")

    raw_mapping = _load_json_mapping(specialist_prompt_map_json)
    if not isinstance(raw_mapping, dict):
        raise ValueError("specialist_prompt_map_json must decode to a JSON object.")

    resolved: Dict[str, List[str]] = {}
    for raw_key, prompt_files in raw_mapping.items():
        resolved_key = _resolve_specialist_key(str(raw_key), specialist_labels, specialist_paths)
        if not isinstance(prompt_files, list) or not prompt_files:
            raise ValueError(
                f"specialist_prompt_map_json entry for `{raw_key}` must be a non-empty list of files."
            )
        resolved[resolved_key] = load_prompts([str(item) for item in prompt_files], text_column, max_samples)

    missing = [label for label in specialist_labels if label not in resolved]
    if missing:
        raise ValueError(
            "specialist_prompt_map_json is missing prompt files for specialists: "
            + ", ".join(missing)
        )
    return resolved


def prepare_unit_locations(attention_mask: torch.Tensor, seq_len: int, positions: int) -> torch.Tensor:
    base_unit_location = seq_len - 1
    shift = attention_mask.argmax(dim=1).unsqueeze(1)
    prefix = torch.arange(positions, device=attention_mask.device).repeat(attention_mask.shape[0], 1) + shift
    suffix = torch.tensor(
        [base_unit_location - i - 1 for i in range(positions - 1, -1, -1)],
        device=attention_mask.device,
    ).repeat(attention_mask.shape[0], 1)
    locations = torch.cat([prefix, suffix], dim=1)
    return locations.clamp(min=0, max=seq_len - 1)


def resolve_target_layers(
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    requested_layers: List[int],
) -> List[int]:
    common_layers = set(specialist_states[0].keys())
    for state in specialist_states[1:]:
        common_layers &= set(state.keys())
    if not common_layers:
        raise ValueError("No common layers found across specialists.")
    if requested_layers == [-1]:
        return sorted(common_layers)
    missing = [layer for layer in requested_layers if layer not in common_layers]
    if missing:
        raise ValueError(f"Requested target layers missing from specialists: {missing}")
    return requested_layers


def init_stats(target_layers: List[int], specialist_labels: List[str], calibration_mode: str) -> Dict:
    return {
        "version": 1,
        "normalizer_scope": "layer_specialist",
        "calibration_mode": calibration_mode,
        "specialist_labels": specialist_labels,
        "layers": {
            str(layer): {
                label: {
                    "count": 0,
                    "sum": 0.0,
                    "sq_sum": 0.0,
                    "log_sum": 0.0,
                    "log_sq_sum": 0.0,
                }
                for label in specialist_labels
            }
            for layer in target_layers
        },
    }


def update_stats_entry(entry: Dict[str, float], values: torch.Tensor, eps: float) -> None:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return
    log_values = torch.log(values + eps)
    entry["count"] += int(values.numel())
    entry["sum"] += float(values.sum().item())
    entry["sq_sum"] += float((values ** 2).sum().item())
    entry["log_sum"] += float(log_values.sum().item())
    entry["log_sq_sum"] += float((log_values ** 2).sum().item())


def finalize_stats(raw_stats: Dict, prompt_sources: Dict[str, int]) -> Dict:
    finalized = {
        "version": raw_stats["version"],
        "normalizer_scope": raw_stats["normalizer_scope"],
        "calibration_mode": raw_stats.get("calibration_mode", "shared"),
        "specialist_labels": raw_stats["specialist_labels"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sources": prompt_sources,
        "layers": {},
    }

    for layer, layer_stats in raw_stats["layers"].items():
        finalized["layers"][layer] = {}
        for label, entry in layer_stats.items():
            count = int(entry["count"])
            if count <= 0:
                finalized["layers"][layer][label] = {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "log_mean": 0.0,
                    "log_std": 0.0,
                }
                continue

            mean = entry["sum"] / count
            var = max(entry["sq_sum"] / count - mean * mean, 0.0)
            log_mean = entry["log_sum"] / count
            log_var = max(entry["log_sq_sum"] / count - log_mean * log_mean, 0.0)
            finalized["layers"][layer][label] = {
                "count": count,
                "mean": mean,
                "std": var ** 0.5,
                "log_mean": log_mean,
                "log_std": log_var ** 0.5,
            }

    return finalized


def _process_prompt_batch(
    model,
    tokenizer,
    prompts: List[str],
    args,
    device: str,
    target_layers: List[int],
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    specialist_labels: List[str],
    stats: Dict,
    active_specialists: List[int],
) -> None:
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        unit_locations = prepare_unit_locations(
            attention_mask=inputs["attention_mask"],
            seq_len=inputs["input_ids"].shape[1],
            positions=args.positions,
        )

        for layer in target_layers:
            hidden = outputs.hidden_states[layer + 1].float()
            gather_index = unit_locations.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            hidden_at_units = hidden.gather(1, gather_index)

            for specialist_idx in active_specialists:
                label = specialist_labels[specialist_idx]
                state = specialist_states[specialist_idx][layer]
                rotate_weight = state["rotate_weight"].to(hidden_at_units.device).float()
                source_weight = state["source_weight"].to(hidden_at_units.device).float()
                source_bias = state["source_bias"].to(hidden_at_units.device).float()

                rotated = torch.einsum("bsd,dr->bsr", hidden_at_units, rotate_weight)
                learned = torch.einsum("bsd,rd->bsr", hidden_at_units, source_weight) + source_bias.view(1, 1, -1)
                delta = learned - rotated
                delta_norm = delta.norm(dim=-1)
                update_stats_entry(stats["layers"][str(layer)][label], delta_norm, eps=1e-6)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    specialist_states = [load_loreft_specialist_state(path) for path in args.reft_specialists]
    specialist_labels = [normalize_specialist_label(path) for path in args.reft_specialists]
    target_layers = resolve_target_layers(specialist_states, args.target_layers)
    stats = init_stats(target_layers, specialist_labels, args.calibration_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    prompt_sources: Dict[str, int]
    if args.calibration_mode == "shared":
        if not args.prompt_files:
            raise ValueError("shared calibration requires --prompt_files.")
        prompts = load_prompts(args.prompt_files, args.text_column, args.max_samples)
        prompt_sources = {"shared": len(prompts)}
        _process_prompt_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            args=args,
            device=device,
            target_layers=target_layers,
            specialist_states=specialist_states,
            specialist_labels=specialist_labels,
            stats=stats,
            active_specialists=list(range(len(specialist_labels))),
        )
    else:
        specialist_prompts = load_specialist_prompts(
            specialist_prompt_map_json=args.specialist_prompt_map_json,
            specialist_labels=specialist_labels,
            specialist_paths=args.reft_specialists,
            text_column=args.text_column,
            max_samples=args.max_samples,
        )
        prompt_sources = {label: len(prompts) for label, prompts in specialist_prompts.items()}
        for specialist_idx, label in enumerate(specialist_labels):
            _process_prompt_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=specialist_prompts[label],
                args=args,
                device=device,
                target_layers=target_layers,
                specialist_states=specialist_states,
                specialist_labels=specialist_labels,
                stats=stats,
                active_specialists=[specialist_idx],
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(finalize_stats(stats, prompt_sources=prompt_sources), f, indent=2)
    print(f"Residual stats written to: {output_path}")


if __name__ == "__main__":
    main()
