import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_train.eval_common.build_residual_stats import (
    load_prompts,
    load_specialist_prompts,
    prepare_unit_locations,
    resolve_target_layers,
)
from multi_train.eval_common.composable_loreft import (
    load_loreft_specialist_state,
    normalize_specialist_label,
)


DEFAULT_ROUTER_FEATURE_NAMES = [
    "delta_norm",
    "intervention_norm",
    "subspace_energy",
    "mean_compat",
    "neg_compat_mass",
]


def parse_args():
    parser = argparse.ArgumentParser("Build router feature calibration statistics")
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
    parser.add_argument("--specialist_prompt_map_json", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument(
        "--feature_names",
        type=str,
        nargs="+",
        default=DEFAULT_ROUTER_FEATURE_NAMES,
        choices=DEFAULT_ROUTER_FEATURE_NAMES,
    )
    parser.add_argument("--output_json", type=str, required=True)
    return parser.parse_args()


def init_stats(target_layers: List[int], specialist_labels: List[str], feature_names: List[str], calibration_mode: str):
    return {
        "version": 1,
        "normalizer_scope": "layer_specialist_feature",
        "calibration_mode": calibration_mode,
        "feature_names": feature_names,
        "specialist_labels": specialist_labels,
        "layers": {
            str(layer): {
                label: {
                    feature_name: {
                        "count": 0,
                        "sum": 0.0,
                        "sq_sum": 0.0,
                    }
                    for feature_name in feature_names
                }
                for label in specialist_labels
            }
            for layer in target_layers
        },
    }


def update_stats_entry(entry: Dict[str, float], values: torch.Tensor) -> None:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return
    entry["count"] += int(values.numel())
    entry["sum"] += float(values.sum().item())
    entry["sq_sum"] += float((values ** 2).sum().item())


def finalize_stats(raw_stats: Dict, prompt_sources: Dict[str, int]) -> Dict:
    finalized = {
        "version": raw_stats["version"],
        "normalizer_scope": raw_stats["normalizer_scope"],
        "calibration_mode": raw_stats["calibration_mode"],
        "feature_names": raw_stats["feature_names"],
        "specialist_labels": raw_stats["specialist_labels"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sources": prompt_sources,
        "layers": {},
    }

    for layer, layer_stats in raw_stats["layers"].items():
        finalized["layers"][layer] = {}
        for label, feature_stats in layer_stats.items():
            finalized["layers"][layer][label] = {}
            for feature_name, entry in feature_stats.items():
                count = int(entry["count"])
                if count <= 0:
                    finalized["layers"][layer][label][feature_name] = {
                        "count": 0,
                        "mean": 0.0,
                        "std": 0.0,
                    }
                    continue
                mean = entry["sum"] / count
                var = max(entry["sq_sum"] / count - mean * mean, 0.0)
                finalized["layers"][layer][label][feature_name] = {
                    "count": count,
                    "mean": mean,
                    "std": var ** 0.5,
                }
    return finalized


def _compute_router_features(hidden_at_units, layer_states, feature_names):
    rotate_stack = torch.stack([state["rotate_weight"] for state in layer_states], dim=0).to(hidden_at_units.device).float()
    source_weight_stack = torch.stack([state["source_weight"] for state in layer_states], dim=0).to(hidden_at_units.device).float()
    source_bias_stack = torch.stack([state["source_bias"] for state in layer_states], dim=0).to(hidden_at_units.device).float()

    rotated = torch.einsum("bsd,tdr->bstr", hidden_at_units, rotate_stack)
    learned = torch.einsum("bsd,trd->bstr", hidden_at_units, source_weight_stack)
    learned = learned + source_bias_stack.unsqueeze(0).unsqueeze(0)
    delta = learned - rotated
    lifted = torch.einsum("bstr,tdr->bstd", delta, rotate_stack)

    features = {}
    if "rh_log_norm" in feature_names:
        rh_norm = rotated.norm(dim=-1)
        features["rh_log_norm"] = torch.log(rh_norm.clamp_min(1e-6))
    if "delta_log_norm" in feature_names:
        delta_norm = delta.norm(dim=-1)
        features["delta_log_norm"] = torch.log(delta_norm.clamp_min(1e-6))
    if "delta_norm" in feature_names:
        features["delta_norm"] = delta.norm(dim=-1)
    if "intervention_norm" in feature_names:
        features["intervention_norm"] = lifted.norm(dim=-1)
    if "subspace_energy" in feature_names:
        features["subspace_energy"] = rotated.pow(2).sum(dim=-1)
    if "mean_compat" in feature_names or "neg_compat_mass" in feature_names:
        normalized = F.normalize(lifted, dim=-1, eps=1e-6)
        pairwise_cos = torch.einsum("bstd,bsud->bstu", normalized, normalized)
        diag = torch.eye(pairwise_cos.shape[-1], device=pairwise_cos.device, dtype=torch.bool).view(
            1, 1, pairwise_cos.shape[-1], pairwise_cos.shape[-1]
        )
        offdiag = pairwise_cos.masked_fill(diag, 0.0)
        denom = max(1, pairwise_cos.shape[-1] - 1)
        if "mean_compat" in feature_names:
            features["mean_compat"] = offdiag.sum(dim=-1) / denom
        if "neg_compat_mass" in feature_names:
            features["neg_compat_mass"] = torch.clamp(-offdiag, min=0.0).sum(dim=-1)
    return features


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
):
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
            layer_states = [state[layer] for state in specialist_states]
            feature_map = _compute_router_features(hidden_at_units, layer_states, args.feature_names)

            for specialist_idx in active_specialists:
                label = specialist_labels[specialist_idx]
                for feature_name in args.feature_names:
                    values = feature_map[feature_name][..., specialist_idx]
                    update_stats_entry(stats["layers"][str(layer)][label][feature_name], values)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    specialist_states = [load_loreft_specialist_state(path) for path in args.reft_specialists]
    specialist_labels = [normalize_specialist_label(path) for path in args.reft_specialists]
    target_layers = resolve_target_layers(specialist_states, args.target_layers)
    stats = init_stats(target_layers, specialist_labels, args.feature_names, args.calibration_mode)

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
    print(f"Router feature stats written to: {output_path}")


if __name__ == "__main__":
    main()
