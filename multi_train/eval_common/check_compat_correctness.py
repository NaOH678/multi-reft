import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multi_train.eval_common.build_residual_stats import load_prompts, prepare_unit_locations, resolve_target_layers
from multi_train.eval_common.composable_loreft import (
    _build_layer_score_stats,
    load_score_stats,
    load_loreft_specialist_state,
    normalize_specialist_label,
)

try:
    from pyreft import ComposableLoreftIntervention
except ImportError:
    from pyreft.pyreft import ComposableLoreftIntervention


def parse_args():
    parser = argparse.ArgumentParser("Check legacy vs optimized compatibility correctness")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--reft_specialists", type=str, nargs="+", required=True)
    parser.add_argument("--score_stats_path", type=str, required=True)
    parser.add_argument("--prompt_files", type=str, nargs="+", required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--target_layers", type=int, nargs="+", default=[-1])
    parser.add_argument("--compose_domain", type=str, default="output", choices=["output", "projected_output", "shared_latent"])
    parser.add_argument("--composition_temperature", type=float, default=1.0)
    parser.add_argument("--composition_topk", type=int, default=2)
    parser.add_argument("--compat_threshold", type=float, default=0.0)
    parser.add_argument("--score_source", type=str, default="intervention_norm", choices=["delta_norm", "intervention_norm"])
    parser.add_argument("--score_normalizer", type=str, default="log_zscore", choices=["none", "mean_ratio", "log_zscore"])
    return parser.parse_args()


def _max_abs_diff(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a.float() - b.float()).abs().max().item())


def _compare_bool(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    return float((a.to(torch.int) != b.to(torch.int)).sum().item())


def _build_intervention(args, layer, layer_states, layer_score_stats, compat_impl):
    rotate_weight = torch.stack([state["rotate_weight"] for state in layer_states], dim=0)
    source_weight = torch.stack([state["source_weight"] for state in layer_states], dim=0)
    source_bias = torch.stack([state["source_bias"] for state in layer_states], dim=0)
    return ComposableLoreftIntervention(
        embed_dim=rotate_weight.shape[1],
        low_rank_dimension=source_weight.shape[1],
        num_specialists=len(layer_states),
        compose_domain=args.compose_domain,
        policy_type="compat_filtered_topk",
        temperature=args.composition_temperature,
        topk=args.composition_topk,
        compat_threshold=args.compat_threshold,
        score_source=args.score_source,
        score_normalizer=args.score_normalizer,
        score_stats=layer_score_stats,
        rotate_weight=rotate_weight,
        source_weight=source_weight,
        source_bias=source_bias,
        enable_debug_cache=True,
        compat_impl=compat_impl,
        add_bias=False,
    )


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args.prompt_files, text_column=None, max_samples=args.max_samples)
    specialist_states = [load_loreft_specialist_state(path) for path in args.reft_specialists]
    specialist_labels = [normalize_specialist_label(path) for path in args.reft_specialists]
    target_layers = resolve_target_layers(specialist_states, args.target_layers)
    score_stats = load_score_stats(args.score_stats_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    report: Dict[str, Dict] = {
        "version": 1,
        "target_layers": target_layers,
        "composition_topk": args.composition_topk,
        "score_source": args.score_source,
        "score_normalizer": args.score_normalizer,
        "layers": {},
    }

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
            layer_score_stats = _build_layer_score_stats(score_stats, layer, specialist_labels)

            legacy = _build_intervention(args, layer, layer_states, layer_score_stats, compat_impl="legacy").to(device)
            optimized = _build_intervention(args, layer, layer_states, layer_score_stats, compat_impl="optimized").to(device)
            legacy.eval()
            optimized.eval()

            with torch.no_grad():
                legacy_out = legacy(hidden_at_units)
                optimized_out = optimized(hidden_at_units)

            layer_key = str(layer)
            layer_report = report["layers"].setdefault(
                layer_key,
                {
                    "max_output_diff": 0.0,
                    "max_alpha_diff": 0.0,
                    "max_score_diff": 0.0,
                    "max_pairwise_diff": 0.0,
                    "max_delta_norm_diff": 0.0,
                    "max_intervention_norm_diff": 0.0,
                    "selected_mask_mismatches": 0.0,
                    "topk_mask_mismatches": 0.0,
                    "rejected_conflict_mask_mismatches": 0.0,
                },
            )

            layer_report["max_output_diff"] = max(layer_report["max_output_diff"], _max_abs_diff(legacy_out, optimized_out))
            layer_report["max_alpha_diff"] = max(layer_report["max_alpha_diff"], _max_abs_diff(legacy.latest_alpha, optimized.latest_alpha))
            layer_report["max_score_diff"] = max(layer_report["max_score_diff"], _max_abs_diff(legacy.latest_scores, optimized.latest_scores))
            layer_report["max_pairwise_diff"] = max(
                layer_report["max_pairwise_diff"],
                _max_abs_diff(legacy.latest_pairwise_cos, optimized.latest_pairwise_cos),
            )
            layer_report["max_delta_norm_diff"] = max(
                layer_report["max_delta_norm_diff"],
                _max_abs_diff(legacy.latest_delta_norm, optimized.latest_delta_norm),
            )
            layer_report["max_intervention_norm_diff"] = max(
                layer_report["max_intervention_norm_diff"],
                _max_abs_diff(legacy.latest_intervention_norm, optimized.latest_intervention_norm),
            )
            layer_report["selected_mask_mismatches"] += _compare_bool(
                legacy.latest_selected_mask,
                optimized.latest_selected_mask,
            )
            layer_report["topk_mask_mismatches"] += _compare_bool(
                legacy.latest_topk_mask,
                optimized.latest_topk_mask,
            )
            layer_report["rejected_conflict_mask_mismatches"] += _compare_bool(
                legacy.latest_rejected_conflict_mask,
                optimized.latest_rejected_conflict_mask,
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Compatibility correctness report written to: {output_path}")


if __name__ == "__main__":
    main()
