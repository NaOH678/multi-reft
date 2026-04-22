from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from pyreft import ComposableLoreftIntervention, ReftConfig, get_reft_model
except ImportError:
    from pyreft.pyreft import ComposableLoreftIntervention, ReftConfig, get_reft_model
from multi_train.eval_common.output_naming import normalize_base_model_name


_LAYER_FILE_RE = re.compile(r"intkey_layer_(\d+)_comp_block_output_unit_pos_nunit_1#0\.bin$")


def _resolve_intervenable_dir(path_value: str) -> Path:
    path = Path(os.path.normpath(str(path_value).rstrip("/")))
    if path.name == "intervenable_model":
        return path
    candidate = path / "intervenable_model"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"Could not resolve intervenable_model directory from: {path_value}")


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()) or "unknown"


def _format_float_tag(value: float) -> str:
    formatted = f"{float(value):.4g}"
    return formatted.replace("-", "m").replace(".", "p")


def normalize_specialist_label(weight_dir: str) -> str:
    path = _resolve_intervenable_dir(weight_dir)
    checkpoint = path.parent.name
    experiment = path.parent.parent.name if path.parent.parent != Path("") else path.parent.name
    return _sanitize_component(f"{experiment}-{checkpoint}")


def _compact_specialist_label(label: str) -> str:
    checkpoint_match = re.search(r"(?:^|-)checkpoint-(\d+)$", label)
    checkpoint = checkpoint_match.group(1) if checkpoint_match else None

    stem = re.sub(r"(?:^|-)checkpoint-\d+$", "", label)
    stem = re.sub(r"^Llama\d+(?:\.\d+)?-\d+[A-Za-z]*-", "", stem)
    stem = re.sub(r"^(?:Lo)?Reft_", "", stem)
    stem = re.sub(r"^Loreft_", "", stem)
    stem = stem.replace("Llama3_8b_", "")
    stem = stem.replace("Llama3-8b-", "")
    stem = _sanitize_component(stem)

    if checkpoint:
        return f"{stem}_{checkpoint}"
    return stem


def _compact_stats_tag(score_stats_path: str) -> str:
    stem = Path(score_stats_path).stem
    stem = re.sub(r"^residual_stats_", "", stem)
    replacements = {
        "shared": "sh",
        "specialist": "sp",
        "train": "tr",
        "input": "in",
        "residual": "res",
        "stats": "st",
    }
    parts = [_sanitize_component(part) for part in stem.split("_") if part.strip()]
    compact_parts = [replacements.get(part, part) for part in parts]
    compact = "_".join(compact_parts)
    return compact[:24]


def build_composable_model_tag(
    base_model_path: str,
    specialist_dirs: List[str],
    compose_domain: str,
    composition_method: str,
    composition_temperature: Optional[float] = None,
    composition_topk: Optional[int] = None,
    compat_threshold: Optional[float] = None,
    single_index: Optional[int] = None,
    shared_basis_type: Optional[str] = None,
    shared_basis_rank: Optional[int] = None,
    transport_type: Optional[str] = None,
    score_normalizer: Optional[str] = None,
    score_stats_path: Optional[str] = None,
    score_clip: Optional[float] = None,
) -> str:
    base_name = normalize_base_model_name(base_model_path)
    specialist_names = [normalize_specialist_label(path) for path in specialist_dirs]
    compact_specialists = [_compact_specialist_label(name) for name in specialist_names]
    joined = "+".join(compact_specialists[:4])
    if len(specialist_names) > 4:
        joined += f"+{len(specialist_names) - 4}more"
    effective_score_normalizer = _infer_score_normalizer(composition_method, score_normalizer)

    detail_parts = []
    if composition_method == "single" and single_index is not None:
        detail_parts.append(f"idx{single_index}")
    if composition_method in {"topk_residual", "compat_filtered_topk"} and composition_topk is not None:
        detail_parts.append(f"k{composition_topk}")
    if composition_method in {
        "residual_softmax",
        "residual_scaled_softmax",
        "residual_logz_softmax",
        "intervention_softmax",
        "topk_residual",
        "compat_filtered_topk",
    } and composition_temperature is not None:
        detail_parts.append(f"temp{_format_float_tag(composition_temperature)}")
    if composition_method == "compat_filtered_topk" and compat_threshold is not None:
        detail_parts.append(f"ct{_format_float_tag(compat_threshold)}")
    if effective_score_normalizer not in {"none", "mean_ratio", "log_zscore"}:
        detail_parts.append(f"norm_{_sanitize_component(effective_score_normalizer)}")
    if effective_score_normalizer != "none" and score_stats_path:
        stats_tag = _compact_stats_tag(score_stats_path)
        detail_parts.append(f"stats_{_sanitize_component(stats_tag)}")
    if score_clip is not None and effective_score_normalizer != "none":
        detail_parts.append(f"clip{_format_float_tag(score_clip)}")
    if compose_domain in {"projected_output", "shared_latent"}:
        if shared_basis_type:
            detail_parts.append(f"basis_{shared_basis_type}")
        if shared_basis_rank is not None:
            detail_parts.append(f"r{shared_basis_rank}")
    if compose_domain == "shared_latent" and transport_type:
        detail_parts.append(f"trans_{transport_type}")

    detail_suffix = ""
    if detail_parts:
        detail_suffix = "-" + "-".join(detail_parts)
    tag = f"{base_name}-composable-{compose_domain}-{composition_method}{detail_suffix}-{joined}"
    if len(tag) > 180:
        digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
        tag = f"{base_name}-composable-{compose_domain}-{composition_method}-{digest}"
    return tag


def load_loreft_specialist_state(weight_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    intervenable_dir = _resolve_intervenable_dir(weight_dir)
    layer_states: Dict[int, Dict[str, torch.Tensor]] = {}

    for weight_file in sorted(intervenable_dir.glob("intkey_layer_*_comp_block_output_unit_pos_nunit_1#0.bin")):
        match = _LAYER_FILE_RE.match(weight_file.name)
        if not match:
            continue
        layer = int(match.group(1))
        state = torch.load(weight_file, map_location="cpu")
        if not {"weight", "bias", "rotate_layer"}.issubset(set(state.keys())):
            raise ValueError(f"Unsupported Loreft checkpoint format in {weight_file}")
        layer_states[layer] = {
            "source_weight": state["weight"].detach().cpu(),
            "source_bias": state["bias"].detach().cpu(),
            "rotate_weight": state["rotate_layer"].detach().cpu(),
        }

    if not layer_states:
        raise FileNotFoundError(f"No layer intervention files found under: {intervenable_dir}")

    return layer_states


def build_shared_basis_for_layer(
    rotate_weights: List[torch.Tensor],
    basis_type: str,
    basis_rank: int,
) -> torch.Tensor:
    if not rotate_weights:
        raise ValueError("rotate_weights must be non-empty.")
    if basis_rank <= 0:
        raise ValueError("basis_rank must be positive.")

    matrix_dtype = torch.float32
    rotate_weights = [weight.to(matrix_dtype) for weight in rotate_weights]
    hidden_size = rotate_weights[0].shape[0]

    if basis_type == "orth_mean":
        mean_basis = torch.stack(rotate_weights, dim=0).mean(dim=0)
        q, _ = torch.linalg.qr(mean_basis, mode="reduced")
        if basis_rank > q.shape[1]:
            raise ValueError(f"orth_mean basis_rank={basis_rank} exceeds available rank {q.shape[1]}.")
        return q[:, :basis_rank].T.contiguous()

    if basis_type == "svd_union":
        union_basis = torch.cat(rotate_weights, dim=1)
        u, _, _ = torch.linalg.svd(union_basis, full_matrices=False)
        if basis_rank > u.shape[1]:
            raise ValueError(f"svd_union basis_rank={basis_rank} exceeds available rank {u.shape[1]}.")
        return u[:, :basis_rank].T.contiguous()

    raise ValueError(f"Unsupported shared basis type: {basis_type}")


def build_transport_for_layer(
    rotate_weights: List[torch.Tensor],
    shared_basis: torch.Tensor,
    transport_type: str,
) -> torch.Tensor:
    if transport_type == "identity":
        rank = rotate_weights[0].shape[1]
        if shared_basis.shape[0] != rank:
            raise ValueError("identity transport requires shared_basis_rank == specialist rank.")
        eye = torch.eye(rank, dtype=torch.float32)
        return torch.stack([eye.clone() for _ in rotate_weights], dim=0)

    if transport_type == "overlap":
        transported = []
        for weight in rotate_weights:
            transported.append(shared_basis.float() @ weight.float())
        return torch.stack(transported, dim=0)

    raise ValueError(f"Unsupported transport_type: {transport_type}")


def _collect_target_layers(
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    target_layers: Optional[List[int]],
) -> List[int]:
    common_layers = set(specialist_states[0].keys())
    for state in specialist_states[1:]:
        common_layers &= set(state.keys())
    if not common_layers:
        raise ValueError("No common target layers found across specialists.")

    if not target_layers or target_layers == [-1]:
        return sorted(common_layers)

    missing = [layer for layer in target_layers if layer not in common_layers]
    if missing:
        raise ValueError(f"Requested target layers are missing from some specialists: {missing}")
    return list(target_layers)


def _validate_layer_shapes(
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    target_layers: List[int],
) -> None:
    for layer in target_layers:
        rotate_shapes = [tuple(state[layer]["rotate_weight"].shape) for state in specialist_states]
        source_weight_shapes = [tuple(state[layer]["source_weight"].shape) for state in specialist_states]
        source_bias_shapes = [tuple(state[layer]["source_bias"].shape) for state in specialist_states]
        if len(set(rotate_shapes)) != 1:
            raise ValueError(f"Inconsistent rotate shapes at layer {layer}: {rotate_shapes}")
        if len(set(source_weight_shapes)) != 1:
            raise ValueError(f"Inconsistent source_weight shapes at layer {layer}: {source_weight_shapes}")
        if len(set(source_bias_shapes)) != 1:
            raise ValueError(f"Inconsistent source_bias shapes at layer {layer}: {source_bias_shapes}")


def _infer_score_normalizer(
    composition_method: str,
    score_normalizer: Optional[str],
) -> str:
    if composition_method == "residual_scaled_softmax":
        if score_normalizer in {None, "", "none", "mean_ratio"}:
            return "mean_ratio"
        raise ValueError(
            f"composition_method={composition_method} requires score_normalizer=mean_ratio, got {score_normalizer}."
        )
    if composition_method == "residual_logz_softmax":
        if score_normalizer in {None, "", "none", "log_zscore"}:
            return "log_zscore"
        raise ValueError(
            f"composition_method={composition_method} requires score_normalizer=log_zscore, got {score_normalizer}."
        )
    return score_normalizer or "none"


def load_residual_stats(score_stats_path: Optional[str]) -> Optional[Dict]:
    if not score_stats_path:
        return None
    with open(score_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_layer_score_stats(
    residual_stats: Dict,
    layer: int,
    specialist_labels: List[str],
) -> Dict[str, torch.Tensor]:
    layers = residual_stats.get("layers", {})
    layer_stats = layers.get(str(layer))
    if layer_stats is None:
        raise ValueError(f"Missing residual stats for layer {layer}.")

    collected = {
        "mean": [],
        "std": [],
        "log_mean": [],
        "log_std": [],
    }

    for label in specialist_labels:
        specialist_stats = layer_stats.get(label)
        if specialist_stats is None:
            raise ValueError(f"Missing residual stats for layer={layer}, specialist={label}.")
        for key in collected:
            if key not in specialist_stats:
                raise ValueError(f"Missing residual stat `{key}` for layer={layer}, specialist={label}.")
            collected[key].append(float(specialist_stats[key]))

    return {
        key: torch.tensor(values, dtype=torch.float32)
        for key, values in collected.items()
    }


def build_composable_reft_config(
    model,
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    specialist_labels: List[str],
    target_layers: List[int],
    compose_domain: str,
    composition_method: str,
    composition_temperature: float,
    composition_topk: Optional[int],
    compat_threshold: float,
    single_index: Optional[int],
    shared_basis_type: Optional[str],
    shared_basis_rank: Optional[int],
    transport_type: Optional[str],
    score_normalizer: Optional[str] = None,
    residual_stats: Optional[Dict] = None,
    score_eps: float = 1e-6,
    score_clip: Optional[float] = None,
    use_trainable_policy: bool = False,
    policy_hidden_dim: int = 32,
):
    representations = []
    specialist_rank = specialist_states[0][target_layers[0]]["source_weight"].shape[0]
    basis_rank = shared_basis_rank if shared_basis_rank is not None else specialist_rank
    needs_shared_basis = compose_domain in {"projected_output", "shared_latent"}
    if compose_domain == "shared_latent" and transport_type is None:
        transport_type = "identity"
    normalized_score_mode = _infer_score_normalizer(composition_method, score_normalizer)

    for layer in target_layers:
        layer_rotate = [state[layer]["rotate_weight"] for state in specialist_states]
        layer_source_weight = [state[layer]["source_weight"] for state in specialist_states]
        layer_source_bias = [state[layer]["source_bias"] for state in specialist_states]

        intervention_kwargs = {
            "embed_dim": model.config.hidden_size,
            "low_rank_dimension": specialist_rank,
            "num_specialists": len(specialist_states),
            "compose_domain": compose_domain,
            "policy_type": composition_method,
            "temperature": composition_temperature,
            "topk": composition_topk,
            "compat_threshold": compat_threshold,
            "single_index": single_index,
            "shared_basis_type": shared_basis_type,
            "shared_basis_rank": basis_rank,
            "transport_type": transport_type,
            "score_normalizer": normalized_score_mode,
            "score_eps": score_eps,
            "score_clip": score_clip,
            "use_trainable_policy": use_trainable_policy,
            "policy_hidden_dim": policy_hidden_dim,
            "rotate_weight": torch.stack(layer_rotate, dim=0),
            "source_weight": torch.stack(layer_source_weight, dim=0),
            "source_bias": torch.stack(layer_source_bias, dim=0),
            "add_bias": False,
        }
        if normalized_score_mode != "none":
            if residual_stats is None:
                raise ValueError(
                    f"composition_method={composition_method} requires score_stats_path / residual stats."
                )
            intervention_kwargs["score_stats"] = _build_layer_score_stats(
                residual_stats=residual_stats,
                layer=layer,
                specialist_labels=specialist_labels,
            )

        if needs_shared_basis:
            if not shared_basis_type:
                raise ValueError(f"{compose_domain} composition requires shared_basis_type.")
            shared_basis = build_shared_basis_for_layer(layer_rotate, shared_basis_type, basis_rank)
            intervention_kwargs["shared_basis"] = shared_basis
            if compose_domain == "shared_latent":
                intervention_kwargs["transport_weight"] = build_transport_for_layer(
                    layer_rotate,
                    shared_basis,
                    transport_type or "identity",
                )

        representations.append(
            {
                "layer": layer,
                "component": "block_output",
                "intervention": ComposableLoreftIntervention(**intervention_kwargs),
            }
        )

    return ReftConfig(representations=representations)


def load_composed_reft_model(
    model,
    specialist_dirs: List[str],
    target_layers: Optional[List[int]],
    compose_domain: str,
    composition_method: str,
    composition_temperature: float,
    composition_topk: Optional[int],
    compat_threshold: float,
    single_index: Optional[int],
    shared_basis_type: Optional[str],
    shared_basis_rank: Optional[int],
    transport_type: Optional[str],
    score_normalizer: Optional[str] = None,
    score_stats_path: Optional[str] = None,
    score_eps: float = 1e-6,
    score_clip: Optional[float] = None,
    use_trainable_policy: bool = False,
    policy_hidden_dim: int = 32,
):
    specialist_states = [load_loreft_specialist_state(path) for path in specialist_dirs]
    specialist_labels = [normalize_specialist_label(path) for path in specialist_dirs]
    resolved_layers = _collect_target_layers(specialist_states, target_layers)
    _validate_layer_shapes(specialist_states, resolved_layers)
    residual_stats = load_residual_stats(score_stats_path)
    if residual_stats is not None:
        stats_labels = residual_stats.get("specialist_labels", [])
        missing = [label for label in specialist_labels if label not in stats_labels]
        if missing:
            raise ValueError(f"Residual stats file missing specialist labels: {missing}")
    reft_config = build_composable_reft_config(
        model=model,
        specialist_states=specialist_states,
        specialist_labels=specialist_labels,
        target_layers=resolved_layers,
        compose_domain=compose_domain,
        composition_method=composition_method,
        composition_temperature=composition_temperature,
        composition_topk=composition_topk,
        compat_threshold=compat_threshold,
        single_index=single_index,
        shared_basis_type=shared_basis_type,
        shared_basis_rank=shared_basis_rank,
        transport_type=transport_type,
        score_normalizer=score_normalizer,
        residual_stats=residual_stats,
        score_eps=score_eps,
        score_clip=score_clip,
        use_trainable_policy=use_trainable_policy,
        policy_hidden_dim=policy_hidden_dim,
    )
    return get_reft_model(model, reft_config)
