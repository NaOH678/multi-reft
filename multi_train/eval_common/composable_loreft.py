from __future__ import annotations

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    from pyreft import ComposableLoreftIntervention, ReftConfig, get_reft_model
except ImportError:
    from pyreft.pyreft import ComposableLoreftIntervention, ReftConfig, get_reft_model
from multi_train.eval_common.output_naming import normalize_base_model_name


_LAYER_FILE_RE = re.compile(r"intkey_layer_(\d+)_comp_block_output_unit_pos_nunit_1#0\.bin$")
_ROUTER_SCALAR_FEATURE_NAMES = frozenset({
    "rh_log_norm",
    "delta_log_norm",
    "delta_norm",
    "intervention_norm",
    "subspace_energy",
    "mean_compat",
    "neg_compat_mass",
})


def _is_checkpoint_like(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return (
        normalized.startswith("checkpoint-")
        or normalized.startswith("chpoint-")
        or normalized.startswith("ckpt-")
    )


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


def _resolve_router_model_dir(path_value: str) -> Path:
    path = Path(os.path.normpath(str(path_value).rstrip("/")))
    if path.name == "intervenable_model":
        return path.parent
    if (path / "intervenable_model").is_dir():
        return path
    raise FileNotFoundError(f"Could not resolve router checkpoint directory from: {path_value}")


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


def _compact_score_source_tag(score_source: str) -> str:
    mapping = {
        "delta_norm": "dn",
        "intervention_norm": "in",
    }
    return mapping.get(score_source, _sanitize_component(score_source))


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
    score_source: Optional[str] = None,
    score_normalizer: Optional[str] = None,
    score_stats_path: Optional[str] = None,
    score_clip: Optional[float] = None,
    truthful_score_penalty: float = 0.0,
) -> str:
    base_name = normalize_base_model_name(base_model_path)
    specialist_names = [normalize_specialist_label(path) for path in specialist_dirs]
    compact_specialists = [_compact_specialist_label(name) for name in specialist_names]
    joined = "+".join(compact_specialists[:4])
    if len(specialist_names) > 4:
        joined += f"+{len(specialist_names) - 4}more"
    effective_score_normalizer = _infer_score_normalizer(composition_method, score_normalizer)
    effective_score_source = _infer_score_source(composition_method, score_source)

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
    if composition_method in {"intervention_softmax", "compat_filtered_topk"}:
        detail_parts.append(f"src_{_compact_score_source_tag(effective_score_source)}")
    if composition_method == "compat_filtered_topk" and compat_threshold is not None:
        detail_parts.append(f"ct{_format_float_tag(compat_threshold)}")
    if effective_score_normalizer not in {"none", "mean_ratio", "log_zscore"}:
        detail_parts.append(f"norm_{_sanitize_component(effective_score_normalizer)}")
    if effective_score_normalizer != "none" and score_stats_path:
        stats_tag = _compact_stats_tag(score_stats_path)
        detail_parts.append(f"stats_{_sanitize_component(stats_tag)}")
    if score_clip is not None and effective_score_normalizer != "none":
        detail_parts.append(f"clip{_format_float_tag(score_clip)}")
    if abs(float(truthful_score_penalty)) > 0:
        detail_parts.append(f"tpen{_format_float_tag(truthful_score_penalty)}")
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


def build_trained_router_model_tag(
    base_model_path: str,
    router_checkpoint_dir: str,
) -> str:
    router_dir = _resolve_router_model_dir(router_checkpoint_dir)
    if _is_checkpoint_like(router_dir.name) and router_dir.parent.name:
        return _sanitize_component(router_dir.parent.name)
    return _sanitize_component(router_dir.name)


def _collect_intervention_layers(weight_dir: str) -> List[int]:
    intervenable_dir = _resolve_intervenable_dir(weight_dir)
    layers = []
    for weight_file in sorted(intervenable_dir.glob("intkey_layer_*_comp_block_output_unit_pos_nunit_1#0.bin")):
        match = _LAYER_FILE_RE.match(weight_file.name)
        if not match:
            continue
        layers.append(int(match.group(1)))
    if not layers:
        raise FileNotFoundError(f"No layer intervention files found under: {intervenable_dir}")
    return sorted(layers)


def _load_normalized_intervention_layers(weight_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    intervenable_dir = _resolve_intervenable_dir(weight_dir)
    layer_states: Dict[int, Dict[str, torch.Tensor]] = {}

    for weight_file in sorted(intervenable_dir.glob("intkey_layer_*_comp_block_output_unit_pos_nunit_1#0.bin")):
        match = _LAYER_FILE_RE.match(weight_file.name)
        if not match:
            continue
        layer = int(match.group(1))
        raw_state = torch.load(weight_file, map_location="cpu")

        if {"weight", "bias", "rotate_layer"}.issubset(set(raw_state.keys())):
            normalized_state = {
                "source_weight": raw_state["weight"].detach().cpu(),
                "source_bias": raw_state["bias"].detach().cpu(),
                "rotate_weight": raw_state["rotate_layer"].detach().cpu(),
            }
        elif {"source_weight", "source_bias", "rotate_weight"}.issubset(set(raw_state.keys())):
            normalized_state = {
                "source_weight": raw_state["source_weight"].detach().cpu(),
                "source_bias": raw_state["source_bias"].detach().cpu(),
                "rotate_weight": raw_state["rotate_weight"].detach().cpu(),
            }
            if "shared_basis" in raw_state:
                normalized_state["shared_basis"] = raw_state["shared_basis"].detach().cpu()
            if "transport_weight" in raw_state:
                normalized_state["transport_weight"] = raw_state["transport_weight"].detach().cpu()
        else:
            raise ValueError(f"Unsupported intervention checkpoint format in {weight_file}")

        layer_states[layer] = normalized_state

    if not layer_states:
        raise FileNotFoundError(f"No layer intervention files found under: {intervenable_dir}")

    return layer_states


def load_loreft_specialist_state(weight_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    layer_states = _load_normalized_intervention_layers(weight_dir)
    for layer, state in layer_states.items():
        if state["source_weight"].dim() != 2 or state["source_bias"].dim() != 1 or state["rotate_weight"].dim() != 2:
            raise ValueError(f"Expected single-specialist Loreft state for layer {layer} under {weight_dir}.")
    return layer_states


def load_composable_intervention_state(weight_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    layer_states = _load_normalized_intervention_layers(weight_dir)
    for layer, state in layer_states.items():
        if state["source_weight"].dim() != 3 or state["source_bias"].dim() != 2 or state["rotate_weight"].dim() != 3:
            raise ValueError(f"Expected composable intervention state for layer {layer} under {weight_dir}.")
    return layer_states


def _load_raw_intervention_checkpoint(weight_dir: str, layer: int) -> Dict[str, Any]:
    intervenable_dir = _resolve_intervenable_dir(weight_dir)
    checkpoint_path = intervenable_dir / f"intkey_layer_{int(layer)}_comp_block_output_unit_pos_nunit_1#0.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing intervention checkpoint for layer {layer}: {checkpoint_path}")
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw_state, dict):
        raise ValueError(f"Unsupported intervention checkpoint payload at {checkpoint_path}")
    return raw_state


def _infer_trainable_policy_dims_from_checkpoint(
    weight_dir: str,
    candidate_layers: List[int],
) -> Dict[str, int]:
    for layer in candidate_layers:
        raw_state = _load_raw_intervention_checkpoint(weight_dir, int(layer))
        hidden_dim = None
        projection_dim = None
        pre_hidden_state_dim = None

        if "policy_head.0.weight" in raw_state:
            hidden_dim = int(raw_state["policy_head.0.weight"].shape[0])
        if "specialist_proj_weight" in raw_state:
            projection_dim = int(raw_state["specialist_proj_weight"].shape[1])
        elif "specialist_proj_bias" in raw_state:
            projection_dim = int(raw_state["specialist_proj_bias"].shape[1])
        if "pre_hidden_proj.weight" in raw_state:
            pre_hidden_state_dim = int(raw_state["pre_hidden_proj.weight"].shape[0])

        if hidden_dim is not None or projection_dim is not None or pre_hidden_state_dim is not None:
            inferred = {}
            if hidden_dim is not None:
                inferred["router_hidden_dim"] = hidden_dim
            if projection_dim is not None:
                inferred["router_projection_dim"] = projection_dim
            if pre_hidden_state_dim is not None:
                inferred["pre_hidden_state_dim"] = pre_hidden_state_dim
            return inferred
    return {}


def load_router_training_metadata(
    router_checkpoint_dir: str,
    router_metadata_path: Optional[str] = None,
) -> Dict[str, Any]:
    if router_metadata_path:
        metadata_path = Path(router_metadata_path)
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    router_model_dir = _resolve_router_model_dir(router_checkpoint_dir)
    candidate_paths = [
        router_model_dir / "router_training_metadata.json",
        router_model_dir.parent / "router_training_metadata.json",
    ]
    for metadata_path in candidate_paths:
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)

    tried = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        "Could not find router_training_metadata.json. "
        f"Tried: {tried}. "
        "You can also pass --router_metadata_path explicitly."
    )


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


def _infer_score_source(
    composition_method: str,
    score_source: Optional[str],
) -> str:
    if composition_method in {
        "residual_softmax",
        "residual_scaled_softmax",
        "residual_logz_softmax",
        "topk_residual",
    }:
        if score_source in {None, "", "delta_norm"}:
            return "delta_norm"
        raise ValueError(
            f"composition_method={composition_method} requires score_source=delta_norm, got {score_source}."
        )
    if composition_method in {"intervention_softmax", "compat_filtered_topk"}:
        if score_source in {None, ""}:
            if composition_method == "intervention_softmax":
                return "intervention_norm"
            return "delta_norm"
        if score_source not in {"delta_norm", "intervention_norm"}:
            raise ValueError(f"Unsupported score_source: {score_source}")
        return score_source
    return "delta_norm"


def load_score_stats(score_stats_path: Optional[str]) -> Optional[Dict]:
    if not score_stats_path:
        return None
    with open(score_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_unique_specialist_index(specialist_labels: List[str], keyword: str) -> int:
    matches = [idx for idx, label in enumerate(specialist_labels) if keyword in str(label).lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(f"Could not find specialist label containing `{keyword}` in {specialist_labels}.")
    raise ValueError(f"Found multiple specialist labels containing `{keyword}`: {specialist_labels}")


def _build_specialist_score_bias(
    specialist_labels: List[str],
    truthful_score_penalty: float,
) -> Optional[torch.Tensor]:
    truthful_score_penalty = float(truthful_score_penalty)
    if abs(truthful_score_penalty) <= 0:
        return None
    truthful_idx = _resolve_unique_specialist_index(specialist_labels, "truthful")
    bias = torch.zeros(len(specialist_labels), dtype=torch.float32)
    bias[truthful_idx] = -truthful_score_penalty
    return bias


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


def _build_layer_router_feature_stats(
    router_feature_stats: Dict,
    layer: int,
    specialist_labels: List[str],
    feature_names: List[str],
) -> Dict[str, torch.Tensor]:
    layers = router_feature_stats.get("layers", {})
    layer_stats = layers.get(str(layer))
    if layer_stats is None:
        raise ValueError(f"Missing router feature stats for layer {layer}.")

    mean_rows = []
    std_rows = []
    for label in specialist_labels:
        specialist_stats = layer_stats.get(label)
        if specialist_stats is None:
            raise ValueError(f"Missing router feature stats for layer={layer}, specialist={label}.")
        mean_values = []
        std_values = []
        for feature_name in feature_names:
            feature_stats = specialist_stats.get(feature_name)
            if feature_stats is None:
                raise ValueError(
                    f"Missing router feature stat `{feature_name}` for layer={layer}, specialist={label}."
                )
            mean_values.append(float(feature_stats["mean"]))
            std_values.append(float(feature_stats["std"]))
        mean_rows.append(mean_values)
        std_rows.append(std_values)

    return {
        "feature_names": list(feature_names),
        "mean": torch.tensor(mean_rows, dtype=torch.float32),
        "std": torch.tensor(std_rows, dtype=torch.float32),
    }


def _maybe_load_json_dict(value: Optional[Any]) -> Optional[Dict]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, Path)):
        if not str(value).strip():
            return None
        with open(value, "r", encoding="utf-8") as f:
            return json.load(f)
    raise TypeError(f"Unsupported stats/config value type: {type(value)}")


def _get_layer_override(layer_policy_specs: Optional[Dict[Any, Dict]], layer: int) -> Dict:
    if not layer_policy_specs:
        return {}
    if layer in layer_policy_specs:
        return dict(layer_policy_specs[layer])
    if str(layer) in layer_policy_specs:
        return dict(layer_policy_specs[str(layer)])
    return {}


def _build_mixed_layer_intervention_kwargs(
    model,
    layer: int,
    specialist_labels: List[str],
    layer_spec: Dict[str, Any],
    layer_state: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    compose_domain = layer_spec.get("compose_domain", "output")
    composition_method = layer_spec.get("composition_method", "equal")
    composition_temperature = float(layer_spec.get("composition_temperature", 1.0))
    composition_topk = layer_spec.get("composition_topk")
    compat_threshold = float(layer_spec.get("compat_threshold", 0.0))
    single_index = layer_spec.get("single_index")
    shared_basis_type = layer_spec.get("shared_basis_type")
    shared_basis_rank = layer_spec.get("shared_basis_rank")
    transport_type = layer_spec.get("transport_type")
    score_source = layer_spec.get("score_source")
    score_normalizer = layer_spec.get("score_normalizer")
    score_eps = float(layer_spec.get("score_eps", 1e-6))
    score_clip = layer_spec.get("score_clip")
    use_trainable_policy = bool(layer_spec.get("use_trainable_policy", False))
    policy_hidden_dim = int(layer_spec.get("policy_hidden_dim", 32))
    policy_projection_dim = int(layer_spec.get("policy_projection_dim", 32))
    use_pre_hidden_state_feature = bool(layer_spec.get("use_pre_hidden_state_feature", False))
    pre_hidden_state_dim = int(layer_spec.get("pre_hidden_state_dim", 64))
    enable_debug_cache = bool(layer_spec.get("enable_debug_cache", False))
    enable_monitor_cache = bool(layer_spec.get("enable_monitor_cache", False))
    compat_impl = layer_spec.get("compat_impl", "optimized")
    router_feature_names = list(layer_spec.get("router_feature_names", []))
    router_feature_eps = float(layer_spec.get("router_feature_eps", 1e-6))
    router_feature_clip = layer_spec.get("router_feature_clip", 5.0)
    dropout = float(layer_spec.get("dropout", 0.0))
    scalar_router_feature_names = [name for name in router_feature_names if name in _ROUTER_SCALAR_FEATURE_NAMES]

    specialist_rank = int(layer_state["source_weight"].shape[1])
    num_specialists = int(layer_state["source_weight"].shape[0])
    basis_rank = shared_basis_rank if shared_basis_rank is not None else specialist_rank
    needs_shared_basis = compose_domain in {"projected_output", "shared_latent"}
    if compose_domain == "shared_latent" and transport_type is None and "transport_weight" not in layer_state:
        transport_type = "identity"

    normalized_score_mode = _infer_score_normalizer(composition_method, score_normalizer)
    effective_score_source = _infer_score_source(composition_method, score_source)
    raw_score_stats = _maybe_load_json_dict(layer_spec.get("score_stats") or layer_spec.get("score_stats_path"))
    raw_router_feature_stats = _maybe_load_json_dict(
        layer_spec.get("router_feature_stats") or layer_spec.get("router_feature_stats_path")
    )
    if raw_score_stats is not None:
        stats_labels = raw_score_stats.get("specialist_labels", [])
        missing = [label for label in specialist_labels if label not in stats_labels]
        if missing:
            raise ValueError(f"Score stats file missing specialist labels: {missing}")
        stats_score_source = raw_score_stats.get("score_source")
        if stats_score_source is not None and stats_score_source != effective_score_source:
            raise ValueError(
                "Score stats file source mismatch: "
                f"expected {effective_score_source}, got {stats_score_source}."
            )
    if raw_router_feature_stats is not None:
        stats_labels = raw_router_feature_stats.get("specialist_labels", [])
        missing = [label for label in specialist_labels if label not in stats_labels]
        if missing:
            raise ValueError(f"Router feature stats file missing specialist labels: {missing}")

    intervention_kwargs = {
        "embed_dim": model.config.hidden_size,
        "low_rank_dimension": specialist_rank,
        "num_specialists": num_specialists,
        "compose_domain": compose_domain,
        "policy_type": composition_method,
        "temperature": composition_temperature,
        "topk": composition_topk,
        "compat_threshold": compat_threshold,
        "single_index": single_index,
        "shared_basis_type": shared_basis_type,
        "shared_basis_rank": basis_rank,
        "transport_type": transport_type,
        "score_source": effective_score_source,
        "score_stats_source": raw_score_stats.get("score_source") if raw_score_stats is not None else None,
        "score_normalizer": normalized_score_mode,
        "score_eps": score_eps,
        "score_clip": score_clip,
        "use_trainable_policy": use_trainable_policy,
        "policy_hidden_dim": policy_hidden_dim,
        "policy_projection_dim": policy_projection_dim,
        "use_pre_hidden_state_feature": use_pre_hidden_state_feature,
        "pre_hidden_state_dim": pre_hidden_state_dim,
        "dropout": dropout,
        "enable_debug_cache": enable_debug_cache,
        "enable_monitor_cache": enable_monitor_cache,
        "compat_impl": compat_impl,
        "router_feature_names": router_feature_names,
        "router_feature_eps": router_feature_eps,
        "router_feature_clip": router_feature_clip,
        "rotate_weight": layer_state["rotate_weight"],
        "source_weight": layer_state["source_weight"],
        "source_bias": layer_state["source_bias"],
        "add_bias": False,
    }

    if normalized_score_mode != "none":
        if raw_score_stats is None:
            raise ValueError(
                f"composition_method={composition_method} requires score_stats / score_stats_path."
            )
        intervention_kwargs["score_stats"] = _build_layer_score_stats(
            residual_stats=raw_score_stats,
            layer=layer,
            specialist_labels=specialist_labels,
        )

    if use_trainable_policy and raw_router_feature_stats is not None and scalar_router_feature_names:
        intervention_kwargs["router_feature_stats"] = _build_layer_router_feature_stats(
            router_feature_stats=raw_router_feature_stats,
            layer=layer,
            specialist_labels=specialist_labels,
            feature_names=scalar_router_feature_names,
        )

    if needs_shared_basis:
        shared_basis = layer_state.get("shared_basis")
        if shared_basis is None:
            if not shared_basis_type:
                raise ValueError(f"{compose_domain} composition requires shared_basis_type.")
            rotate_weights = [weight.detach().cpu() for weight in layer_state["rotate_weight"]]
            shared_basis = build_shared_basis_for_layer(rotate_weights, shared_basis_type, basis_rank)
        intervention_kwargs["shared_basis"] = shared_basis
        if compose_domain == "shared_latent":
            transport_weight = layer_state.get("transport_weight")
            if transport_weight is None:
                rotate_weights = [weight.detach().cpu() for weight in layer_state["rotate_weight"]]
                transport_weight = build_transport_for_layer(
                    rotate_weights,
                    shared_basis,
                    transport_type or "identity",
                )
            intervention_kwargs["transport_weight"] = transport_weight

    return intervention_kwargs


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
    score_source: Optional[str] = None,
    score_normalizer: Optional[str] = None,
    score_stats: Optional[Dict] = None,
    score_eps: float = 1e-6,
    score_clip: Optional[float] = None,
    truthful_score_penalty: float = 0.0,
    use_trainable_policy: bool = False,
    policy_hidden_dim: int = 32,
    policy_projection_dim: int = 32,
    use_pre_hidden_state_feature: bool = False,
    pre_hidden_state_dim: int = 64,
    dropout: float = 0.0,
):
    representations = []
    specialist_rank = specialist_states[0][target_layers[0]]["source_weight"].shape[0]
    basis_rank = shared_basis_rank if shared_basis_rank is not None else specialist_rank
    needs_shared_basis = compose_domain in {"projected_output", "shared_latent"}
    if compose_domain == "shared_latent" and transport_type is None:
        transport_type = "identity"
    normalized_score_mode = _infer_score_normalizer(composition_method, score_normalizer)
    effective_score_source = _infer_score_source(composition_method, score_source)
    specialist_score_bias = _build_specialist_score_bias(specialist_labels, truthful_score_penalty)

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
            "score_source": effective_score_source,
            "score_stats_source": score_stats.get("score_source") if score_stats is not None else None,
            "score_normalizer": normalized_score_mode,
            "score_eps": score_eps,
            "score_clip": score_clip,
            "specialist_score_bias": specialist_score_bias,
            "use_trainable_policy": use_trainable_policy,
            "policy_hidden_dim": policy_hidden_dim,
            "policy_projection_dim": policy_projection_dim,
            "use_pre_hidden_state_feature": use_pre_hidden_state_feature,
            "pre_hidden_state_dim": pre_hidden_state_dim,
            "dropout": dropout,
            "rotate_weight": torch.stack(layer_rotate, dim=0),
            "source_weight": torch.stack(layer_source_weight, dim=0),
            "source_bias": torch.stack(layer_source_bias, dim=0),
            "add_bias": False,
        }
        if normalized_score_mode != "none":
            if score_stats is None:
                raise ValueError(
                    f"composition_method={composition_method} requires score_stats_path / residual stats."
                )
            intervention_kwargs["score_stats"] = _build_layer_score_stats(
                residual_stats=score_stats,
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


def build_mixed_composable_reft_config(
    model,
    specialist_states: List[Dict[int, Dict[str, torch.Tensor]]],
    specialist_labels: List[str],
    target_layers: List[int],
    default_layer_spec: Dict[str, Any],
    layer_policy_specs: Optional[Dict[Any, Dict[str, Any]]] = None,
):
    representations = []

    for layer in target_layers:
        stacked_layer_state = {
            "rotate_weight": torch.stack([state[layer]["rotate_weight"] for state in specialist_states], dim=0),
            "source_weight": torch.stack([state[layer]["source_weight"] for state in specialist_states], dim=0),
            "source_bias": torch.stack([state[layer]["source_bias"] for state in specialist_states], dim=0),
        }
        layer_spec = dict(default_layer_spec)
        layer_spec.update(_get_layer_override(layer_policy_specs, layer))
        intervention_kwargs = _build_mixed_layer_intervention_kwargs(
            model=model,
            layer=layer,
            specialist_labels=specialist_labels,
            layer_spec=layer_spec,
            layer_state=stacked_layer_state,
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
    score_source: Optional[str] = None,
    score_normalizer: Optional[str] = None,
    score_stats_path: Optional[str] = None,
    score_eps: float = 1e-6,
    score_clip: Optional[float] = None,
    truthful_score_penalty: float = 0.0,
    use_trainable_policy: bool = False,
    policy_hidden_dim: int = 32,
    policy_projection_dim: int = 32,
    use_pre_hidden_state_feature: bool = False,
    pre_hidden_state_dim: int = 64,
    dropout: float = 0.0,
):
    specialist_states = [load_loreft_specialist_state(path) for path in specialist_dirs]
    specialist_labels = [normalize_specialist_label(path) for path in specialist_dirs]
    resolved_layers = _collect_target_layers(specialist_states, target_layers)
    _validate_layer_shapes(specialist_states, resolved_layers)
    effective_score_source = _infer_score_source(composition_method, score_source)
    score_stats = load_score_stats(score_stats_path)
    if score_stats is not None:
        stats_labels = score_stats.get("specialist_labels", [])
        missing = [label for label in specialist_labels if label not in stats_labels]
        if missing:
            raise ValueError(f"Score stats file missing specialist labels: {missing}")
        stats_score_source = score_stats.get("score_source")
        if stats_score_source is not None and stats_score_source != effective_score_source:
            raise ValueError(
                "Score stats file source mismatch: "
                f"expected {effective_score_source}, got {stats_score_source}."
            )
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
        score_source=effective_score_source,
        score_normalizer=score_normalizer,
        score_stats=score_stats,
        score_eps=score_eps,
        score_clip=score_clip,
        truthful_score_penalty=truthful_score_penalty,
        use_trainable_policy=use_trainable_policy,
        policy_hidden_dim=policy_hidden_dim,
        policy_projection_dim=policy_projection_dim,
        use_pre_hidden_state_feature=use_pre_hidden_state_feature,
        pre_hidden_state_dim=pre_hidden_state_dim,
        dropout=dropout,
    )
    return get_reft_model(model, reft_config)


def load_mixed_composed_reft_model(
    model,
    specialist_dirs: List[str],
    target_layers: Optional[List[int]],
    default_layer_spec: Dict[str, Any],
    layer_policy_specs: Optional[Dict[Any, Dict[str, Any]]] = None,
):
    specialist_states = [load_loreft_specialist_state(path) for path in specialist_dirs]
    specialist_labels = [normalize_specialist_label(path) for path in specialist_dirs]
    resolved_layers = _collect_target_layers(specialist_states, target_layers)
    _validate_layer_shapes(specialist_states, resolved_layers)
    reft_config = build_mixed_composable_reft_config(
        model=model,
        specialist_states=specialist_states,
        specialist_labels=specialist_labels,
        target_layers=resolved_layers,
        default_layer_spec=default_layer_spec,
        layer_policy_specs=layer_policy_specs,
    )
    return get_reft_model(model, reft_config)


def _normalize_optional_path(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _coerce_int_list(value: Optional[Any]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(item) for item in value]


def _build_trained_router_layer_specs(
    metadata: Dict[str, Any],
    base_score_stats_path: Optional[str] = None,
    router_feature_stats_path: Optional[str] = None,
    enable_debug_cache: bool = False,
) -> tuple[Dict[str, Any], Dict[int, Dict[str, Any]]]:
    compose_domain = str(metadata.get("compose_domain", "output"))
    router_layer_policy = str(metadata.get("router_layer_policy", "trainable"))
    router_feature_names = list(metadata.get("router_feature_names") or [])
    resolved_base_score_stats_path = (
        _normalize_optional_path(base_score_stats_path)
        if base_score_stats_path is not None
        else _normalize_optional_path(metadata.get("base_score_stats_path"))
    )
    resolved_router_feature_stats_path = (
        _normalize_optional_path(router_feature_stats_path)
        if router_feature_stats_path is not None
        else _normalize_optional_path(metadata.get("router_feature_stats_path"))
    )
    dropout = float(metadata.get("dropout", 0.0))

    default_layer_spec = {
        "compose_domain": compose_domain,
        "composition_method": str(metadata.get("base_composition_method", "compat_filtered_topk")),
        "composition_temperature": float(metadata.get("base_composition_temperature", 1.0)),
        "composition_topk": metadata.get("base_composition_topk", 2),
        "compat_threshold": float(metadata.get("base_compat_threshold", 0.0)),
        "score_source": str(metadata.get("base_score_source", "intervention_norm")),
        "score_normalizer": str(metadata.get("base_score_normalizer", "log_zscore")),
        "score_stats_path": resolved_base_score_stats_path,
        "compat_impl": str(metadata.get("compat_impl", "optimized")),
        "dropout": dropout,
        "enable_debug_cache": bool(enable_debug_cache),
    }

    if router_layer_policy == "trainable":
        router_layer_spec = {
            "compose_domain": compose_domain,
            "composition_method": "trainable",
            "composition_temperature": float(metadata.get("router_temperature", 1.5)),
            "score_source": str(metadata.get("base_score_source", "intervention_norm")),
            "score_normalizer": "none",
            "score_stats_path": None,
            "use_trainable_policy": True,
            "policy_hidden_dim": int(metadata.get("router_hidden_dim", metadata.get("policy_hidden_dim", 32))),
            "policy_projection_dim": int(
                metadata.get("router_projection_dim", metadata.get("policy_projection_dim", 32))
            ),
            "use_pre_hidden_state_feature": bool(metadata.get("use_pre_hidden_state_feature", False)),
            "pre_hidden_state_dim": int(metadata.get("pre_hidden_state_dim", 64)),
            "router_feature_names": router_feature_names,
            "router_feature_stats_path": resolved_router_feature_stats_path,
            "router_feature_clip": float(metadata.get("router_feature_clip", 5.0)),
            "router_feature_eps": float(metadata.get("router_feature_eps", 1e-6)),
            "compat_impl": str(metadata.get("compat_impl", "optimized")),
            "dropout": dropout,
            "enable_monitor_cache": True,
            "enable_debug_cache": bool(enable_debug_cache),
        }
    elif router_layer_policy == "no_training":
        router_layer_spec = dict(default_layer_spec)
        router_layer_spec.update(
            {
                "enable_monitor_cache": True,
                "enable_debug_cache": bool(enable_debug_cache),
            }
        )
    else:
        raise ValueError(f"Unsupported router_layer_policy: {router_layer_policy}")

    router_layers = [int(layer) for layer in metadata.get("router_layers", [])]
    layer_overrides = {layer: dict(router_layer_spec) for layer in router_layers}
    return default_layer_spec, layer_overrides


def build_saved_mixed_composable_reft_config(
    model,
    layer_states: Dict[int, Dict[str, torch.Tensor]],
    specialist_labels: List[str],
    target_layers: List[int],
    default_layer_spec: Dict[str, Any],
    layer_policy_specs: Optional[Dict[Any, Dict[str, Any]]] = None,
):
    representations = []
    for layer in target_layers:
        layer_spec = dict(default_layer_spec)
        layer_spec.update(_get_layer_override(layer_policy_specs, layer))
        intervention_kwargs = _build_mixed_layer_intervention_kwargs(
            model=model,
            layer=layer,
            specialist_labels=specialist_labels,
            layer_spec=layer_spec,
            layer_state=layer_states[layer],
        )
        representations.append(
            {
                "layer": layer,
                "component": "block_output",
                "intervention": ComposableLoreftIntervention(**intervention_kwargs),
            }
        )
    return ReftConfig(representations=representations)


def load_trained_router_reft_model(
    model,
    router_checkpoint_dir: str,
    target_layers: Optional[List[int]] = None,
    router_metadata_path: Optional[str] = None,
    base_score_stats_path: Optional[str] = None,
    router_feature_stats_path: Optional[str] = None,
    enable_debug_cache: bool = False,
):
    router_model_dir = _resolve_router_model_dir(router_checkpoint_dir)
    metadata = load_router_training_metadata(
        router_checkpoint_dir=router_checkpoint_dir,
        router_metadata_path=router_metadata_path,
    )
    layer_states = load_composable_intervention_state(str(router_model_dir))
    saved_layers = sorted(layer_states.keys())
    metadata_target_layers = _coerce_int_list(metadata.get("target_layers"))
    resolved_target_layers = target_layers or metadata_target_layers or saved_layers
    missing_layers = [layer for layer in resolved_target_layers if layer not in layer_states]
    if missing_layers:
        raise ValueError(f"Requested router target layers missing from checkpoint: {missing_layers}")

    specialist_labels = [str(label) for label in metadata.get("specialist_labels", [])]
    if not specialist_labels:
        sample_layer = layer_states[resolved_target_layers[0]]
        specialist_labels = [f"specialist_{idx}" for idx in range(sample_layer["source_weight"].shape[0])]

    metadata = dict(metadata)
    candidate_router_layers = _coerce_int_list(metadata.get("router_layers")) or resolved_target_layers
    inferred_router_dims = _infer_trainable_policy_dims_from_checkpoint(
        weight_dir=str(router_model_dir),
        candidate_layers=[int(layer) for layer in candidate_router_layers],
    )
    metadata.update(inferred_router_dims)

    default_layer_spec, layer_policy_specs = _build_trained_router_layer_specs(
        metadata=metadata,
        base_score_stats_path=base_score_stats_path,
        router_feature_stats_path=router_feature_stats_path,
        enable_debug_cache=enable_debug_cache,
    )
    reft_config = build_saved_mixed_composable_reft_config(
        model=model,
        layer_states=layer_states,
        specialist_labels=specialist_labels,
        target_layers=resolved_target_layers,
        default_layer_spec=default_layer_spec,
        layer_policy_specs=layer_policy_specs,
    )
    reft_model = get_reft_model(model, reft_config)
    reft_model.load_intervention(str(router_model_dir / "intervenable_model"), include_model=True)
    return reft_model
