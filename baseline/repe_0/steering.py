from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import torch

from multi_train.eval_common.output_naming import normalize_base_model_name


REFT_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


SUPPORTED_REPE_COMPOSITIONS = {
    "single",
    "sum",
    "mean",
    "norm_mean",
    "weighted_sum",
}


def format_repe_text(question: str, answer: str, mode: str = "qa_response") -> str:
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    if mode == "raw_contrastive":
        if question:
            return REFT_PROMPT_TEMPLATE.format(instruction=question, response=answer)
        return answer
    if mode == "qa_response" and question:
        return REFT_PROMPT_TEMPLATE.format(instruction=question, response=answer)
    return answer


def get_decoder_layers(model):
    layer_candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("gpt_neox.layers", lambda m: m.gpt_neox.layers),
    ]
    for _, getter in layer_candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise AttributeError("Unsupported model architecture: cannot locate decoder layers.")


def resolve_target_layers(model, layers: Iterable[int] | None) -> list[int]:
    decoder_layers = get_decoder_layers(model)
    if not layers:
        return list(range(len(decoder_layers)))
    layers = [int(layer) for layer in layers]
    if layers == [-1]:
        return list(range(len(decoder_layers)))
    return layers


def infer_layers_from_vector_dir(vector_dir: str | Path) -> list[int]:
    vector_dir = Path(vector_dir)
    found: list[int] = []
    for path in vector_dir.glob("layer_*.pt"):
        match = re.fullmatch(r"layer_(\d+)\.pt", path.name)
        if match:
            found.append(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No layer_*.pt files found under {vector_dir}")
    return sorted(found)


def resolve_repe_vector_dirs(
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
) -> list[str]:
    single_value = str(vector_dir).strip() if vector_dir is not None else ""
    multi_values = [str(path).strip() for path in (vector_dirs or []) if str(path).strip()]

    if single_value and multi_values:
        raise ValueError("Use either `vector_dir` or `vector_dirs`, not both.")
    if single_value:
        return [single_value]
    if multi_values:
        return multi_values
    raise ValueError("At least one RepE vector directory must be provided.")


def infer_layers_from_vector_dirs(vector_dirs: Iterable[str | Path]) -> list[int]:
    vector_dir_list = resolve_repe_vector_dirs(vector_dirs=vector_dirs)
    base_layers = infer_layers_from_vector_dir(vector_dir_list[0])
    base_layer_set = set(base_layers)
    for candidate in vector_dir_list[1:]:
        candidate_layers = infer_layers_from_vector_dir(candidate)
        if set(candidate_layers) != base_layer_set:
            raise ValueError(
                "All RepE vector directories must contain the same layer set when `layers` is not specified. "
                f"Reference={vector_dir_list[0]} layers={base_layers}, candidate={candidate} layers={candidate_layers}"
            )
    return base_layers


def _load_vector_map(
    vector_dir: str | Path,
    layers: Iterable[int] | None = None,
) -> dict[int, torch.Tensor]:
    vector_dir = Path(vector_dir)
    target_layers = list(layers) if layers is not None else infer_layers_from_vector_dir(vector_dir)
    vector_map: dict[int, torch.Tensor] = {}
    for layer in target_layers:
        path = vector_dir / f"layer_{int(layer)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing RepE steering vector for layer {layer}: {path}")
        vector_map[int(layer)] = torch.load(path, map_location="cpu").float()
    return vector_map


def _vector_norm_safe(vector: torch.Tensor) -> torch.Tensor:
    return vector.norm().clamp_min(1e-12)


def _resolve_composition(composition: str, num_dirs: int) -> str:
    composition_name = str(composition or "single").strip().lower()
    if composition_name not in SUPPORTED_REPE_COMPOSITIONS:
        raise ValueError(
            f"Unsupported RepE composition={composition}. Expected one of {sorted(SUPPORTED_REPE_COMPOSITIONS)}."
        )
    if num_dirs == 1 and composition_name != "single":
        return composition_name
    if num_dirs > 1 and composition_name == "single":
        return "mean"
    return composition_name


def _resolve_weights(
    weights: Iterable[float] | None,
    expected_count: int,
    default_uniform: bool,
) -> list[float]:
    if weights is None:
        if default_uniform:
            return [1.0 / expected_count] * expected_count
        return [1.0] * expected_count

    resolved = [float(value) for value in weights]
    if len(resolved) != expected_count:
        raise ValueError(f"Expected {expected_count} RepE weights, got {len(resolved)}.")
    return resolved


def compose_repe_vector_map(
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> tuple[dict[int, torch.Tensor], dict]:
    resolved_dirs = resolve_repe_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
    resolved_layers = list(layers) if layers is not None else infer_layers_from_vector_dirs(resolved_dirs)
    resolved_composition = _resolve_composition(composition, len(resolved_dirs))

    source_maps = [
        _load_vector_map(vector_dir=source_dir, layers=resolved_layers)
        for source_dir in resolved_dirs
    ]

    if resolved_composition == "single":
        if len(source_maps) != 1:
            raise ValueError("`single` RepE composition requires exactly one vector directory.")
        return source_maps[0], {
            "vector_mode": "single",
            "vector_dir": resolved_dirs[0],
            "vector_dirs": resolved_dirs,
            "layers": resolved_layers,
            "composition": resolved_composition,
            "weights": None,
        }

    use_uniform_weights = resolved_composition in {"mean", "norm_mean", "weighted_sum"}
    resolved_weights = _resolve_weights(
        weights=weights,
        expected_count=len(source_maps),
        default_uniform=use_uniform_weights,
    )

    combined_map: dict[int, torch.Tensor] = {}
    for layer_idx in resolved_layers:
        layer_vectors = [source_map[layer_idx] for source_map in source_maps]

        if resolved_composition == "sum":
            combined = sum(layer_vectors)
        elif resolved_composition == "mean":
            combined = sum(layer_vectors) / len(layer_vectors)
        elif resolved_composition == "norm_mean":
            normalized_vectors = [vec / _vector_norm_safe(vec) for vec in layer_vectors]
            combined = sum(normalized_vectors) / len(normalized_vectors)
        elif resolved_composition == "weighted_sum":
            combined = sum(weight * vec for weight, vec in zip(resolved_weights, layer_vectors))
        else:
            raise ValueError(f"Unexpected RepE composition={resolved_composition}")

        combined_map[layer_idx] = combined.float()

    return combined_map, {
        "vector_mode": "composed",
        "vector_dir": None,
        "vector_dirs": resolved_dirs,
        "layers": resolved_layers,
        "composition": resolved_composition,
        "weights": resolved_weights,
    }


def clear_repe_steering(model) -> None:
    handles = getattr(model, "_repe_hook_handles", None)
    if handles:
        for handle in handles:
            handle.remove()
    model._repe_hook_handles = []
    model._repe_config = None


def apply_repe_steering(
    model,
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> dict:
    if token_strategy not in {"last", "all"}:
        raise ValueError(f"Unsupported token_strategy={token_strategy}. Expected 'last' or 'all'.")

    clear_repe_steering(model)

    decoder_layers = get_decoder_layers(model)
    candidate_dirs = resolve_repe_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
    default_layers = layers or infer_layers_from_vector_dirs(candidate_dirs)
    resolved_layers = resolve_target_layers(model, default_layers)
    vector_map, vector_meta = compose_repe_vector_map(
        vector_dir=vector_dir,
        vector_dirs=vector_dirs,
        layers=resolved_layers,
        composition=composition,
        weights=weights,
    )

    handles = []
    alpha_value = float(alpha)

    def make_hook(layer_idx: int):
        vector = vector_map[layer_idx]

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None:
                return output
            if hidden.ndim != 3:
                return output

            steer = vector.to(device=hidden.device, dtype=hidden.dtype)
            updated = hidden.clone()
            if token_strategy == "last":
                updated[:, -1, :] = updated[:, -1, :] + alpha_value * steer
            else:
                updated = updated + alpha_value * steer.view(1, 1, -1)
            if isinstance(output, tuple):
                return (updated,) + output[1:]
            return updated

        return hook

    for layer_idx in resolved_layers:
        handles.append(decoder_layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    config = {
        **vector_meta,
        "alpha": alpha_value,
        "token_strategy": token_strategy,
    }
    model._repe_hook_handles = handles
    model._repe_config = config
    return config


def sanitize_path_tag(value: str | Path) -> str:
    raw = Path(str(value)).name or str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")


def build_repe_model_tag(
    base_model_path: str,
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> str:
    model_tag = normalize_base_model_name(base_model_path)
    resolved_dirs = resolve_repe_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
    resolved_layers = list(layers) if layers else infer_layers_from_vector_dirs(resolved_dirs)
    resolved_composition = _resolve_composition(composition, len(resolved_dirs))

    if len(resolved_dirs) == 1 and resolved_composition == "single":
        vector_tag = sanitize_path_tag(resolved_dirs[0])
        composition_tag = "single"
        weight_tag = ""
    else:
        vector_tag = "_".join(sanitize_path_tag(path) for path in resolved_dirs)
        composition_tag = resolved_composition
        if weights is not None:
            weight_tag = "-w" + "_".join(str(float(value)).replace(".", "p") for value in weights)
        else:
            weight_tag = ""

    layer_tag = "_".join(str(int(layer)) for layer in resolved_layers)
    alpha_tag = str(alpha).replace(".", "p")
    return (
        f"{model_tag}-repe-{composition_tag}-{vector_tag}{weight_tag}"
        f"-l{layer_tag}-a{alpha_tag}-{token_strategy}"
    )
