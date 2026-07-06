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


SUPPORTED_ITI_COMPOSITIONS = {
    "single",
    "sum",
    "mean",
    "norm_mean",
    "weighted_sum",
}


def format_iti_text(question: str, answer: str, mode: str = "qa_response") -> str:
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


def get_attention_output_projection(layer):
    projection_candidates = [
        ("self_attn.o_proj", lambda module: module.self_attn.o_proj),
        ("attention.o_proj", lambda module: module.attention.o_proj),
        ("attn.out_proj", lambda module: module.attn.out_proj),
    ]
    for _, getter in projection_candidates:
        try:
            proj = getter(layer)
        except AttributeError:
            continue
        if proj is not None:
            return proj
    raise AttributeError("Unsupported decoder layer: cannot locate attention output projection.")


def infer_layers_from_artifact_dir(artifact_dir: str | Path) -> list[int]:
    artifact_dir = Path(artifact_dir)
    found: list[int] = []
    for path in artifact_dir.glob("layer_*.pt"):
        match = re.fullmatch(r"layer_(\d+)\.pt", path.name)
        if match:
            found.append(int(match.group(1)))
    if not found:
        raise FileNotFoundError(f"No layer_*.pt files found under {artifact_dir}")
    return sorted(found)


def resolve_iti_artifact_dirs(
    artifact_dir: str | Path | None = None,
    artifact_dirs: Iterable[str | Path] | None = None,
) -> list[str]:
    single_value = str(artifact_dir).strip() if artifact_dir is not None else ""
    multi_values = [str(path).strip() for path in (artifact_dirs or []) if str(path).strip()]

    if single_value and multi_values:
        raise ValueError("Use either `artifact_dir` or `artifact_dirs`, not both.")
    if single_value:
        return [single_value]
    if multi_values:
        return multi_values
    raise ValueError("At least one ITI artifact directory must be provided.")


def infer_layers_from_artifact_dirs(artifact_dirs: Iterable[str | Path]) -> list[int]:
    artifact_dir_list = resolve_iti_artifact_dirs(artifact_dirs=artifact_dirs)
    base_layers = infer_layers_from_artifact_dir(artifact_dir_list[0])
    base_layer_set = set(base_layers)
    for candidate in artifact_dir_list[1:]:
        candidate_layers = infer_layers_from_artifact_dir(candidate)
        if set(candidate_layers) != base_layer_set:
            raise ValueError(
                "All ITI artifact directories must contain the same layer set when layers are not specified. "
                f"Reference={artifact_dir_list[0]} layers={base_layers}, candidate={candidate} layers={candidate_layers}"
            )
    return base_layers


def _load_vector_map(
    artifact_dir: str | Path,
    layers: Iterable[int] | None = None,
) -> dict[int, torch.Tensor]:
    artifact_dir = Path(artifact_dir)
    target_layers = list(layers) if layers is not None else infer_layers_from_artifact_dir(artifact_dir)
    vector_map: dict[int, torch.Tensor] = {}
    for layer in target_layers:
        path = artifact_dir / f"layer_{int(layer)}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing ITI intervention vector for layer {layer}: {path}")
        vector_map[int(layer)] = torch.load(path, map_location="cpu").float()
    return vector_map


def _vector_norm_safe(vector: torch.Tensor) -> torch.Tensor:
    return vector.norm().clamp_min(1e-12)


def _resolve_composition(
    composition: str,
    num_dirs: int,
) -> str:
    composition_name = str(composition or "single").strip().lower()
    if composition_name not in SUPPORTED_ITI_COMPOSITIONS:
        raise ValueError(
            f"Unsupported ITI composition={composition}. Expected one of {sorted(SUPPORTED_ITI_COMPOSITIONS)}."
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
        raise ValueError(f"Expected {expected_count} ITI weights, got {len(resolved)}.")
    return resolved


def compose_iti_vector_map(
    artifact_dir: str | Path | None = None,
    artifact_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> tuple[dict[int, torch.Tensor], dict]:
    resolved_dirs = resolve_iti_artifact_dirs(artifact_dir=artifact_dir, artifact_dirs=artifact_dirs)
    resolved_layers = list(layers) if layers is not None else infer_layers_from_artifact_dirs(resolved_dirs)
    resolved_composition = _resolve_composition(composition, len(resolved_dirs))

    source_maps = [
        _load_vector_map(artifact_dir=source_dir, layers=resolved_layers)
        for source_dir in resolved_dirs
    ]

    if resolved_composition == "single":
        if len(source_maps) != 1:
            raise ValueError("`single` ITI composition requires exactly one artifact directory.")
        return source_maps[0], {
            "artifact_mode": "single",
            "artifact_dir": resolved_dirs[0],
            "artifact_dirs": resolved_dirs,
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
            raise ValueError(f"Unexpected ITI composition={resolved_composition}")

        combined_map[layer_idx] = combined.float()

    return combined_map, {
        "artifact_mode": "composed",
        "artifact_dir": None,
        "artifact_dirs": resolved_dirs,
        "layers": resolved_layers,
        "composition": resolved_composition,
        "weights": resolved_weights,
    }


def clear_iti_steering(model) -> None:
    handles = getattr(model, "_iti_hook_handles", None)
    if handles:
        for handle in handles:
            handle.remove()
    model._iti_hook_handles = []
    model._iti_config = None


def apply_iti_steering(
    model,
    artifact_dir: str | Path | None = None,
    artifact_dirs: Iterable[str | Path] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
    include_prompt: bool = False,
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> dict:
    if token_strategy not in {"last", "all"}:
        raise ValueError(f"Unsupported ITI token_strategy={token_strategy}. Expected 'last' or 'all'.")

    clear_iti_steering(model)

    decoder_layers = get_decoder_layers(model)
    candidate_dirs = resolve_iti_artifact_dirs(artifact_dir=artifact_dir, artifact_dirs=artifact_dirs)
    resolved_layers = infer_layers_from_artifact_dirs(candidate_dirs)
    vector_map, vector_meta = compose_iti_vector_map(
        artifact_dir=artifact_dir,
        artifact_dirs=artifact_dirs,
        layers=resolved_layers,
        composition=composition,
        weights=weights,
    )

    handles = []
    alpha_value = float(alpha)
    include_prompt_value = bool(include_prompt)

    for layer_idx in resolved_layers:
        vector = vector_map[layer_idx].detach().clone().float()
        projection = get_attention_output_projection(decoder_layers[layer_idx])

        def pre_hook(_module, inputs, layer_vector=vector):
            if not inputs:
                return inputs

            hidden = inputs[0]
            if hidden is None:
                return inputs

            if hidden.ndim != 3:
                return inputs

            # ITI has two distinct runtime modes:
            # - include_prompt=True: intervene once on the prompt's last token.
            # - include_prompt=False: intervene on decode-time single-token steps only.
            #
            # HuggingFace generation typically runs:
            # 1. one full-prompt forward pass, where seq_len > 1
            # 2. cached decode steps, where seq_len == 1
            #
            # The previous implementation always used decode-only by default,
            # which does not match the intended ITI setup for truth-style MCQA.
            if include_prompt_value:
                if hidden.shape[1] <= 1:
                    return inputs
            else:
                if hidden.shape[1] > 1:
                    return inputs

            steer = (layer_vector.to(hidden.device, dtype=hidden.dtype) * alpha_value)
            hidden = hidden.clone()

            if token_strategy == "last":
                hidden[:, -1, :] = hidden[:, -1, :] + steer
            else:
                hidden = hidden + steer.view(1, 1, -1)

            return (hidden, *inputs[1:])

        handles.append(projection.register_forward_pre_hook(pre_hook))

    config = {
        "artifact_dir": artifact_dir,
        "artifact_dirs": candidate_dirs,
        "layers": resolved_layers,
        "alpha": alpha_value,
        "token_strategy": token_strategy,
        "include_prompt": include_prompt_value,
        "composition": composition,
        "weights": None if weights is None else [float(value) for value in weights],
        **vector_meta,
    }
    model._iti_hook_handles = handles
    model._iti_config = config
    return config


def build_iti_model_tag(
    base_model_path: str,
    artifact_dir: str | Path | None = None,
    artifact_dirs: Iterable[str | Path] | None = None,
    alpha: float = 1.0,
    token_strategy: str = "last",
    include_prompt: bool = False,
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> str:
    model_name = normalize_base_model_name(base_model_path)
    resolved_dirs = resolve_iti_artifact_dirs(artifact_dir=artifact_dir, artifact_dirs=artifact_dirs)
    resolved_composition = _resolve_composition(composition, len(resolved_dirs))

    if len(resolved_dirs) == 1:
        artifact_tag = Path(resolved_dirs[0]).name
    else:
        artifact_tag = f"{len(resolved_dirs)}dirs-{resolved_composition}"

    alpha_tag = str(alpha).replace(".", "p")
    prompt_tag = "inclprompt" if include_prompt else "decodeonly"
    token_tag = token_strategy
    tag = f"{model_name}-iti-{resolved_composition}-{artifact_tag}-a{alpha_tag}-{token_tag}-{prompt_tag}"

    if weights is not None:
        weight_tag = "-".join(str(float(value)).replace(".", "p") for value in weights)
        tag = f"{tag}-w{weight_tag}"
    return tag
