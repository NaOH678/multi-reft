from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYVENE_PROJECT_ROOT = PROJECT_ROOT / "pyvene"
if str(PYVENE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVENE_PROJECT_ROOT))

import pyvene as pv

from multi_train.eval_common.output_naming import normalize_base_model_name


SUPPORTED_PYVENE_COMPOSITIONS = {
    "single",
    "sum",
    "mean",
    "norm_mean",
    "weighted_sum",
}


def _infer_model_device_dtype(model) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device, parameter.dtype

    device = getattr(model, "device", None)
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    dtype = getattr(model, "dtype", torch.float32)
    return device, dtype


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


def resolve_pyvene_vector_dirs(
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
    raise ValueError("At least one pyvene vector directory must be provided.")


def infer_layers_from_vector_dirs(vector_dirs: Iterable[str | Path]) -> list[int]:
    vector_dir_list = resolve_pyvene_vector_dirs(vector_dirs=vector_dirs)
    base_layers = infer_layers_from_vector_dir(vector_dir_list[0])
    base_layer_set = set(base_layers)
    for candidate in vector_dir_list[1:]:
        candidate_layers = infer_layers_from_vector_dir(candidate)
        if set(candidate_layers) != base_layer_set:
            raise ValueError(
                "All pyvene vector directories must contain the same layer set when layers are not specified. "
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
            raise FileNotFoundError(f"Missing pyvene steering vector for layer {layer}: {path}")
        vector_map[int(layer)] = torch.load(path, map_location="cpu").float()
    return vector_map


def _vector_norm_safe(vector: torch.Tensor) -> torch.Tensor:
    return vector.norm().clamp_min(1e-12)


def _resolve_composition(composition: str, num_dirs: int) -> str:
    composition_name = str(composition or "single").strip().lower()
    if composition_name not in SUPPORTED_PYVENE_COMPOSITIONS:
        raise ValueError(
            f"Unsupported pyvene composition={composition}. Expected one of {sorted(SUPPORTED_PYVENE_COMPOSITIONS)}."
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
        raise ValueError(f"Expected {expected_count} pyvene weights, got {len(resolved)}.")
    return resolved


def compose_pyvene_vector_map(
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    composition: str = "single",
    weights: Iterable[float] | None = None,
) -> tuple[dict[int, torch.Tensor], dict]:
    resolved_dirs = resolve_pyvene_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
    resolved_layers = list(layers) if layers is not None else infer_layers_from_vector_dirs(resolved_dirs)
    resolved_composition = _resolve_composition(composition, len(resolved_dirs))

    source_maps = [
        _load_vector_map(vector_dir=source_dir, layers=resolved_layers)
        for source_dir in resolved_dirs
    ]

    if resolved_composition == "single":
        if len(source_maps) != 1:
            raise ValueError("`single` pyvene composition requires exactly one vector directory.")
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
            raise ValueError(f"Unexpected pyvene composition={resolved_composition}")

        combined_map[layer_idx] = combined.float()

    return combined_map, {
        "vector_mode": "composed",
        "vector_dir": None,
        "vector_dirs": resolved_dirs,
        "layers": resolved_layers,
        "composition": resolved_composition,
        "weights": resolved_weights,
    }


class PyveneGenerateWrapper:
    def __init__(
        self,
        *,
        intervenable_model,
        base_model,
        source_representations: list[torch.Tensor],
        intervene_on_prompt: bool,
        unit_locations,
        component: str,
        method_tag: str,
        alpha: float,
        layers: list[int],
        composition: str,
        vector_meta: dict,
        weights: list[float] | None,
    ) -> None:
        self.intervenable_model = intervenable_model
        self.base_model = base_model
        self.source_representations = source_representations
        self.intervene_on_prompt = bool(intervene_on_prompt)
        self.unit_locations = unit_locations
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.device = getattr(base_model, "device", None)
        self.pyvene_config = {
            "method_tag": method_tag,
            "component": component,
            "alpha": float(alpha),
            "layers": [int(layer) for layer in layers],
            "intervene_on_prompt": bool(intervene_on_prompt),
            "unit_locations": unit_locations,
            "composition": composition,
            "weights": weights,
            **vector_meta,
        }

    def _materialize_source_representations(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        materialized = [
            representation.to(device=device, dtype=dtype)
            for representation in self.source_representations
        ]
        self.source_representations = materialized
        return materialized

    def _resolve_runtime_unit_locations(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_return_sequences: int,
    ):
        batch_size = int(input_ids.shape[0])
        repeat_count = max(1, int(num_return_sequences))
        effective_batch_size = batch_size * repeat_count
        num_interventions = len(self.source_representations)
        sequence_length = int(input_ids.shape[-1])

        if self.unit_locations is None:
            if not self.intervene_on_prompt:
                per_sample_locations = [[0] for _ in range(effective_batch_size)]
                per_intervention_locations = [per_sample_locations for _ in range(num_interventions)]
                return {"sources->base": (None, per_intervention_locations)}
            return None

        if "sources->base" in self.unit_locations:
            return self.unit_locations

        if "base" not in self.unit_locations:
            return self.unit_locations

        base_location = int(self.unit_locations["base"])

        if attention_mask is not None:
            valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)
        else:
            valid_lengths = torch.full(
                (batch_size,),
                fill_value=sequence_length,
                dtype=torch.long,
                device=input_ids.device,
            )

        per_sample_locations: list[list[int]] = []
        for sample_idx in range(batch_size):
            sample_length = int(valid_lengths[sample_idx].item())
            if sample_length <= 0:
                raise ValueError("Encountered an empty prompt while resolving pyvene unit locations.")

            if base_location < 0:
                resolved_location = sample_length + base_location
            else:
                resolved_location = base_location

            if resolved_location < 0 or resolved_location >= sample_length:
                raise ValueError(
                    f"Resolved pyvene unit location {resolved_location} is outside valid prompt length {sample_length}."
                )
            per_sample_locations.append([int(resolved_location)])

        if repeat_count > 1:
            expanded_locations: list[list[int]] = []
            for location in per_sample_locations:
                expanded_locations.extend([location] * repeat_count)
            per_sample_locations = expanded_locations

        per_intervention_locations = [per_sample_locations for _ in range(num_interventions)]
        return {"sources->base": (None, per_intervention_locations)}

    def generate(self, *args, **kwargs):
        if args:
            if len(args) > 1:
                raise ValueError("PyveneGenerateWrapper.generate expects at most one positional input_ids tensor.")
            kwargs["input_ids"] = args[0]

        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if input_ids is None:
            raise ValueError("PyveneGenerateWrapper.generate requires `input_ids`.")

        model_device, model_dtype = _infer_model_device_dtype(self.base_model)
        source_representations = self._materialize_source_representations(
            device=model_device,
            dtype=model_dtype,
        )
        runtime_unit_locations = self._resolve_runtime_unit_locations(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=kwargs.get("num_return_sequences", 1),
        )

        base = {"input_ids": input_ids}
        if attention_mask is not None:
            base["attention_mask"] = attention_mask

        _, counterfactual_outputs = self.intervenable_model.generate(
            base=base,
            source_representations=source_representations,
            unit_locations=runtime_unit_locations,
            intervene_on_prompt=self.intervene_on_prompt,
            **kwargs,
        )
        return counterfactual_outputs

    def eval(self):
        self.base_model.eval()
        return self

    def to(self, *args, **kwargs):
        self.base_model.to(*args, **kwargs)
        model_device, model_dtype = _infer_model_device_dtype(self.base_model)
        self._materialize_source_representations(device=model_device, dtype=model_dtype)
        return self

    def parameters(self):
        return self.base_model.parameters()

    def named_parameters(self, *args, **kwargs):
        return self.base_model.named_parameters(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.base_model, name)


def apply_pyvene_additive_steering(
    model,
    *,
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    alpha: float = 1.0,
    component: str = "block_output",
    composition: str = "single",
    weights: Iterable[float] | None = None,
    method_tag: str = "pyvene",
    intervene_on_prompt: bool = False,
    base_unit_location: int | None = None,
):
    if not str(component or "").strip():
        raise ValueError("`component` must be a non-empty pyvene representation target, e.g. `block_output`.")
    if not str(method_tag or "").strip():
        raise ValueError("`method_tag` must be a non-empty identifier for pyvene steering.")

    resolved_dirs = resolve_pyvene_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
    resolved_layers = list(layers) if layers is not None else infer_layers_from_vector_dirs(resolved_dirs)
    vector_map, vector_meta = compose_pyvene_vector_map(
        vector_dir=vector_dir,
        vector_dirs=vector_dirs,
        layers=resolved_layers,
        composition=composition,
        weights=weights,
    )

    config = pv.IntervenableConfig(
        model_type=type(model),
        representations=[
            pv.RepresentationConfig(int(layer), component, "pos", 1)
            for layer in resolved_layers
        ],
        intervention_types=pv.AdditionIntervention,
    )
    intervenable_model = pv.IntervenableModel(config, model=model)

    alpha_value = float(alpha)
    unit_locations = None
    if base_unit_location is not None:
        unit_locations = {"base": int(base_unit_location)}
    model_device, model_dtype = _infer_model_device_dtype(model)
    source_representations = [
        (
            vector_map[int(layer)].detach().clone().to(device=model_device, dtype=model_dtype)
            * alpha_value
        )
        for layer in resolved_layers
    ]

    wrapped = PyveneGenerateWrapper(
        intervenable_model=intervenable_model,
        base_model=model,
        source_representations=source_representations,
        intervene_on_prompt=bool(intervene_on_prompt),
        unit_locations=unit_locations,
        component=component,
        method_tag=method_tag,
        alpha=alpha_value,
        layers=[int(layer) for layer in resolved_layers],
        composition=composition,
        vector_meta=vector_meta,
        weights=None if weights is None else [float(value) for value in weights],
    )
    return wrapped, wrapped.pyvene_config


def sanitize_path_tag(value: str | Path) -> str:
    raw = Path(str(value)).name or str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")


def build_pyvene_model_tag(
    *,
    method_tag: str,
    base_model_path: str,
    vector_dir: str | Path | None = None,
    vector_dirs: Iterable[str | Path] | None = None,
    layers: Iterable[int] | None = None,
    alpha: float = 1.0,
    component: str = "block_output",
    composition: str = "single",
    weights: Iterable[float] | None = None,
    intervene_on_prompt: bool = False,
    base_unit_location: int | None = None,
) -> str:
    if not str(method_tag or "").strip():
        raise ValueError("`method_tag` must be provided when building a pyvene model tag.")
    if not str(component or "").strip():
        raise ValueError("`component` must be provided when building a pyvene model tag.")

    model_tag = normalize_base_model_name(base_model_path)
    resolved_dirs = resolve_pyvene_vector_dirs(vector_dir=vector_dir, vector_dirs=vector_dirs)
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
    if intervene_on_prompt and base_unit_location == -1:
        prompt_tag = "promptlast"
    elif intervene_on_prompt:
        prompt_tag = "promptonly"
    elif base_unit_location is None:
        prompt_tag = "genall"
    else:
        prompt_tag = f"genloc{base_unit_location}"
    component_tag = component.replace(".", "-")
    return (
        f"{model_tag}-{method_tag}-{composition_tag}-{vector_tag}{weight_tag}"
        f"-{component_tag}-l{layer_tag}-a{alpha_tag}-{prompt_tag}"
    )
