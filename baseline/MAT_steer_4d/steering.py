from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .interveners import MATIntervener, build_mat_model_tag


class SteeringModule(nn.Module):
    def __init__(self, input_dim: int, num_attributes: int) -> None:
        super().__init__()
        self.num_attributes = num_attributes
        self.steering_vectors = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_attributes)]
        )
        self.gating_weights = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_attributes)])

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        adjusted = activations.clone()
        gates = []
        for idx in range(self.num_attributes):
            gate = torch.sigmoid(self.gating_weights[idx](activations))
            gates.append(gate)
            adjusted = adjusted + gate * self.steering_vectors[idx]
        return adjusted, torch.cat(gates, dim=1)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    pairwise_dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-pairwise_dist / (2 * (sigma ** 2)))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    return gaussian_kernel(x, x, sigma).mean() + gaussian_kernel(y, y, sigma).mean() - 2 * gaussian_kernel(x, y, sigma).mean()


def normalize_activations(original: torch.Tensor, adjusted: torch.Tensor) -> torch.Tensor:
    original_norm = torch.norm(original, p=2, dim=1, keepdim=True)
    adjusted_norm = torch.norm(adjusted, p=2, dim=1, keepdim=True).clamp_min(1e-8)
    return adjusted * (original_norm / adjusted_norm)


def sparsity_loss(gates: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(gates))


def orthogonality_loss(steering_vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    vectors = list(steering_vectors)
    loss = torch.tensor(0.0, dtype=vectors[0].dtype, device=vectors[0].device)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            numerator = torch.dot(vectors[i], vectors[j])
            denominator = torch.norm(vectors[i]) * torch.norm(vectors[j])
            loss = loss + (numerator / denominator.clamp_min(1e-8)) ** 2
    return loss


def preservation_loss(gates: torch.Tensor) -> torch.Tensor:
    return torch.mean(gates ** 2)


def clear_mat_steering(model) -> None:
    handles = getattr(model, "_mat_hook_handles", None)
    if handles:
        for handle in handles:
            handle.remove()
    model._mat_hook_handles = []
    model._mat_config = None


def _resolve_mat_target_module(model, layer: int):
    return model.get_submodule(f"model.layers.{int(layer)}.self_attn.o_proj")


def apply_mat_steering(
    model,
    checkpoint_path: str | Path,
    layer: int,
    alpha: float = 1.0,
    token_strategy: str = "last",
    preserve_norm: bool = True,
):
    clear_mat_steering(model)
    intervener = MATIntervener.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        multiplier=alpha,
        layer_norm_preserve=preserve_norm,
        token_strategy=token_strategy,
    )
    target_module = _resolve_mat_target_module(model, layer)

    def mat_pre_hook(module, inputs):
        if not inputs:
            return inputs
        hidden_states = inputs[0]
        if hidden_states is None:
            return inputs
        hidden_states = hidden_states.clone()
        hidden_shape = tuple(hidden_states.shape)
        if hidden_states.dim() < 2:
            return inputs
        if token_strategy == "all":
            original_states = hidden_states.detach().clone()
            flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            adjusted, delta = intervener._apply_to_states(flat_states)
            hidden_states = adjusted.reshape_as(hidden_states)
            intervener.states.append(original_states.cpu())
            intervener.actions.append(delta.reshape(*hidden_shape).detach().cpu())
        else:
            original_states = hidden_states[:, -1, :].detach().clone()
            adjusted, delta = intervener._apply_to_states(hidden_states[:, -1, :])
            hidden_states[:, -1, :] = adjusted
            intervener.states.append(original_states.cpu())
            intervener.actions.append(delta.detach().cpu())
        return (hidden_states, *inputs[1:])

    handle = target_module.register_forward_pre_hook(mat_pre_hook)
    model._mat_hook_handles = [handle]
    mat_config = {
        "checkpoint_path": str(checkpoint_path),
        "layer": int(layer),
        "alpha": float(alpha),
        "token_strategy": token_strategy,
        "preserve_norm": bool(preserve_norm),
        "component": f"model.layers[{int(layer)}].self_attn.o_proj.input",
        "model_tag_builder": build_mat_model_tag,
    }
    model._mat_config = mat_config
    model._mat_intervener = intervener
    return model, mat_config


def save_checkpoint(
    model: SteeringModule,
    save_path: str | Path,
    *,
    input_dim: int,
    task_names: list[str],
    layer: int,
    model_name: str,
    hyperparams: dict,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(input_dim),
        "num_attributes": len(task_names),
        "task_names": list(task_names),
    }
    torch.save(checkpoint, save_path)

    metadata_path = save_path.with_name(f"{save_path.stem}_metadata.json")
    metadata = {
        "layer": int(layer),
        "model_name": model_name,
        "datasets": list(task_names),
        "hyperparams": hyperparams,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
