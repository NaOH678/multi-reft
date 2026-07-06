from __future__ import annotations

from pathlib import Path

import torch

from multi_train.eval_common.output_naming import normalize_base_model_name


def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)

    return wrapped


class MATIntervener:
    collect_state = True
    collect_action = True

    def __init__(
        self,
        steering_vectors: torch.Tensor,
        gates_weights: torch.Tensor,
        gates_biases: torch.Tensor | None = None,
        multiplier: float = 1.0,
        layer_norm_preserve: bool = True,
        token_strategy: str = "last",
    ) -> None:
        if token_strategy not in {"last", "all"}:
            raise ValueError(f"Unsupported token_strategy={token_strategy}. Expected 'last' or 'all'.")
        self.steering_vectors = steering_vectors.float().cpu()
        self.gates_weights = gates_weights.float().cpu()
        self.gates_biases = (
            gates_biases.float().cpu()
            if gates_biases is not None
            else torch.zeros(steering_vectors.shape[0], dtype=torch.float32)
        )
        self.multiplier = float(multiplier)
        self.layer_norm_preserve = bool(layer_norm_preserve)
        self.token_strategy = token_strategy
        self.states = []
        self.actions = []

    def reset(self) -> None:
        self.states = []
        self.actions = []

    def _apply_to_states(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        steering_vectors = self.steering_vectors.to(device=states.device, dtype=states.dtype)
        gates_weights = self.gates_weights.to(device=states.device, dtype=states.dtype)
        gates_biases = self.gates_biases.to(device=states.device, dtype=states.dtype)

        original = states
        gates = torch.sigmoid(torch.matmul(states, gates_weights.T) + gates_biases)
        delta = torch.matmul(gates, steering_vectors)
        adjusted = original + self.multiplier * delta

        if self.layer_norm_preserve:
            original_norm = torch.norm(original, p=2, dim=-1, keepdim=True)
            adjusted_norm = torch.norm(adjusted, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
            adjusted = adjusted * (original_norm / adjusted_norm)

        return adjusted, delta

    def __call__(self, b, s):
        if self.token_strategy == "all":
            original_states = b.detach().clone()
            flat_states = b.reshape(-1, b.shape[-1])
            adjusted, delta = self._apply_to_states(flat_states)
            b[...] = adjusted.reshape_as(b)
            self.states.append(original_states.cpu())
            self.actions.append(delta.reshape_as(b).detach().cpu())
            return b

        original_states = b[:, -1, :].detach().clone()
        adjusted, delta = self._apply_to_states(b[:, -1, :])
        b[:, -1, :] = adjusted
        self.states.append(original_states.cpu())
        self.actions.append(delta.detach().cpu())
        return b

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        multiplier: float = 1.0,
        layer_norm_preserve: bool = True,
        token_strategy: str = "last",
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        num_attributes = int(checkpoint["num_attributes"])

        steering_vectors = torch.stack(
            [state_dict[f"steering_vectors.{idx}"] for idx in range(num_attributes)],
            dim=0,
        )
        gates_weights = torch.stack(
            [state_dict[f"gating_weights.{idx}.weight"].squeeze(0) for idx in range(num_attributes)],
            dim=0,
        )
        gates_biases = torch.stack(
            [state_dict[f"gating_weights.{idx}.bias"].squeeze(0) for idx in range(num_attributes)],
            dim=0,
        )
        return cls(
            steering_vectors=steering_vectors,
            gates_weights=gates_weights,
            gates_biases=gates_biases,
            multiplier=multiplier,
            layer_norm_preserve=layer_norm_preserve,
            token_strategy=token_strategy,
        )


def build_mat_model_tag(
    *,
    base_model_path: str,
    checkpoint_path: str | Path,
    layer: int,
    alpha: float,
    token_strategy: str = "last",
) -> str:
    base_model_name = normalize_base_model_name(base_model_path)
    checkpoint_name = Path(str(checkpoint_path)).stem.replace("_metadata", "")
    token_suffix = f"tok{token_strategy}"
    alpha_suffix = str(alpha).replace(".", "p")
    return f"{base_model_name}-mat4d-{checkpoint_name}-l{int(layer)}-a{alpha_suffix}-{token_suffix}"
