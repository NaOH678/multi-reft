import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from pyvene import (
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from transformers.activations import ACT2FN


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)

        # Caveat: without creating a new layer, it might not work (still not sure why)
        # We have to recreate a layer, and load back the columns.
        overload_w = state_dict["rotate_layer"].to(
            self.learned_source.weight.device)
        overload_w_width = overload_w.shape[-1]
        rotate_layer = LowRankRotateLayer(
            self.embed_dim, overload_w_width, init_orth=True).to(
            self.learned_source.weight.device)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        assert torch.allclose(self.rotate_layer.weight.data, overload_w.data) == True # we must match!
        
        return


class ComposableLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention
):
    """
    Token-local composition of multiple LoReFT specialists.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        dtype = kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16

        rotate_weight = kwargs.get("rotate_weight")
        source_weight = kwargs.get("source_weight")
        source_bias = kwargs.get("source_bias")
        if rotate_weight is None or source_weight is None or source_bias is None:
            raise ValueError("ComposableLoreftIntervention requires rotate_weight/source_weight/source_bias.")

        rotate_weight = rotate_weight.detach().clone().to(torch.float32)
        source_weight = source_weight.detach().clone().to(dtype)
        source_bias = source_bias.detach().clone().to(dtype)

        if rotate_weight.dim() != 3 or source_weight.dim() != 3 or source_bias.dim() != 2:
            raise ValueError("Expected stacked specialist tensors with shapes [T,d,r], [T,r,d], [T,r].")

        self.num_specialists = int(rotate_weight.shape[0])
        self.low_rank_dimension = int(source_weight.shape[1])
        self.compose_domain = kwargs.get("compose_domain", "output")
        self.policy_type = kwargs.get("policy_type", "equal")
        self.temperature = float(kwargs.get("temperature", 1.0))
        self.topk = kwargs.get("topk", None)
        self.compat_threshold = float(kwargs.get("compat_threshold", 0.0))
        self.single_index = kwargs.get("single_index", None)
        self.shared_basis_type = kwargs.get("shared_basis_type")
        self.shared_basis_rank = kwargs.get("shared_basis_rank")
        self.transport_type = kwargs.get("transport_type")
        self.score_normalizer = kwargs.get("score_normalizer", "none")
        self.score_eps = float(kwargs.get("score_eps", 1e-6))
        score_clip = kwargs.get("score_clip", None)
        self.score_clip = None if score_clip is None else float(score_clip)
        self.use_trainable_policy = bool(kwargs.get("use_trainable_policy", False))
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]

        self.rotate_weight = nn.Parameter(rotate_weight, requires_grad=True)
        self.source_weight = nn.Parameter(source_weight, requires_grad=True)
        self.source_bias = nn.Parameter(source_bias, requires_grad=True)

        shared_basis = kwargs.get("shared_basis")
        if shared_basis is not None:
            self.shared_basis = nn.Parameter(shared_basis.detach().clone().to(torch.float32), requires_grad=False)
        else:
            self.shared_basis = None

        transport_weight = kwargs.get("transport_weight")
        if transport_weight is not None:
            self.transport_weight = nn.Parameter(transport_weight.detach().clone().to(torch.float32), requires_grad=False)
        else:
            self.transport_weight = None

        score_stats = kwargs.get("score_stats") or {}
        self.register_buffer(
            "score_mean",
            score_stats.get("mean").detach().clone().to(torch.float32) if score_stats.get("mean") is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "score_std",
            score_stats.get("std").detach().clone().to(torch.float32) if score_stats.get("std") is not None else None,
            persistent=False,
        )
        self.register_buffer(
            "score_log_mean",
            score_stats.get("log_mean").detach().clone().to(torch.float32)
            if score_stats.get("log_mean") is not None
            else None,
            persistent=False,
        )
        self.register_buffer(
            "score_log_std",
            score_stats.get("log_std").detach().clone().to(torch.float32)
            if score_stats.get("log_std") is not None
            else None,
            persistent=False,
        )

        policy_hidden_dim = int(kwargs.get("policy_hidden_dim", 32))
        if self.use_trainable_policy:
            feature_dim = 5
            self.policy_head = nn.Sequential(
                nn.Linear(feature_dim, policy_hidden_dim),
                nn.ReLU(),
                nn.Linear(policy_hidden_dim, 1),
            ).to(dtype)
        else:
            self.policy_head = None

        self.latest_alpha = None
        self.latest_scores = None
        self.latest_normalized_score = None
        self.latest_delta_norm = None
        self.latest_intervention_norm = None
        self.latest_pairwise_cos = None
        self.latest_selected_mask = None
        self.latest_conflict_score = None
        self.latest_compose_domain = None
        self.latest_shared_basis_stats = None
        self.latest_transport_stats = None

    def freeze_specialists(self):
        self.rotate_weight.requires_grad = False
        self.source_weight.requires_grad = False
        self.source_bias.requires_grad = False

    def freeze_policy(self):
        if self.policy_head is None:
            return
        for param in self.policy_head.parameters():
            param.requires_grad = False

    def unfreeze_policy(self):
        if self.policy_head is None:
            return
        for param in self.policy_head.parameters():
            param.requires_grad = True

    def _compute_specialist_states(self, base):
        base_source = base.to(self.source_weight.dtype)
        base_rotate = base.to(self.rotate_weight.dtype)

        rotated_base = torch.einsum("bsd,tdr->bstr", base_rotate, self.rotate_weight)
        learned_source = torch.einsum("bsd,trd->bstr", base_source, self.source_weight)
        learned_source = learned_source + self.source_bias.unsqueeze(0).unsqueeze(0)
        learned_source = self.act_fn(learned_source)
        delta = learned_source - rotated_base.to(learned_source.dtype)
        lifted = torch.einsum("bstr,tdr->bstd", delta.to(self.rotate_weight.dtype), self.rotate_weight)

        return {
            "base": base,
            "rotated_base": rotated_base,
            "learned_source": learned_source,
            "delta": delta,
            "Delta": lifted,
        }

    def _compute_pairwise_cos(self, lifted):
        normalized = F.normalize(lifted.float(), dim=-1, eps=1e-6)
        return torch.einsum("bstd,bsud->bstu", normalized, normalized)

    def _extract_policy_features(self, states):
        delta = states["delta"]
        lifted = states["Delta"]
        rotated = states["rotated_base"]

        delta_norm = delta.float().norm(dim=-1)
        intervention_norm = lifted.float().norm(dim=-1)
        subspace_energy = rotated.float().pow(2).sum(dim=-1)
        pairwise_cos = self._compute_pairwise_cos(lifted)

        diag = torch.eye(self.num_specialists, device=pairwise_cos.device, dtype=torch.bool).view(
            1, 1, self.num_specialists, self.num_specialists
        )
        offdiag = pairwise_cos.masked_fill(diag, 0.0)
        denom = max(1, self.num_specialists - 1)
        mean_compat = offdiag.sum(dim=-1) / denom
        neg_compat_mass = torch.clamp(-offdiag, min=0.0).sum(dim=-1)

        return {
            "delta_norm": delta_norm,
            "intervention_norm": intervention_norm,
            "subspace_energy": subspace_energy,
            "pairwise_cos": pairwise_cos,
            "mean_compat": mean_compat,
            "neg_compat_mass": neg_compat_mass,
        }

    def _masked_softmax(self, scores, mask):
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(masked_scores, dim=-1)
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
        denom = alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return alpha / denom

    def _normalize_delta_scores(self, delta_norm, mode=None):
        norm_mode = mode or self.score_normalizer or "none"
        delta_norm = delta_norm.float()

        if norm_mode == "none":
            scores = delta_norm
        elif norm_mode == "mean_ratio":
            if self.score_mean is None:
                raise ValueError("mean_ratio normalization requires score_mean stats.")
            denom = self.score_mean.float().view(1, 1, -1).clamp_min(self.score_eps)
            scores = delta_norm / denom
        elif norm_mode == "log_zscore":
            if self.score_log_mean is None or self.score_log_std is None:
                raise ValueError("log_zscore normalization requires score_log_mean / score_log_std stats.")
            log_delta = torch.log(delta_norm + self.score_eps)
            mean = self.score_log_mean.float().view(1, 1, -1)
            std = self.score_log_std.float().view(1, 1, -1).clamp_min(self.score_eps)
            scores = (log_delta - mean) / std
        else:
            raise ValueError(f"Unsupported score_normalizer: {norm_mode}")

        if self.score_clip is not None:
            scores = torch.clamp(scores, min=-self.score_clip, max=self.score_clip)
        return scores

    def _policy_single(self, features):
        if self.single_index is None:
            raise ValueError("single policy requires `single_index`.")
        alpha = torch.zeros_like(features["delta_norm"])
        alpha[..., int(self.single_index)] = 1.0
        return alpha, alpha, None

    def _policy_equal(self, features):
        alpha = torch.full_like(features["delta_norm"], 1.0 / self.num_specialists)
        scores = torch.ones_like(alpha)
        return alpha, scores, None

    def _policy_residual_softmax(self, features):
        scores = features["delta_norm"] / max(self.temperature, 1e-6)
        alpha = torch.softmax(scores, dim=-1)
        return alpha, features["delta_norm"], None

    def _policy_residual_scaled_softmax(self, features):
        normalized_scores = self._normalize_delta_scores(features["delta_norm"], mode="mean_ratio")
        alpha = torch.softmax(normalized_scores / max(self.temperature, 1e-6), dim=-1)
        return alpha, features["delta_norm"], normalized_scores

    def _policy_residual_logz_softmax(self, features):
        normalized_scores = self._normalize_delta_scores(features["delta_norm"], mode="log_zscore")
        alpha = torch.softmax(normalized_scores / max(self.temperature, 1e-6), dim=-1)
        return alpha, features["delta_norm"], normalized_scores

    def _policy_intervention_softmax(self, features):
        scores = features["intervention_norm"] / max(self.temperature, 1e-6)
        alpha = torch.softmax(scores, dim=-1)
        return alpha, features["intervention_norm"], None

    def _policy_topk_residual(self, features):
        scores = features["delta_norm"]
        topk = self.topk if self.topk is not None else 1
        topk = min(int(topk), self.num_specialists)
        _, indices = torch.topk(scores, k=topk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(dim=-1, index=indices, value=True)
        alpha = self._masked_softmax(scores / max(self.temperature, 1e-6), mask)
        return alpha, scores, None

    def _policy_compat_filtered_topk(self, features):
        scores = features["delta_norm"]
        pairwise_cos = features["pairwise_cos"]
        topk = self.topk if self.topk is not None else 1
        topk = min(int(topk), self.num_specialists)
        _, indices = torch.topk(scores, k=topk, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)

        batch_size, num_positions, _ = scores.shape
        for b in range(batch_size):
            for s in range(num_positions):
                chosen = []
                for idx in indices[b, s].tolist():
                    keep = True
                    for prev in chosen:
                        if float(pairwise_cos[b, s, idx, prev]) < self.compat_threshold:
                            keep = False
                            break
                    if keep:
                        chosen.append(idx)
                if not chosen:
                    chosen.append(int(indices[b, s, 0]))
                mask[b, s, chosen] = True

        alpha = self._masked_softmax(scores / max(self.temperature, 1e-6), mask)
        return alpha, scores, None

    def _policy_trainable(self, features):
        if self.policy_head is None:
            raise ValueError("Trainable policy requested without policy_head.")
        feature_tensor = torch.stack(
            [
                features["delta_norm"],
                features["intervention_norm"],
                features["subspace_energy"],
                features["mean_compat"],
                features["neg_compat_mass"],
            ],
            dim=-1,
        ).to(self.source_weight.dtype)
        logits = self.policy_head(feature_tensor).squeeze(-1).float()
        alpha = torch.softmax(logits, dim=-1)
        return alpha, logits, None

    def _compute_alpha(self, states, features):
        if self.use_trainable_policy and self.policy_type == "trainable":
            return self._policy_trainable(features)
        if self.policy_type == "single":
            return self._policy_single(features)
        if self.policy_type == "equal":
            return self._policy_equal(features)
        if self.policy_type == "residual_softmax":
            return self._policy_residual_softmax(features)
        if self.policy_type == "residual_scaled_softmax":
            return self._policy_residual_scaled_softmax(features)
        if self.policy_type == "residual_logz_softmax":
            return self._policy_residual_logz_softmax(features)
        if self.policy_type == "intervention_softmax":
            return self._policy_intervention_softmax(features)
        if self.policy_type == "topk_residual":
            return self._policy_topk_residual(features)
        if self.policy_type == "compat_filtered_topk":
            return self._policy_compat_filtered_topk(features)
        raise ValueError(f"Unsupported policy_type: {self.policy_type}")

    def _compose_output_space(self, states, alpha):
        lifted = states["Delta"].float()
        return torch.einsum("bst,bstd->bsd", alpha.float(), lifted)

    def _compose_projected_output(self, states, alpha):
        if self.shared_basis is None:
            raise ValueError("projected_output requires shared_basis.")
        raw_mix = self._compose_output_space(states, alpha)
        basis = self.shared_basis.float()
        coeff = torch.einsum("bsd,kd->bsk", raw_mix, basis)
        return torch.einsum("bsk,kd->bsd", coeff, basis)

    def _compose_shared_latent(self, states, alpha):
        if self.shared_basis is None:
            raise ValueError("shared_latent requires shared_basis.")
        basis = self.shared_basis.float()
        delta = states["delta"].float()

        if self.transport_weight is not None:
            transported = torch.einsum("bstr,tkr->bstk", delta, self.transport_weight.float())
        else:
            if basis.shape[0] != delta.shape[-1]:
                raise ValueError("identity shared_latent composition requires shared_basis_rank == low_rank_dimension.")
            transported = delta

        latent_mix = torch.einsum("bst,bstk->bsk", alpha.float(), transported)
        return torch.einsum("bsk,kd->bsd", latent_mix, basis)

    def _cache_debug_tensors(self, features, alpha, scores, normalized_score=None):
        self.latest_alpha = alpha.detach().cpu()
        self.latest_scores = scores.detach().cpu()
        self.latest_normalized_score = normalized_score.detach().cpu() if normalized_score is not None else None
        self.latest_delta_norm = features["delta_norm"].detach().cpu()
        self.latest_intervention_norm = features["intervention_norm"].detach().cpu()
        self.latest_pairwise_cos = features["pairwise_cos"].detach().cpu()
        selected_mask = alpha > 0
        self.latest_selected_mask = selected_mask.detach().cpu()
        pairwise = features["pairwise_cos"]
        alpha_pairs = alpha.unsqueeze(-1) * alpha.unsqueeze(-2)
        conflict = (alpha_pairs * torch.clamp(-pairwise, min=0.0)).sum(dim=(-1, -2))
        self.latest_conflict_score = conflict.detach().cpu()
        self.latest_compose_domain = self.compose_domain
        if self.shared_basis is not None:
            self.latest_shared_basis_stats = {
                "rank": int(self.shared_basis.shape[0]),
                "norm": float(self.shared_basis.float().norm().item()),
            }
        else:
            self.latest_shared_basis_stats = None
        if self.transport_weight is not None:
            self.latest_transport_stats = {
                "shape": tuple(self.transport_weight.shape),
                "norm": float(self.transport_weight.float().norm().item()),
            }
        else:
            self.latest_transport_stats = None

    def forward(self, base, source=None, subspaces=None, **kwargs):
        states = self._compute_specialist_states(base)
        features = self._extract_policy_features(states)
        alpha, scores, normalized_score = self._compute_alpha(states, features)

        if self.compose_domain == "output":
            mixed = self._compose_output_space(states, alpha)
        elif self.compose_domain == "projected_output":
            mixed = self._compose_projected_output(states, alpha)
        elif self.compose_domain == "shared_latent":
            mixed = self._compose_shared_latent(states, alpha)
        else:
            raise ValueError(f"Unsupported compose_domain: {self.compose_domain}")

        self._cache_debug_tensors(features, alpha, scores, normalized_score=normalized_score)
        output = base + mixed.to(base.dtype)
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        state_dict = OrderedDict()
        state_dict["rotate_weight"] = self.rotate_weight.data
        state_dict["source_weight"] = self.source_weight.data
        state_dict["source_bias"] = self.source_bias.data
        if self.shared_basis is not None:
            state_dict["shared_basis"] = self.shared_basis.data
        if self.transport_weight is not None:
            state_dict["transport_weight"] = self.transport_weight.data
        if self.policy_head is not None:
            for k, v in self.policy_head.state_dict().items():
                state_dict[f"policy_head.{k}"] = v
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        if "rotate_weight" in state_dict:
            self.rotate_weight.data.copy_(state_dict["rotate_weight"].to(self.rotate_weight.device))
        if "source_weight" in state_dict:
            self.source_weight.data.copy_(state_dict["source_weight"].to(self.source_weight.device))
        if "source_bias" in state_dict:
            self.source_bias.data.copy_(state_dict["source_bias"].to(self.source_bias.device))
        if self.shared_basis is not None and "shared_basis" in state_dict:
            self.shared_basis.data.copy_(state_dict["shared_basis"].to(self.shared_basis.device))
        if self.transport_weight is not None and "transport_weight" in state_dict:
            self.transport_weight.data.copy_(state_dict["transport_weight"].to(self.transport_weight.device))
        if self.policy_head is not None:
            policy_state = {
                k[len("policy_head."):]: v
                for k, v in state_dict.items()
                if k.startswith("policy_head.")
            }
            if policy_state:
                self.policy_head.load_state_dict(policy_state, strict=False)


class NoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NoReFT(h) = h + W2^T(W1h + b − W2h)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    ConsReFT(h) = h + R^T(b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)


class LobireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    LobiReFT(h) = h + R^T(b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        output = base + torch.matmul(
            self.learned_source, self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    DiReFT(h) = h + R^T(Wh + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        cast_base = base.to(self.learned_source.weight.dtype)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(cast_base))).to(self.rotate_layer.weight.dtype), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))


class NodireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    NodiReFT(h) = h + W2^T(W1h + b)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "act_fn" not in kwargs or kwargs["act_fn"] is None else ACT2FN[kwargs["act_fn"]]
        
    def forward(
        self, base, source=None, subspaces=None, **kwargs
    ):
        if base is None:
            raise ValueError("NodireftIntervention.forward expected `base` to be a Tensor, got None.")
        output = base + torch.matmul(
            self.act_fn(self.learned_source(base)), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))



class InternalRoutingFunction(nn.Module):
    def __init__(self, input_dim, num_total_subspaces, topk=2):
        super().__init__()
        self.num_total_subspaces = num_total_subspaces
        # Assuming routing is based on a pooled representation of the diff tensor
        self.topk = topk
        self.gate = nn.Sequential(
            # nn.LayerNorm(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_total_subspaces),
        )

    def forward(self, pooled_input_representation):
        # pooled_input_representation shape: (batch_size, input_dim)
        # if self.gate[0].weight.dtype != pooled_input_representation.dtype:
        #     self.gate = self.gate.to(dtype=pooled_input_representation.dtype, device=pooled_input_representation.device)
        scores = self.gate(pooled_input_representation) # scores shape: (batch_size, num_total_subspaces)
        return scores

class SubNodireftIntervention(NodireftIntervention):

    """
      This is a NodireReFT that supports subspace interventions with internal routing (Soft Assignment).

    """
    def __init__(self, num_total_subspaces, subspace_rank, topk=2, use_residual_gate=True, **kwargs):
        # The low_rank_dimension in kwargs is the dimension of diff
        super().__init__(**kwargs)
        self.num_total_subspaces = num_total_subspaces
        self.subspace_rank = subspace_rank
        self.use_residual_gate = use_residual_gate

        print(topk)
        print()
        # Instantiate the internal routing function
        self.routing_function = InternalRoutingFunction(
            input_dim=self.embed_dim, # Dimension of the input to the routing function (e.g. embed_dim)
            num_total_subspaces=num_total_subspaces, # Total number of available subspaces
            topk=topk
        ).to(self.learned_source.weight.dtype)

        if self.use_residual_gate:
            self.residual_gate_layer = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.Sigmoid()
            ).to(self.learned_source.weight.dtype)
            # self.residual_gate_layer.to(self.learned_source.weight.dtype)
        
    def freeze_except_routing_and_bias(self):
        self.proj_layer.weight.requires_grad = False
        self.learned_source.weight.requires_grad = False
        self.learned_source.bias.requires_grad = False

        for param in self.routing_function.parameters():
            param.requires_grad = True
            
        if self.use_residual_gate:
            for param in self.residual_gate_layer.parameters():
                param.requires_grad = True

    def forward(self, base, source=None, subspaces=None, **kwargs):
        if base is None:
            raise ValueError("SubNodireftIntervention.forward expected `base` to be a Tensor, got None.")

        # In this modified version, subspaces input is ignored.
        # The intervention will dynamically select dimensions using the routing function (Soft Assignment).
       
        # --- Dynamic Subspace Selection using Internal Routing (Soft Assignment) --- 
        # Assuming routing is based on a pooled representation of the diff tensor
        # base: shape (batch_size, sequence_length(px + lx), embed_dim)
        # origin/last_element can still be passed through kwargs for debugging.
        # They are not required by the intervention computation.
        origin = kwargs.get("origin", None)
        last_element = kwargs.get("last_element", None)
        # print(origin)
        # print(origin[0].shape)

        ######### use origin representation to get sentence embedding ########
        # last_element = torch.tensor(last_element, device=origin[0].device, dtype=torch.long) 
        # mask = torch.arange(origin[0].size(1), device=origin[0].device)[None, :] <= last_element[:, None]
        # mask = mask.unsqueeze(-1).float()
        # masked_embedding = origin[0] * mask
        # # print(masked_embedding.shape)
        # sentence_embeddings = masked_embedding[:, 1:, :].sum(dim=1) / last_element.unsqueeze(1)
        # print(sentence_embeddings.shape)
        # (batch_size, embed_dim)

        #######################################################################


        # print(torch.allclose(origin[0][:,1:8,], base[:,:7]))
        # print(torch.allclose(origin[0][0, last_element[0]-6:last_element[0]+1,], base[0,7:]))
        # print(torch.allclose(origin[0][1, last_element[1]-7:last_element[1],], base[1,7:]))
        
        #########  use base to get sentence embedding ########
        sentence_embeddings = torch.mean(base, dim=1, dtype=torch.bfloat16) # shape: (batch_size, embed_dim)
        ######################################################
        # 这里不是完整的prompt表征！！！！！oh no！！！！

        # Get raw scores from the internal routing function
        # The routing function expects input_dim to match the pooled_diff dimension (low_rank_dimension)
        raw_scores = self.routing_function(sentence_embeddings)
        self.raw_scores = raw_scores
        # print(self.raw_scores)
        
        topk_scores, topk_indices = torch.topk(raw_scores, k=self.routing_function.topk, dim=-1)
        # print(topk_scores)
        topk_weights = torch.softmax(topk_scores, dim=-1, dtype=torch.bfloat16)
        subspace_weights = torch.zeros_like(raw_scores, dtype=torch.bfloat16)
        # print(subspace_weights.dtype)
        # print(topk_weights.dtype)
        subspace_weights.scatter_(dim=-1, index=topk_indices, src=topk_weights)


        print("subspace_weights:",subspace_weights)
       
        subspace_weights_expanded = subspace_weights.unsqueeze(1).expand(-1, base.shape[1], -1)


        diff = self.act_fn(self.learned_source(base)).to(torch.bfloat16)
        diff = diff.view(diff.shape[0], diff.shape[1], self.num_total_subspaces, self.subspace_rank).to(torch.bfloat16)

        try:
            proj_weight_reshaped = self.proj_layer.weight.view(
                self.num_total_subspaces, self.subspace_rank, self.embed_dim
            )
        except RuntimeError as e:
            print(f"Error reshaping proj_layer.weight: {e}")
            print(f"Expected shape for reshape: ({self.num_total_subspaces}, {self.subspace_rank}, {self.embed_dim})")
            print(f"Actual proj_layer.weight shape: {self.proj_layer.weight.shape}")
            raise # Re-raise the error after printing debug info

        
        subspace_outputs = torch.einsum('bskd,kdi->bski', diff, proj_weight_reshaped)
        weighted_sum_output = torch.einsum('bsk,bski->bsi', subspace_weights_expanded, subspace_outputs)

         # --- Residual Gate Fusion ---
        if self.use_residual_gate:
            gate_input = torch.cat([base, weighted_sum_output], dim=-1).to(torch.bfloat16)
            gate = self.residual_gate_layer(gate_input)  # shape: (b, s, d)
            # print("gate:",gate)
            self.latest_gate = gate.detach().cpu()

            # print("gate shape:",gate.shape)
            output = gate * weighted_sum_output + (1 - gate) * base
        else:
            output = base + weighted_sum_output

        # output = base + weighted_sum_output

        return self.dropout(output.to(base.dtype))
