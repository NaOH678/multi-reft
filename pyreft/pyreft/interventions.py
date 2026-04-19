import torch
import torch.nn as nn
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
