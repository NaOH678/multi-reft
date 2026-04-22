# Composable LoReFT Code Design

## Scope

这份文档把 `COMPOSABLE_LOREFT_PLAN.md` 进一步细化成代码级设计。

目标：

- 支持多个独立 LoReFT specialist 的 token-local 组合
- 同时支持三种组合后端：
  - `output`
  - `projected_output`
  - `shared_latent`
- 第一阶段支持 rule-based policy
- 第二阶段支持冻结 specialist、只训练 arbitration policy

约束：

- 保持原始 ReFT 语义
- 不做 prompt-fixed
- 不做跨 token pooling
- 每个位置只用该位置的 hidden state 决定组合

## Files

### New / Updated Files

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/interventions.py`
  - 新增 `ComposableLoreftIntervention`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/__init__.py`
  - 导出 `ComposableLoreftIntervention`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`
  - specialist 加载、shared basis 构造、模型组装
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/machine_ethics_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_toxicity/toxicity_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_bias/stereotype_exp.py`
  - 统一接入 composable loader
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/train_composer.py`
  - 第二阶段训练入口

## Core Abstraction

把组合模块拆成两个独立维度：

- `policy_type`
- `compose_domain`

这样实验矩阵自然变成：

- `single + output`
- `single + shared_latent`
- `residual_softmax + output`
- `residual_softmax + shared_latent`

而不是写死某个 policy 只服务某个后端。

## Data Model

### Specialist Definition

每个 specialist `t` 对应一个单独训练好的 LoReFT intervention。

对每一层，需要提取：

- `R_t`: rotate layer，shape `[d, r]`
- `W_t`: learned_source.weight，shape `[r, d]`
- `b_t`: learned_source.bias，shape `[r]`

其中：

- `d`: hidden size
- `r`: 单个 specialist 的 low-rank dimension

### Stacked Layout

在 `ComposableLoreftIntervention` 内部统一 stack 为：

- `rotate_weight`: `[T, d, r]`
- `source_weight`: `[T, r, d]`
- `source_bias`: `[T, r]`

其中：

- `T`: specialist 数量

### Shared Basis / Transport

若启用 `projected_output` 或 `shared_latent`，每层还需要：

- `shared_basis`: `U`，shape `[k, d]` 或 `[d, k]`
- `transport_weight`: `M`，shape `[T, k, r]`

建议内部统一使用：

- `shared_basis`: `[k, d]`
- `transport_weight`: `[T, k, r]`

这样：

- `delta_t`: `[r]`
- `M_t delta_t`: `[k]`
- `U^T (...)`: `[d]`

## Tensor Conventions

前向输入：

- `base`: `[B, S, d]`

其中：

- `B`: batch size
- `S`: 当前 intervention 生效的位置数
- `d`: hidden size

中间量建议统一为：

- `Rh`: `[B, S, T, r]`
- `Wh_plus_b`: `[B, S, T, r]`
- `delta`: `[B, S, T, r]`
- `Delta`: `[B, S, T, d]`
- `scores`: `[B, S, T]`
- `alpha`: `[B, S, T]`

若启用 shared-latent：

- `transported_delta`: `[B, S, T, k]`
- `latent_mix`: `[B, S, k]`
- `mixed`: `[B, S, d]`

若启用 compatibility：

- `pairwise_cos`: `[B, S, T, T]`

## How Stacked Specialist Weights Are Used At Inference

当前实现里，多个 specialist 的权重不是逐个循环调用，而是先在 loader 中沿着 specialist 维 stack，再在一次前向里并行计算。

### Stacked Parameter Shapes

在 `ComposableLoreftIntervention` 内部：

- `rotate_weight`: `[T, d, r]`
- `source_weight`: `[T, r, d]`
- `source_bias`: `[T, r]`

其中：

- `T`: specialist 数量
- `d`: hidden size
- `r`: 单个 specialist 的 LoReFT rank

这些 stacked tensor 是在 loader 中构造的：multi_train/eval_common/composable_loreft.py:190，而不是在 forward 时临时拼接。

具体就是：

  - `rotate_weight = torch.stack(layer_rotate, dim=0)`
  - `source_weight = torch.stack(layer_source_weight, dim=0)`
  - `source_bias = torch.stack(layer_source_bias, dim=0)`

### Input Tensor

当前层的 intervention 输入是：

- `base: [B, S, d]`

其中：

- `B`: batch size
- `S`: 当前层被干预的位置数
- `d`: hidden size

### Step 1: Compute All Specialists In Parallel

前向时先统一计算每个 specialist 在每个 token 位置上的 latent quantities。

计算：

- `rotated_base = einsum("bsd,tdr->bstr", base, rotate_weight)`
- `learned_source = einsum("bsd,trd->bstr", base, source_weight) + source_bias`

得到：

- `rotated_base: [B, S, T, r]`
- `learned_source: [B, S, T, r]`

然后：

- `delta = learned_source - rotated_base`

得到：

- `delta: [B, S, T, r]`

这就是每个 specialist 在每个位置上的 latent residual。

再把它们 lift 回原始表征空间：

- `Delta = einsum("bstr,tdr->bstd", delta, rotate_weight)`

得到：

- `Delta: [B, S, T, d]`

所以当前实现不是：

- 先选一个 specialist
- 再单独做一次 LoReFT

而是：

- 一次前向并行算出所有 specialist 的 `delta` 和 `Delta`

### Step 2: Compute Token-Local Alpha

然后从这些中间量提取 token-local 特征：

- `delta_norm: [B, S, T]`
- `intervention_norm: [B, S, T]`
- `pairwise_cos: [B, S, T, T]`

再根据 `policy_type` 计算：

- `alpha: [B, S, T]`

这意味着：

- 每个 token 位置都有自己的一组 specialist 权重
- 不是整条样本共享一组 `alpha`

### Step 3: Compose By Domain

#### Output Domain

若 `compose_domain == "output"`：

- `mixed = einsum("bst,bstd->bsd", alpha, Delta)`

得到：

- `mixed: [B, S, d]`

最后：

- `output = base + mixed`

#### Projected Output Domain

若 `compose_domain == "projected_output"`：

1. 先得到 output-space 的 `mixed`
2. 再把 `mixed` 投影到 shared basis 对应的子空间

最终仍然得到：

- `mixed: [B, S, d]`

#### Shared Latent Domain

若 `compose_domain == "shared_latent"`：

先在 shared latent 中组合 `delta`：

- 若存在 transport：
  - `transported = einsum("bstr,tkr->bstk", delta, transport_weight)`
- 再：
  - `latent_mix = einsum("bst,bstk->bsk", alpha, transported)`
- 最后 lift 回原空间：
  - `mixed = einsum("bsk,kd->bsd", latent_mix, shared_basis)`

最终仍然得到：

- `mixed: [B, S, d]`

### Inference Pipeline
  一层里、一次 intervention 前向，实际就是：

  1. 输入 `base: [B,S,d]`
  2. 并行算出所有 specialist 的 `delta: [B,S,T,r]`
  3. 并行算出所有 specialist 的 `Delta: [B,S,T,d]`
  4. 根据 policy 算 `alpha: [B,S,T]`
  5. 按 compose_domain 组合成 `mixed: [B,S,d]`
  6. 返回 `base + mixed`

### Practical Interpretation

可以把当前实现理解成：

- 多个 specialist 权重被打包成一个大 intervention 模块
- 每个 token 位置在一次前向里同时看所有 specialists
- 先并行计算所有 specialist 的 rewrite 候选
- 再在 token 级别按 `alpha` 做组合

因此，当前推理逻辑的本质不是“选一个 specialist 再推理”，而是：

- “并行生成多个 specialist 的局部 rewrite”
- “再根据当前 token 的 score/policy 做加权组合”

## Class Design

### Class Signature

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/interventions.py`

建议签名：

```python
class ComposableLoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
):
    def __init__(
        self,
        embed_dim: int,
        low_rank_dimension: int,
        num_specialists: int,
        compose_domain: str = "output",
        policy_type: str = "equal",
        temperature: float = 1.0,
        topk: int | None = None,
        compat_threshold: float = 0.0,
        single_index: int | None = None,
        shared_basis_rank: int | None = None,
        shared_basis_type: str | None = None,
        transport_type: str | None = None,
        use_trainable_policy: bool = False,
        policy_hidden_dim: int = 32,
        dropout: float = 0.0,
        add_bias: bool = False,
        dtype=None,
        **kwargs,
    ):
        ...
```

### Internal Parameters

建议成员字段：

- `self.embed_dim`
- `self.low_rank_dimension`
- `self.num_specialists`
- `self.compose_domain`
- `self.policy_type`
- `self.temperature`
- `self.topk`
- `self.compat_threshold`
- `self.single_index`
- `self.shared_basis_rank`
- `self.shared_basis_type`
- `self.transport_type`
- `self.use_trainable_policy`

specialist 权重：

- `self.rotate_weight: nn.Parameter`
- `self.source_weight: nn.Parameter`
- `self.source_bias: nn.Parameter`

shared-latent 相关：

- `self.shared_basis: nn.Parameter | None`
- `self.transport_weight: nn.Parameter | None`

trainable policy：

- `self.policy_head: nn.Module | None`

日志缓存：

- `self.latest_alpha`
- `self.latest_scores`
- `self.latest_delta_norm`
- `self.latest_intervention_norm`
- `self.latest_pairwise_cos`
- `self.latest_selected_mask`
- `self.latest_conflict_score`

## Forward Structure

### Public Forward

```python
def forward(self, base, source=None, subspaces=None, **kwargs):
    states = self._compute_specialist_states(base)
    features = self._extract_policy_features(states)
    alpha = self._compute_alpha(states, features)

    if self.compose_domain == "output":
        mixed = self._compose_output_space(states, alpha)
    elif self.compose_domain == "projected_output":
        mixed = self._compose_projected_output(states, alpha)
    elif self.compose_domain == "shared_latent":
        mixed = self._compose_shared_latent(states, alpha)
    else:
        raise ValueError(...)

    self._cache_debug_tensors(states, features, alpha, mixed)
    return (base + mixed).to(base.dtype)
```

### `_compute_specialist_states`

输入：

- `base: [B, S, d]`

输出建议是一个 dict：

```python
{
    "base": base,
    "Rh": Rh,
    "Wh_plus_b": Wh_plus_b,
    "delta": delta,
    "Delta": Delta,
}
```

### `_extract_policy_features`

建议输出：

```python
{
    "delta_norm": [B, S, T],
    "intervention_norm": [B, S, T],
    "subspace_energy": [B, S, T],
    "pairwise_cos": [B, S, T, T],   # optional
    "mean_compat": [B, S, T],        # optional
    "neg_compat_mass": [B, S, T],    # optional
}
```

说明：

- 第一版 `pairwise_cos` 可用 `Delta` 计算
- 后续若 shared-latent 更有信号，再补 latent-side cosine

### `_compute_alpha`

输入：

- `states`
- `features`

输出：

- `alpha: [B, S, T]`

内部根据 `policy_type` 分派到：

- `_policy_single`
- `_policy_equal`
- `_policy_residual_softmax`
- `_policy_intervention_softmax`
- `_policy_topk_residual`
- `_policy_compat_filtered_topk`
- `_policy_trainable`

### `_compose_output_space`

公式：

```math
mixed = \sum_t \alpha_t \Delta_t
```

输出：

- `mixed: [B, S, d]`

### `_compose_projected_output`

公式：

```math
mixed = P_U\left(\sum_t \alpha_t \Delta_t\right)
```

实现建议：

1. 先得到 `raw_mix: [B, S, d]`
2. 投影到 `U` 张成的空间

若 `shared_basis` 采用 `[k, d]`：

- latent coeffs: `coeff = einsum("bsd,kd->bsk", raw_mix, U)`
- projected: `mixed = einsum("bsk,kd->bsd", coeff, U)`

前提：

- `U` 已正交化

### `_compose_shared_latent`

公式：

```math
latent_mix = \sum_t \alpha_t M_t \delta_t
mixed = U^\top latent_mix
```

输出：

- `mixed: [B, S, d]`

说明：

- 若 `transport_type == "identity"`，要求 `k == r`
- 若 `transport_type == "overlap"`，用预构造好的 `M`

## Policy Implementations

### `single`

输出固定 one-hot：

- `alpha[..., single_index] = 1`

### `equal`

输出常数：

- `alpha = 1 / T`

### `residual_softmax`

分数：

- `scores = ||delta||_2`

输出：

- `alpha = softmax(scores / temperature)`

### `intervention_softmax`

分数：

- `scores = ||Delta||_2`

输出：

- `alpha = softmax(scores / temperature)`

### `topk_residual`

分数：

- `scores = ||delta||_2`

逻辑：

1. 选 top-k
2. top-k 外设为 `-inf`
3. top-k 内做 softmax

### `compat_filtered_topk`

逻辑：

1. 先按 residual 取 top-k
2. 用 `pairwise_cos` 做 greedy filtering
3. 对保留集合按 residual 再 softmax

第一版只实现 greedy version，不做 combinatorial search。

### `trainable`

若 `use_trainable_policy=True`，则从 `features` 中组装：

- `[delta_norm, intervention_norm, subspace_energy, mean_compat, neg_compat_mass]`

形成：

- `policy_features: [B, S, T, F]`

再经 `policy_head` 得到：

- `scores: [B, S, T]`
- `alpha = softmax(scores, dim=-1)`

## Shared Basis Construction

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`

### Specialist State Loader

建议定义：

```python
def load_loreft_specialist_state(weight_dir: str) -> dict:
    """
    Return:
        {
            layer_key: {
                "rotate_weight": Tensor[d, r],
                "source_weight": Tensor[r, d],
                "source_bias": Tensor[r],
            }
        }
    """
```

### Layer Basis Builder

建议定义：

```python
def build_shared_basis_for_layer(
    rotate_weights: list[torch.Tensor],
    basis_type: str,
    basis_rank: int,
) -> torch.Tensor:
    """
    Return:
        shared_basis: Tensor[k, d]
    """
```

支持：

- `orth_mean`
- `svd_union`

#### `orth_mean`

步骤：

1. `R_bar = mean_t R_t`
2. 对 `R_bar` 做正交化
3. 得到 `U`

#### `svd_union`

步骤：

1. 拼接多个 `R_t`
2. 做 SVD / PCA
3. 取前 `k` 个方向

推荐作为更主的 shared basis。

### Layer Transport Builder

建议定义：

```python
def build_transport_for_layer(
    rotate_weights: list[torch.Tensor],
    shared_basis: torch.Tensor,
    transport_type: str,
) -> torch.Tensor:
    """
    Return:
        transport_weight: Tensor[T, k, r]
    """
```

支持：

- `identity`
- `overlap`

#### `identity`

要求：

- `k == r`

返回：

- `M_t = I`

#### `overlap`

第一版可用最简单构造：

- `M_t ≈ U @ R_t`

具体转置方向以最终张量 convention 为准，只要语义是“从 specialist latent 映射到 shared latent”。

## Model Builder API

建议暴露统一入口：

```python
def load_composed_reft_model(
    base_model_name_or_path: str,
    specialist_dirs: list[str],
    target_layers: list[int],
    subspace_rank: int,
    compose_domain: str,
    composition_method: str,
    composition_temperature: float,
    composition_topk: int | None,
    compat_threshold: float,
    single_index: int | None,
    shared_basis_type: str | None,
    shared_basis_rank: int | None,
    transport_type: str | None,
    device: str,
):
    """
    Return:
        tokenizer, reft_model
    """
```

eval 脚本只负责传参数，不自己组装 specialists。

## Eval Script Integration

四个 eval 入口统一新增参数：

- `--reft_specialists`
- `--composition_method`
- `--compose_domain`
- `--composition_temperature`
- `--composition_topk`
- `--compat_threshold`
- `--single_index`
- `--shared_basis_type`
- `--shared_basis_rank`
- `--transport_type`

加载逻辑统一为：

1. 若传 `--reft_specialists`，调用 `load_composed_reft_model`
2. 否则若传 `--reft_weights`，走单个 Loreft
3. 否则若传 `--lora_weights`，走 LoRA
4. 否则走 base model

### Special Note: Toxicity

`/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_toxicity/toxicity_exp.py`

当前是 `NodireftIntervention`，接入 composable 路线前应先统一回 Loreft 体系。

## State Dict Design

### Intervention Save

`ComposableLoreftIntervention.state_dict()` 应包含：

- `rotate_weight`
- `source_weight`
- `source_bias`
- `shared_basis`（若有）
- `transport_weight`（若有）
- `policy_head.*`（若有）

### Metadata

建议同时保存 metadata json：

- `num_specialists`
- `specialist_names`
- `policy_type`
- `compose_domain`
- `shared_basis_type`
- `shared_basis_rank`
- `transport_type`
- `temperature`
- `topk`
- `compat_threshold`

这样结果可复现。

## Freeze / Train Strategy

### Phase 1

无训练组合：

- specialists 冻结
- shared basis / transport 固定
- policy 为 heuristic

### Phase 2

冻结 specialists，只训练 policy：

```python
intervention.freeze_specialists()
intervention.unfreeze_policy()
```

后续如 shared-latent 有效，可扩展成：

- 固定 `U`，训练 `policy + M_t`

## `train_composer.py`

建议入口参数：

- `--specialist_dirs`
- `--compose_domain`
- `--composition_method`
- `--shared_basis_type`
- `--shared_basis_rank`
- `--transport_type`
- `--policy_hidden_dim`
- `--freeze_specialists`
- `--train_transport`
- `--conflict_loss_weight`

第一版 loss：

- 只用普通 LM loss

可选后续加入：

```math
L = L_{LM} + \lambda L_{conflict}
```

其中：

```math
L_{conflict} = \sum_{i<j}\alpha_i\alpha_j\max(0,-c_{ij})
```

## Debug and Logging

`ComposableLoreftIntervention` 中应缓存：

- `latest_alpha`
- `latest_scores`
- `latest_delta_norm`
- `latest_intervention_norm`
- `latest_pairwise_cos`
- `latest_selected_mask`
- `latest_conflict_score`
- `latest_compose_domain`

若启用 shared basis / transport：

- `latest_shared_basis_stats`
- `latest_transport_stats`

eval summary 建议记录：

- compose domain
- policy type
- specialist dirs
- temperature / topk / threshold
- shared basis type / rank
- transport type

## Implementation Order

### Step 1

在 `interventions.py` 中实现：

- stacked specialist storage
- `_compute_specialist_states`
- `_extract_policy_features`
- `_compute_alpha`
- `_compose_output_space`

先只支持：

- `single`
- `equal`
- `residual_softmax`
- `topk_residual`
- `compose_domain=output`

### Step 2

实现 `eval_common/composable_loreft.py`：

- specialist state 加载
- composable intervention 组装

先接通 `eval_truth`

### Step 3

实现：

- `_compose_projected_output`
- `build_shared_basis_for_layer`

先支持：

- `shared_basis_type=orth_mean`
- `compose_domain=projected_output`

### Step 4

实现：

- `compose_domain=shared_latent`
- `transport_type=identity`

### Step 5

实现：

- `shared_basis_type=svd_union`
- `transport_type=overlap`

### Step 6

实现：

- `compat_filtered_topk`
- debug logging
- 其余三个 eval 接入

### Step 7

实现：

- `train_composer.py`
- trainable policy

## First Experiment Set

建议第一批先跑：

- `single + output`
- `equal + output`
- `residual_softmax + output`
- `topk_residual + output`
- `equal + shared_latent + orth_mean + identity`
- `residual_softmax + shared_latent + orth_mean + identity`
- `residual_softmax + shared_latent + svd_union + identity`
- `residual_softmax + shared_latent + svd_union + overlap`
- `residual_softmax + projected_output + svd_union`

## Non-Goals for First Version

第一版先不做：

- prompt-fixed routing
- generation-time global arbitration
- cross-token pooling
- jointly training specialists
- learning shared basis `U`

这些都应放到当前框架稳定之后。

## Future Shared-Latent Upgrades

当前 shared-latent 只是第一版无训练实现。

目前已经支持：

- `shared_basis_type = orth_mean`
- `shared_basis_type = svd_union`
- `transport_type = identity`
- `transport_type = overlap`

它们的共同特点是：

- `U` 是固定构造的
- `M_t` 是固定构造的
- 没有通过训练显式优化 latent 对齐

因此这套 shared-latent 更接近：

- 一个结构上合理的 heuristic

而不是：

- 真正学出来的共享 latent 机制

### 1. Learned Transport

当前 transport 为：

- `identity`
- `overlap`

它们都是固定映射。

`learned transport` 的含义是：

- 不再把 `M_t` 固定死
- 而是让每个 specialist 到共享 latent 的映射 `M_t` 成为可训练参数

公式仍然是：

```math
h' = h + U^\top \left(\sum_t \alpha_t M_t \delta_t(h)\right)
```

但这里的 `M_t` 不再是手工构造，而是训练得到。

它解决的核心问题是：

- 不同 specialist 的 latent 坐标系并不天然兼容
- 固定的 overlap 映射可能不够好
- 如果 `M_t` 可学习，它就能主动把各 specialist 的 latent residual 对齐到更可组合的共享空间中

工程上，这通常是 shared-latent 路线中最值得优先尝试的升级。

### 2. Learned Shared Basis

当前 shared basis `U` 为：

- `orth_mean`
- `svd_union`

它们也都是固定构造。

`learned shared basis` 的含义是：

- `U` 不再完全由平均 / SVD 给定
- 而是作为可训练参数，或在固定初始化基础上进一步优化

它解决的核心问题是：

- 静态几何上“共同覆盖的方向”
- 不一定等于“最利于 rewrite 组合的方向”

也就是说：

- `svd_union` 找到的是联合主方向
- 但模型真正需要的共享空间，可能是更适合安全叠加多个 rewrite 的子空间

因此 learned shared basis 是更强、更灵活的一步，但训练难度和不稳定性也会更高。

### 3. Stricter Latent Alignment Mechanisms

这是一类更上层的目标，不是单一一种方法。

它的核心是：

- 不只是构造共享空间
- 还要确保不同 specialist 映射到共享空间后的坐标真正可比较、可加和、可组合

可能的实现方向包括：

1. 更强的 transport 约束

- 让 `M_t` 接近正交
- 或限制 `M_t` 的条件数
- 避免 transport 产生任意扭曲

2. 跨 specialist 的对齐损失

- 约束不同 specialist transport 后的 latent 表示更一致
- 提高共享空间中的可比较性

3. shared latent consistency 正则

- 对 `U^\top M_t delta_t` 的统计结构加约束
- 减少不同 specialist 在共享空间中的冲突

4. token-dependent transport

- 让 `M_t(h)` 随输入变化
- 这是更强但更复杂的版本

所以“更严格的 latent alignment”是一个总方向，通常要通过：

- learned transport
- learned shared basis
- 额外 alignment regularization

共同实现。

## Recommended Upgrade Priority For Shared-Latent

如果第一阶段实验显示 shared-latent 有信号，后续建议按这个顺序推进。

### Stage A. Keep `U` Fixed, Learn Only `M_t`

推荐优先级最高。

做法：

- 固定 specialist 参数
- 固定 shared basis `U`
- 只训练 transport `M_t`
- `alpha` 仍可先用 heuristic，或和 trainable policy 一起训练

原因：

- 实现代价相对低
- 能直接测试“问题是不是主要出在 latent 对齐不够好”
- 风险比同时学习 `U` 更低

### Stage B. Keep `U` Fixed, Learn `alpha + M_t`

第二优先级。

做法：

- 固定 specialist 参数
- 固定 `U`
- 同时训练：
  - arbitration policy
  - transport `M_t`

原因：

- 让“选谁”和“怎么对齐”一起适配
- 但训练变量明显增加

### Stage C. Learn `U` As Well

第三优先级。

做法：

- specialist 参数继续固定
- 训练：
  - `U`
  - 可选 `M_t`
  - 可选 policy

原因：

- 灵活性最强
- 但最容易不稳定
- 在没有明确 shared-latent 信号前，不建议直接做

### Stage D. Add Alignment Regularization

第四优先级。

做法：

- 在 Stage A/B/C 的基础上再加：
  - orthogonality regularization
  - transport smoothness / conditioning constraints
  - shared latent consistency losses
  - conflict-aware regularization

原因：

- 这是进一步提高 shared-latent 质量的手段
- 但前提是前面的 learned transport / learned basis 已经证明有价值

## Practical Recommendation

不建议一开始直接做：

- learned shared basis
- 强 alignment regularization

因为当前最关键的问题还没有回答：

- shared-latent 这个方向本身有没有稳定信号

所以推荐路线是：

1. 先跑当前无训练版
   - `orth_mean / svd_union`
   - `identity / overlap`
2. 若 shared-latent 明显优于 output-space
   - 先做 `learned transport`
3. 若仍有提升空间
   - 再考虑 `learned shared basis`
4. 最后再加更严格的 latent alignment 正则

一句话概括：

- 第一阶段验证 shared-latent 有没有信号
- 第二阶段优先学习 transport
- shared basis 和更强 alignment 放在更后面
