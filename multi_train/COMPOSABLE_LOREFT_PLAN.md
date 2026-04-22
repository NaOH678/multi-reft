# Composable LoReFT Implementation Plan

## Goal

在保持原始 ReFT 语义不变的前提下，实现多个独立 LoReFT specialist 的无训练组合，并为后续第二阶段 arbitration training 预留统一接口。

新的实现采用双分支组合框架：

- `output-space composition`
- `shared-latent composition`

另加一个过渡性 baseline：

- `projected-output composition`

约束：

- 不做 prompt-fixed
- 不做跨 token pooling
- 不让前面位置依赖后面位置
- 每个被干预位置只用该位置当前 hidden state 做组合决策

形式上，对每个被干预位置 `j`：

```math
\Delta h_t(h_j)=R_t^\top(W_t h_j+b_t-R_t h_j)
```

```math
\alpha_t(h_j)=\text{policy}(h_j)
```

```math
h'_j=h_j+\sum_t \alpha_t(h_j)\Delta h_t(h_j)
```

## High-Level Design

实现三层结构：

1. `ComposableLoreftIntervention`
2. 公共加载工具 `multi_train/eval_common/composable_loreft.py`
3. 第二阶段训练入口 `multi_train/train_composer.py`

其中：

- 第一阶段：policy 使用 rule-based heuristic，不训练
- 第二阶段：冻结 specialist，只训练 arbitration policy
- 第一阶段内部再区分不同 `compose_domain`

## Core Principle

组合必须是 token-local，而不是 sample-level。

输入 `base` 的形状为：

- `base: [B, S, d]`

其中：

- `B`: batch size
- `S`: 当前层被干预的位置数
- `d`: hidden size

对于每个位置 `(b, s)`，只允许使用该位置自己的 `h_{b,s}` 计算组合权重。

因此实现必须避免：

- prompt-fixed
- 跨 token pooling
- 用后面位置的信息回写前面位置

## Unified Forward Decomposition

无论使用哪个组合分支，前向都先统一计算每个 specialist 的基础量。

对每个 specialist `t`、每个位置 `j`：

```math
\delta_t(h_j)=W_t h_j+b_t-R_t h_j
```

```math
\Delta h_t(h_j)=R_t^\top \delta_t(h_j)
```

然后前向拆成两部分：

1. `policy`
2. `compose_domain`

也就是说：

- `policy_type` 决定 `alpha_t(h_j)` 怎么来
- `compose_domain` 决定多个 specialist 的 rewrite 在哪个空间里组合

推荐统一实现成四段：

1. `_compute_specialist_states(base)`
2. `_compute_alpha(features)`
3. `_compose_output_space(...)`
4. `_compose_shared_latent(...)`

可选再加：

5. `_compose_projected_output(...)`

## New Intervention Class

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/interventions.py`

新增类：

- `ComposableLoreftIntervention`

职责：

- 持有多个 Loreft specialist 参数
- 对每个 token 位置逐位置计算 `delta_t` 与 `Delta_t`
- 根据指定 policy 逐位置计算 `alpha_t`
- 根据 `compose_domain` 选择组合后端
- 输出加权后的混合 rewrite
- 为第二阶段训练提供冻结与 trainable policy 接口

## Parameter Layout

单个 specialist `t` 的参数：

- `R_t`: rotate matrix，shape `[d, r]`
- `W_t`: learned_source.weight，shape `[r, d]`
- `b_t`: learned_source.bias，shape `[r]`

为便于并行计算，内部统一 stack：

- `rotate_weight`: `[T, d, r]`
- `source_weight`: `[T, r, d]`
- `source_bias`: `[T, r]`

其中：

- `T`: specialist 数量
- `r`: LoReFT subspace rank

## Forward Computation

给定：

- `base: [B, S, d]`

先对所有 specialist 并行计算：

1. `Rh`
2. `Wh + b`
3. `delta = Wh + b - Rh`
4. `Delta = R^T delta`

推荐张量形状：

- `delta: [B, S, T, r]`
- `Delta: [B, S, T, d]`

然后计算每个位置的分数：

- `scores: [B, S, T]`

最后得到：

- `alpha: [B, S, T]`
- `mixed: [B, S, d]` 或共享 latent 下的中间结果
- `output = base + mixed`

整个过程必须保持 token-local，不做跨位置信息汇聚。

## Policy Layer

两个组合分支尽量共用同一套 token-local policy。

也就是说：

- `single + output`
- `single + shared_latent`

是一组对应实验；

- `residual_softmax + output`
- `residual_softmax + shared_latent`

也是一组对应实验。

因此代码抽象上必须把：

- `policy_type`
- `compose_domain`

做成两个独立开关，而不是把某个 policy 写死在某个组合分支里。

## Phase-1 No-Training Policies

第一版先支持以下组合规则：

- `single`
- `equal`
- `residual_softmax`
- `intervention_softmax`
- `topk_residual`
- `compat_filtered_topk`

对应含义：

- `single`: 固定只启用一个 specialist
- `equal`: 所有 specialist 等权
- `residual_softmax`: 按 `||delta_t(h_j)||` softmax
- `intervention_softmax`: 按 `||Delta_t(h_j)||` softmax
- `topk_residual`: 按 `||delta_t(h_j)||` 取 top-k 再归一化
- `compat_filtered_topk`: 先 top-k，再按 compatibility 过滤

说明：

- 这些 policy 主要决定 `alpha_t(h_j)`
- 它们原则上既可用于 `output-space`，也可用于 `shared-latent`
- 第一版可以尽量共用，后续如果发现某些 score 更适合某个 domain，再单独扩展

## Composition Domains

### 1. Output-Space Composition

这是基础 baseline，直接在原始表征空间中组合各 specialist 的 rewrite：

```math
h'_j = h_j + \sum_t \alpha_t(h_j)\Delta h_t(h_j)
```

特点：

- 实现最直接
- 不需要额外 basis / transport
- 是所有组合实验的主 baseline

### 2. Projected-Output Composition

这是一个过渡性 baseline。先在原空间组合，再把结果限制到共享子空间：

```math
h'_j = h_j + P_U\left(\sum_t \alpha_t(h_j)\Delta h_t(h_j)\right)
```

其中：

- `U` 是共享 basis
- `P_U` 是投影到共享子空间的投影算子

用途：

- 判断 output-space mixture 是否只是“太散”
- 判断把最终 rewrite 限制在共享子空间里是否已经有帮助

### 3. Shared-Latent Composition

这是新的 principal 分支。先在共享 latent basis 里组合，再 lift 回原空间：

```math
h'_j = h_j + U^\top \left(\sum_t \alpha_t(h_j)\,M_t\,\delta_t(h_j)\right)
```

其中：

- `U`：共享 latent basis
- `M_t`：把第 `t` 个 specialist 的 latent residual transport 到共享空间的映射

关键点：

- 组合对象不再是 `Delta_t`
- 而是经过对齐后的 `delta_t`

## Shared-Latent Variants

不能直接把不同 specialist 的 latent residual 生硬相加，因为不同 `R_t` 对应的 latent 坐标系通常不一致。

因此 shared-latent 分支必须显式考虑：

- shared basis `U`
- transport `M_t`

第一版建议分三档：

### SL1. Orth-Mean Baseline

构造：

- `U = orth(mean_t R_t)`
- `M_t = I`

组合：

```math
h'_j = h_j + U^\top \left(\sum_t \alpha_t(h_j)\delta_t(h_j)\right)
```

说明：

- 这是很粗的 baseline
- 理论上不严格，但实现便宜
- 用来判断“只要改成 latent-space combine，是否已有趋势”

### SL2. Shared Basis + Identity Transport

构造：

- 对 `[R_1, R_2, ..., R_T]` 做 SVD / PCA
- 取前 `k` 个方向得到 `U`
- 仍设 `M_t = I`

组合：

```math
h'_j = h_j + U^\top \left(\sum_t \alpha_t(h_j)\delta_t(h_j)\right)
```

说明：

- 比 `orth(mean(R_t))` 更合理
- 但仍默认各 specialist latent 坐标近似兼容

### SL3. Shared Basis + Transported Latent

构造：

- `U` 用 `svd_union` 得到
- `M_t` 用 `U` 与 `R_t` 的 overlap 构造

组合：

```math
h'_j = h_j + U^\top \left(\sum_t \alpha_t(h_j)\,M_t\,\delta_t(h_j)\right)
```

第一版最简单可试：

```math
M_t \approx U R_t^\top
```

具体乘法方向在代码里按张量 convention 调整。

说明：

- 这是 shared-latent 分支里最值得试的版本
- 也是后续第二阶段训练最容易继续扩展的版本

## Compatibility Computation

对每个位置，`Delta` 为 `[T, d]`，可计算 pairwise cosine：

```math
c_{ij}=\cos(\Delta_i,\Delta_j)
```

得到：

- `pairwise_cos: [B, S, T, T]`

`compat_filtered_topk` 的第一版可用 greedy filter：

1. 先按 residual 选 top-k
2. 依次加入候选 specialist
3. 如果与已选集合中任意 specialist 的 cosine 小于阈值，则剔除
4. 对保留集合重新 softmax

## Feature Extraction for Future Training

即使第一阶段先不用 trainable policy，也应统一实现 feature extraction。

建议输出：

- `delta_norm: [B, S, T]`
- `intervention_norm: [B, S, T]`
- `subspace_energy: [B, S, T]`
- `mean_compat: [B, S, T]`
- `neg_compat_mass: [B, S, T]`

后续 trainable policy 可直接复用这些 feature。

说明：

- 第一版 compatibility 仍可基于 `Delta` 来算
- 后续如果 shared-latent 分支表现出更强信号，再补 latent-side compatibility

## Phase-2 Trainable Policy

第二阶段只训练 arbitration，不改 specialist。

建议最小实现：

- `policy_type = trainable_mlp`
- `policy_mlp: Linear(F, H) -> ReLU/GELU -> Linear(H, 1)`

输入：

- 每个 specialist、每个位置的 feature，形状 `[B, S, T, F]`

输出：

- `score: [B, S, T]`
- `alpha = softmax(score, dim=-1)`

## Freeze Strategy

`ComposableLoreftIntervention` 应提供：

- `freeze_specialists()`
- `freeze_policy()`
- `unfreeze_policy()`

其中 `freeze_specialists()` 冻结：

- `rotate_weight`
- `source_weight`
- `source_bias`

第二阶段训练时只更新 policy。

如果后续 shared-latent 方向有效，再额外考虑：

- 固定 `U`，训练 `M_t`
- 固定 specialists，训练 `alpha + M_t`

## State Persistence

不要只保存 policy，也不要只保存 specialists。应支持完整恢复组合模型。

建议保存：

- stacked specialist 参数
- policy 配置
- policy 参数
- specialist 名称顺序

可包含：

- `rotate_weight`
- `source_weight`
- `source_bias`
- `policy_head.*`
- `metadata.specialist_names`
- `metadata.policy_type`

metadata 可以单独存 json，也可以复用现有 intervention 保存目录。

## Common Loader

新文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`

职责：

- 从多个单任务 Loreft checkpoint 读取每层参数
- 为每一层构造 `ComposableLoreftIntervention`
- 逐层构造共享 basis `U`
- 可选逐层构造 transport `M_t`
- 返回统一的 reft model

建议函数：

- `load_loreft_specialist_state(weight_dir)`
- `build_composable_reft_config(base_model, specialist_states, args)`
- `load_composed_reft_model(base_model, specialist_dirs, args)`

建议额外加入：

- `build_shared_basis_for_layer(...)`
- `build_transport_for_layer(...)`

shared-latent / projected-output 相关配置建议包括：

- `shared_basis_type`
  - `orth_mean`
  - `svd_union`
- `shared_basis_rank`
- `transport_type`
  - `identity`
  - `overlap`

## Eval Integration

以下评测入口统一接入公共 loader：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/machine_ethics_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_toxicity/toxicity_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_bias/stereotype_exp.py`

需要新增 CLI 参数：

- `--reft_specialists path1 path2 path3 path4`
- `--composition_method`
- `--compose_domain {output,projected_output,shared_latent}`
- `--composition_temperature`
- `--composition_topk`
- `--compat_threshold`
- `--single_index`
- `--shared_basis_type {orth_mean,svd_union}`
- `--shared_basis_rank`
- `--transport_type {identity,overlap}`

兼容逻辑：

- 若传 `--reft_specialists`，走 composable loader
- 否则若传 `--reft_weights`，走单个 Loreft
- 否则走 base / LoRA

额外注意：

- `eval_toxicity/toxicity_exp.py` 当前还在用 `NodireftIntervention`，需要统一回 Loreft 体系

## Logging and Analysis Hooks

第一阶段就应缓存以下中间量，便于后续分析：

- `latest_alpha`
- `latest_delta_norm`
- `latest_intervention_norm`
- `latest_pairwise_cos`
- `latest_selected_mask`
- `latest_conflict_score`

如果启用 shared-latent / projected-output，再额外缓存：

- `latest_compose_domain`
- `latest_shared_basis_stats`
- `latest_transport_stats`

eval summary 至少应记录：

- composition method
- compose domain
- specialist list
- temperature / topk / threshold
- shared basis type / rank
- transport type
- alpha summary
- conflict summary

## Phase-2 Training Script

新文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/train_composer.py`

功能：

- 加载 base model
- 加载多个 frozen Loreft specialist
- 构造 composable intervention
- 只训练 arbitration policy

新增训练参数建议包括：

- `specialist_dirs`
- `policy_type=trainable_mlp`
- `policy_hidden_dim`
- `freeze_specialists=true`
- `conflict_loss_weight`

第一版 loss：

- 主 loss 使用普通 LM loss

后续可选加入：

```math
L = L_{LM} + \lambda L_{conflict}
```

其中：

```math
L_{conflict} = \sum_{i<j}\alpha_i\alpha_j\max(0,-c_{ij})
```

如果 shared-latent 分支有效，第二阶段可以继续扩展为：

- 固定 `U` 和 `M_t`，只学 `alpha`
- 固定 `U`，学习 `alpha + M_t`

第一阶段实现时应为这两条路线预留参数与 state_dict 接口。

## Recommended Implementation Order

建议按以下顺序推进：

1. 新增 `ComposableLoreftIntervention`
2. 新增公共 loader
3. 先实现 `output-space` baseline
4. 再加入 `projected-output` 与 `shared-latent`
5. 先接通 `eval_truth`
6. 跑通：
   - `single + output`
   - `equal + output`
   - `residual_softmax + output`
   - `topk_residual + output`
7. 再加入 shared-latent 对照：
   - `equal + shared_latent + orth_mean + identity`
   - `residual_softmax + shared_latent + orth_mean + identity`
   - `residual_softmax + shared_latent + svd_union + identity`
   - `residual_softmax + shared_latent + svd_union + overlap`
8. 加入 `projected_output` 对照：
   - `residual_softmax + projected_output + svd_union`
9. 加入 compatibility 统计与 `compat_filtered_topk`
10. 再统一其余三个 eval
11. 最后实现 `train_composer.py`

## Minimum Viable Version

如果先做最小版本，可限制为：

- intervention 类
- loader
- `eval_truth`
- 4 个 method：
  - `single + output`
  - `equal + output`
  - `residual_softmax + output`
  - `topk_residual + output`

但内部类结构仍需按“后续支持 trainable arbitration”设计，避免返工。

## Recommended First Experiment Matrix

第一批最值得跑的矩阵：

- `single + output`
- `equal + output`
- `residual_softmax + output`
- `topk_residual + output`
- `equal + shared_latent + orth_mean + identity`
- `residual_softmax + shared_latent + orth_mean + identity`
- `residual_softmax + shared_latent + svd_union + identity`
- `residual_softmax + shared_latent + svd_union + overlap`
- `residual_softmax + projected_output + svd_union`

这组实验足以回答：

- 原空间组合是不是主要瓶颈
- shared latent 是否有信号
- 问题主要出在 basis，还是 transport
