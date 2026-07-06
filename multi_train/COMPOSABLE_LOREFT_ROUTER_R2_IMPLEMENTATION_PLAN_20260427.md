# Composable LoReFT Router R2 实现方案

日期：2026-04-27

本文档整理当前已经收束的第二阶段 router 实现方案。这里的第二阶段只指：

```text
冻结 base model 和所有 specialists，只训练 router。
```

本文档只讨论实现方案，不直接修改代码。

## 1. 目标与范围

当前主要问题不是 specialist 本身完全无效，而是：

- `toxicity / ethics` 上，`truthful` specialist 在中后层尤其最后几层重新拿回收口权
- `toxicity` specialist 经常参与，但不能稳定主导
- 输出会被拉成 truth-style QA 模板，例如：
  - `the correct answer is true`
  - `the correct answer is false`
  - `the correct answer is wrong / not wrong`

因此本次实现的目标是：

1. 落地 `R2 router`
2. 保留非 router 层的 `E6` baseline
3. 优化当前 compatibility 相关实现的性能
4. 保留与当前低效实现的逐项正确性对比，防止优化引入计算错误
5. 在架构上兼容未来的 `R3`，避免返工

## 2. 阶段定义

### 2.1 阶段 1：已经完成

已经分别训练好的 specialist：

- `truthful`
- `moral / ethics`
- `stereotype / bias`
- `toxicity`

这些 specialist 在第二阶段全部冻结。

### 2.2 阶段 2：当前要做的 router 训练

第二阶段只训练：

```text
router / policy head
```

冻结：

- base model
- 所有 LoReFT specialists

即：

```text
theta_base      frozen
theta_specialist frozen
theta_router     trainable
```

## 3. Router 安装层

主版本只在后 8 层安装 router：

```text
L_R = {24, 25, 26, 27, 28, 29, 30, 31}
```

非 router 层继续使用当前最强 no-training baseline：

```text
E6 = compat_filtered_topk + intervention_norm + log_zscore
```

因此：

- `l in L_R`：使用 trainable router
- `l not in L_R`：继续使用 `E6`

后续做一个后 4 层 ablation：

```text
L_R_abl = {28, 29, 30, 31}
```

用于验证问题是否集中在最后收口层。

## 4. Router 结构路线

本次实现以 `R2` 为主线，但接口上兼容未来 `R3`。

### 4.1 R1：stats-only

输入 4 个统计量：

- `delta_norm`
- `intervention_norm`
- `mean_compat`
- `neg_compat_mass`

MLP：

```text
4 -> 32 -> 1
```

### 4.2 R2：stats + subspace_energy

这是本次主方案。

输入 5 个统计量：

- `delta_norm`
- `intervention_norm`
- `subspace_energy`
- `mean_compat`
- `neg_compat_mass`

MLP：

```text
5 -> 32 -> 1
```

这里的 `subspace_energy` 对应当前代码中的：

```text
||R_i h_t||_2^2
```

实现时建议直接沿用当前代码里的平方范数定义，不另外改成开方版本，避免定义分叉。

### 4.3 R3：R2 + hidden state projection

未来如果 `R2` 仍然不足，再加：

```text
hidden_proj = P_h h_t
```

推荐：

```text
P_h: d -> 8 或 16
```

此时 router 输入变成：

- `hidden_proj`
- `delta_norm`
- `intervention_norm`
- `subspace_energy`
- `mean_compat`
- `neg_compat_mass`

因此这次实现不能把 router 输入维度写死成 5。

## 5. 统一的 LoReFT 表达

对第 `l` 层、第 `t` 个 token、第 `i` 个 specialist：

- base hidden state：`h_t^l`
- 低维残差：

```text
delta_{i,t}^l = f(W_i^l h_t^l + b_i^l) - R_i^l h_t^l
```

- lift 回 hidden space 后：

```text
Delta h_{i,t}^l = (R_i^l)^T delta_{i,t}^l
```

融合后的 hidden state：

```text
h_t^l + sum_i alpha_{i,t}^l Delta h_{i,t}^l
```

第二阶段 router 只替换：

```text
alpha_{i,t}^l
```

不改：

- `delta_{i,t}^l`
- `Delta h_{i,t}^l`
- specialists 参数

## 6. R2 的输入特征

R2 router 输入固定为 5 个统计量：

```text
x_{i,t}^l =
[
  ||delta_{i,t}^l||_2,
  ||Delta h_{i,t}^l||_2,
  ||R_i^l h_t^l||_2^2,
  mean_compat_{i,t}^l,
  neg_compat_{i,t}^l
]
```

其中 compatibility 统计量定义为：

```text
pairwise_cos_{ij} = cos(Delta h_i, Delta h_j)
```

```text
mean_compat_i = average_j!=i pairwise_cos_{ij}
```

```text
neg_compat_i = sum_j!=i max(0, -pairwise_cos_{ij})
```

## 7. 特征归一化

R2 第一版使用：

- raw feature
- offline mean/std
- z-score
- clip

具体为：

```text
phi_{i,t,l,k}
= clip(
    (x_{i,t,l,k} - mu_{i,l,k}) / (sigma_{i,l,k} + eps),
    -5, 5
  )
```

注意：

- 第一版不要混用 raw 和 log
- 如果离线统计的是 raw 值，那么训练和推理都使用 raw z-score
- 未来若做 log-zscore ablation，必须重新构建 log-space stats

## 8. Router 输出

R2 router 使用 full softmax，不保留 hard compatibility filter。

定义：

```text
q_{i,t}^l = g_theta(phi_{i,t}^l)
```

```text
alpha_{i,t}^l = softmax_i(q_{i,t}^l / tau)
```

其中：

- `g_theta` 是 `5 -> 32 -> 1` MLP
- `tau` 初始取 `1.0`

router 层不再执行当前 `compat_filtered_topk` 的 greedy hard filtering。

原因：

- 现在已经观察到 `tox` 经常进入 top-k，但经常被 compatibility filter 刷掉
- 如果 router 层仍然保留 hard filter，那么 `tox` 即使被打高分，仍然可能被清零
- 这会直接削弱 router 修复后层 truth 收口的能力

因此：

- compatibility 仍然保留为 feature
- 但不再作为 router 层的 hard decision rule

## 9. 非 Router 层：继续使用 E6

非 router 层保留：

```text
compat_filtered_topk
+ intervention_norm
+ log_zscore
+ topk=2
+ compat_threshold=0.0
```

也就是继续使用当前 `E6`。

但要对实现路径做性能优化，不改变其数学行为。

## 10. 当前 compatibility 实现存在的性能问题

当前低效主要来自 [interventions.py](/mnt/petrelfs/shichaojian/multi-reft/pyreft/pyreft/interventions.py)：

1. `_extract_policy_features` 无条件计算：
   - `pairwise_cos`
   - `mean_compat`
   - `neg_compat_mass`

2. `compat_filtered_topk` 使用 Python 双重循环 + greedy filtering

3. `forward` 无条件执行 `_cache_debug_tensors`

4. `_cache_debug_tensors` 内大量执行：
   - `.detach().cpu()`

5. 训练路径与 debug 路径没有分离，debug 成本被强行带入主路径

因此本次实现必须把：

- 训练 / 推理主路径
- debug / 分析路径

明确拆开。

## 11. E6 compatibility 的优化原则

### 11.1 不先改数学定义

第一轮优化不改这些定义：

- `pairwise_cos`
- `mean_compat`
- `neg_compat_mass`
- `E6` 的 greedy filtering 规则

原因是：

- 先保证新旧实现语义等价
- 否则正确性对比会很复杂

### 11.2 先改实现方式

本次优化重点是：

- 按需计算 compatibility 相关特征
- 关闭训练/推理路径上的 debug cache
- router 层不再执行 hard filter
- 对非 router 层的 E6 保持行为一致，但优化计算路径

一句话说：

```text
保留 E6 的 compatibility 定义，优化实现和使用方式。
```

## 12. 新的实现模块设计

建议拆成 6 个模块。

### 模块 A：Router Feature Schema

不要把 router 输入写死成固定 5 维。

建议引入有序 feature name 列表：

```text
router_feature_names = [
  "delta_norm",
  "intervention_norm",
  "subspace_energy",
  "mean_compat",
  "neg_compat_mass",
]
```

未来 R3 只是在这个列表前面或中间加：

```text
"hidden_proj"
```

这样：

- feature 顺序固定
- stats 文件可对齐
- checkpoint metadata 可对齐

### 模块 B：Feature Bank

把特征提取拆成两层：

1. core states
   - `delta`
   - `Delta`
   - `rotated_base`

2. derived feature bank
   - `delta_norm`
   - `intervention_norm`
   - `subspace_energy`
   - `pairwise_cos`
   - `mean_compat`
   - `neg_compat_mass`

要求：

- 不同 policy 按需取 feature
- 不再默认每个 policy 都算全套 compatibility 特征

### 模块 C：Compatibility Backend

把 compatibility 相关计算独立成后端，统一输出：

- `pairwise_cos`
- `mean_compat`
- `neg_compat_mass`

这样：

- 非 router 层 E6 可继续用 `pairwise_cos`
- router 层只取 `mean_compat / neg_compat_mass`

### 模块 D：Mixed Per-Layer Policy Builder

builder 需要支持按层不同 policy：

- `24-31`：`trainable router`
- 其他层：`E6`

不再假设所有层使用统一 `policy_type`。

### 模块 E：Debug Gate

给 `_cache_debug_tensors` 加总开关：

```text
enable_debug_cache = False
```

训练和正常推理默认关闭，仅在专门 debug/分析时打开。

### 模块 F：Correctness Harness

新增独立的正确性对比脚本，不混进训练主流程。

职责：

- 旧实现 vs 新实现逐项数值对比
- E6 routing 决策对比
- logits / generation 端到端对比

## 13. Router Feature Stats

不要复用现有 residual/intervention stats 文件。

原因：

现有 stats 不包含：

- `subspace_energy`
- `mean_compat`
- `neg_compat_mass`

建议新建独立 stats 文件，例如：

```text
multi_train/calibration/router_features/stats/router_feature_stats_train_input.json
```

建议结构：

```text
{
  "feature_names": [...],
  "layers": {
    "24": {
      "specialist_label_1": {
        "delta_norm": {"mean": ..., "std": ...},
        "intervention_norm": {"mean": ..., "std": ...},
        "subspace_energy": {"mean": ..., "std": ...},
        "mean_compat": {"mean": ..., "std": ...},
        "neg_compat_mass": {"mean": ..., "std": ...}
      },
      ...
    },
    ...
  }
}
```

未来 R3 的 `hidden_proj` 不建议先并入这套 per-specialist stats 文件。

建议未来单独处理：

- projection 后直接 LayerNorm / RMSNorm
- 或单独统计 hidden projection 分布

但不要和当前 5 个 stats 混在同一个接口里。

## 14. Trainable Router Head 的实现要求

当前 `policy_head` 已经有雏形，但这次要改成“配置驱动”：

- 不把 `feature_dim=5` 写死
- 由 `router_feature_names` 推导输入维度
- router hidden dim 可配置

R2：

- `router_input_dim = 5`
- `router_hidden_dim = 32`

R3：

- `router_input_dim = 5 + hidden_proj_dim`
- `hidden_proj_dim = 8 或 16`

## 15. 训练入口

不要复用旧的 [router_train.py](/mnt/petrelfs/shichaojian/multi-reft/multi_train/router_train.py)。

建议新建训练入口：

```text
multi_train/train_composable_router.py
```

职责：

1. 加载 base model
2. 加载 specialists
3. 构建按层 mixed policy 的 composable model
4. 冻结 base / specialists
5. 只打开 router 参数
6. 混合多任务数据
7. 跑 answer-only generation loss
8. 保存 router checkpoint 与 metadata

## 16. 数据与采样

第二阶段复用 specialist 对应训练域数据，但用途变成 router 训练。

混合训练集：

- truthfulQA 相关训练域数据
- BBQ / bias 相关训练域数据
- ethics / moral 相关训练域数据
- toxicity 相关训练域数据

task label 只用于：

- 采样比例控制
- loss 统计
- evaluation 分组
- 未来可选的 route KL mask

不作为 router 输入。

推荐采样比例：

- `toxicity`: `35% - 40%`
- `ethics`: `25%`
- `truthfulQA`: `17.5% - 20%`
- `BBQ`: `17.5% - 20%`

含义：

- `toxicity / ethics` 是 repair tasks
- `truthfulQA / BBQ` 是 anchor tasks

## 17. Loss 设计

第一版训练只使用：

```text
answer-only generation loss
```

也就是：

- prompt token 不算 loss
- answer span 算 causal LM loss

第一版不加 route KL。

但训练入口建议预留接口：

- `router_kl_weight`
- `router_kl_tasks`
- `router_teacher_policy`

以后若 `truthfulQA / BBQ` 明显掉分，再加：

- 只在 `truthfulQA / BBQ` 上启用的 route KL

第一版保持最干净：

```text
L = L_answer
```

## 17.1 训练期监控信号

训练时需要记录一小组关键监控信号，但不要把当前完整 `debug_summary` 原样搬进主训练路径。

原则：

- 训练主路径只记录轻量聚合信号
- 不记录每个 token 的完整 tensor
- 不在每 step 做 `.cpu()` 大量 debug cache
- 更详细的分析放到周期性验证阶段

### 训练主路径建议记录的最小集合

1. `loss_total`
2. `loss_by_task`
   - `loss_truthfulqa`
   - `loss_bbq`
   - `loss_ethics`
   - `loss_toxicity`
3. `alpha_mean[layer, specialist, task]`
   - 重点看 router 层 `24-31`
4. `dominant_rate[layer, specialist, task]`
   - 表示该 specialist 在该层成为最大权重的比例
5. `alpha_entropy[layer, task]`
   - 判断 router 是否过尖或过平
6. `truth_minus_tox_alpha_late` on toxicity
   - 定义为 toxicity 样本上，后层 `24-31` 中 `alpha_truth - alpha_tox` 的平均值
7. `truth_minus_moral_alpha_late` on ethics
   - 定义为 ethics 样本上，后层 `24-31` 中 `alpha_truth - alpha_moral` 的平均值

这组信号已经足够回答最关键的问题：

- 后层 `truthful` 是否仍然主导
- `toxicity / ethics` 上对应 specialist 是否拿回更多控制权
- `truthfulQA / BBQ` 是否因为 router 训练而明显退化

### 周期性验证阶段建议额外记录

每个验证周期额外记录一个轻量版 router debug json，至少包含：

1. `alpha_mean[layer, specialist, task]`
2. `dominant_rate[layer, specialist, task]`
3. `late_truth_alpha_mean`
4. `late_tox_alpha_mean`
5. `late_moral_alpha_mean`
6. `qa_template_rate_toxicity`
7. `qa_template_rate_ethics`

其中：

- `qa_template_rate_toxicity`：统计 toxicity 输出里出现 truth-style QA 模板的比例
- `qa_template_rate_ethics`：统计 ethics 输出里出现 truth-style QA 模板的比例

注意：

- 这些验证期信号应在固定小验证集上统计
- 不建议在每个 training step 上做生成式验证
- 不建议在训练主路径缓存完整 `pairwise_cos / alpha` tensor 到 CPU

因此本次实现需要在训练入口里预留两类日志：

1. step 级轻量聚合日志
2. eval 级小型 `router_debug_summary.json`

## 18. 正确性对比方案

性能优化必须和正确性对比一起做。

### 18.1 算子级对比

对同一批输入，旧实现与新实现逐项比较：

- `delta_norm`
- `intervention_norm`
- `subspace_energy`
- `pairwise_cos`
- `mean_compat`
- `neg_compat_mass`

记录：

- `max_abs_diff`
- `mean_abs_diff`

要求：

- shape 完全一致
- 数值在容差内一致

### 18.2 Policy 级对比

对非 router 层 E6，比较：

- `topk indices`
- `selected_mask`
- `rejected_conflict_mask`
- `alpha`

要求：

- 优化前后 E6 routing 决策一致

如果不一致，必须打印：

- layer
- token position
- specialist
- raw score
- pairwise compat
- 旧版与新版决策差异

### 18.3 端到端对比

在 `router=off` 的条件下，对同一模型、同一 batch、同一 seed，比较：

- mixed hidden update
- logits
- greedy generation 输出

要求：

- 仅做优化、不加 router 时，结果不变

这一步是最关键的回归基线。

## 19. Legacy / Optimized 双实现对照

建议短期保留一个开发期开关：

```text
compat_impl = legacy / optimized
```

作用：

- 同一输入下切换两套实现
- 用于 A/B 正确性对比
- 用于回归测试

这个开关不一定长期暴露给用户，但开发期非常有用。

## 20. 文件级实现建议

### [pyreft/pyreft/interventions.py](/mnt/petrelfs/shichaojian/multi-reft/pyreft/pyreft/interventions.py)

核心改动文件。

需要支持：

- 按需 feature extraction
- compatibility backend 分离
- debug cache 开关
- router feature schema
- trainable router input assembly
- legacy/optimized compatibility 实现对照入口

### [multi_train/eval_common/composable_loreft.py](/mnt/petrelfs/shichaojian/multi-reft/multi_train/eval_common/composable_loreft.py)

需要支持：

- per-layer mixed policy 构造
- router layers 与 non-router layers 混装
- router feature stats 路径
- debug 开关向下传递

### 新增 router feature stats builder

建议新文件，单独负责：

- 离线统计 R2 的 5 个特征
- 保存 mean/std
- 未来兼容 R3

### 新增 router train entry

建议新文件：

```text
multi_train/train_composable_router.py
```

### 新增 correctness check script

建议独立脚本，专门负责：

- feature 对比
- E6 routing 决策对比
- logits / generation 对比

## 21. 推荐开发顺序

### Step 1

先做 compatibility 优化框架，但不改数学行为。

目标：

- 新旧实现都能跑
- 能对比 `pairwise_cos / mean_compat / neg_compat / alpha / selected_mask`

### Step 2

在 `router=off` 条件下，证明优化后与旧版 E6 结果一致。

这是最重要的回归关。

### Step 3

接入 R2 router：

- `24-31` 层 trainable
- 其他层优化后的 E6

### Step 4

跑第一轮训练与评测。

## 22. 第一批实现范围

这次实现只做：

- `R2`
- 但架构预留 `R3`

### 包含

- router feature stats builder
- composable intervention 支持可配置 router feature list
- mixed per-layer policy builder
- 训练脚本 `train_composable_router.py`
- 训练时关闭 debug cache
- 保存 router checkpoint + metadata
- legacy/optimized correctness 对照工具

### 明确不做

- 这次不实现 `R3 hidden projection`
- 这次不加 route KL
- 这次不在 router 层保留 hard compatibility filter
- 这次不复用旧 `router_train.py`

## 23. 验收标准

本次实现完成后，至少满足：

1. `router=off` 时，优化版与旧版 E6 在非 router 层行为一致
2. `pairwise_cos / mean_compat / neg_compat_mass` 数值与旧实现一致
3. 训练路径不再无条件 `.cpu()` debug cache
4. router 层不再执行 hard compatibility filtering
5. `R2` 可以直接训练
6. 后续 `R3` 只需要增加 `hidden_proj`，不需要重写框架

## 24. 最终推荐主方案

如果只选一个先实现并训练，推荐：

```text
R2 = stats + subspace_energy router
```

具体为：

- router layers：`24-31`
- non-router layers：`E6`
- router input：
  - `delta_norm`
  - `intervention_norm`
  - `subspace_energy`
  - `mean_compat`
  - `neg_compat_mass`
- normalization：`raw z-score + clip[-5,5]`
- router MLP：`5 -> 32 -> 1`
- router alpha：`full softmax`
- loss：`answer-only generation loss`
- route KL：第一版关闭
- compatibility 优化：先保留定义，优化实现路径并与旧实现做逐项正确性对比

## 25. 一句话总结

这次实现应该做成：

```text
一个 stats-feature router 框架，先实例化成 R2，
同时把 E6 compatibility 后端做成高效且可验证的实现；
优化前后保留旧实现对照，逐项验证 feature、routing mask 和最终输出一致，
避免性能优化引入计算错误；
并在接口层面天然兼容未来 R3 的 hidden-state feature 扩展。
```

## 26. 实验方案编号（2026-04-28）

下面整理当前最新一版 router 实验方案，并给出统一编号，便于后续训练、记录和对比。

### 26.1 总体原则

- 除额外 router 设计外，其余训练逻辑继续严格对齐 `multi_train/train.py`
- base model 冻结
- 所有 specialists 冻结
- 只训练 router 相关参数
- 非 router 层继续使用当前 E6 逻辑

### 26.2 当前主方案编号

定义当前主方案为：

```text
R2.1 = Rh-only directional router
```

这是当前优先实现和优先训练的版本。

### 26.3 R2.1 结构定义

- router layers：`28-30`
- 不再使用 `L31`
- 第 1 版先只引入 `R_i h`
- 暂不在第 1 版中加入 `delta_i`
- 不直接使用 raw vector
- 对 `R_i h` 先做方向归一化，并单独保留强度项

router 输入建议为：

- `R_i h / ||R_i h||`
- `log ||R_i h||`
- `mean_compat`
- `neg_compat_mass`

暂时去掉旧的幅值统计量：

- `delta_norm`
- `intervention_norm`
- `subspace_energy`

### 26.4 R2.1 Router 头设计

建议采用：

- 不同 layer 间参数分开
- 同一 layer 内，不同 specialist 各自做输入投影
- 投影后接该层共享 scorer

形式上：

```text
u_i = normalize(R_i h)
z_i = W_{l,i} u_i + b_{l,i}
x_i = [z_i ; log||R_i h|| ; mean_compat ; neg_compat_mass]
logit_i = MLP_l(x_i)
alpha = softmax(logits / T)
```

推荐配置：

- `router_proj_dim = 16` 或 `32`
- `scorer_hidden_dim = 32`
- `router_temperature = 1.5`
- router dropout：`0.0`

### 26.5 R2.1 训练配置

当前建议训练配置：

- learning rate：`5e-4`
- warmup ratio：`0.05`
- router layers：`28 29 30`
- 其余训练参数继续与 `multi_train/train.py` 对齐

当前保守档数据量：

- toxicity：`15000`
- moral：`10000`
- stereotype：`3067`
- truthful：`3067`

### 26.6 第一轮实验编号

第一轮只建议跑以下 3 组，不要一次铺太大：

#### Exp-R2.1-A

```text
Exp-R2.1-A = Rh-only directional router
```

- 使用 `R2.1`
- 输入为 `normalize(R_i h) + log||R_i h|| + compat`
- 这是主实验

#### Exp-R2.1-B

```text
Exp-R2.1-B = Rh + delta directional router
```

- 在 `Exp-R2.1-A` 基础上加入：
  - `delta_i / ||delta_i||`
  - `log ||delta_i||`
- 用来验证 `delta_i` 是否提供额外有效信息

#### Exp-R2.1-C

```text
Exp-R2.1-C = Rh-only directional router + higher temperature
```

- 与 `Exp-R2.1-A` 相同
- 只把 `router_temperature` 调到 `2.0`
- 用来判断是否能减轻 router 过早离散化

### 26.7 评估重点

这一轮实验不要只看总 loss，要重点观察：

- train loss 是否在 `epoch 2-3` 后还能继续下降
- `loss_bbq` 是否继续下降
- `loss_toxicity` 是否继续下降
- `alpha_entropy/L28-L30` 是否不再过早塌缩
- `dominant_rate` 是否不再过早逼近 `1.0`
- 去掉 `L31` 后训练是否更稳定

### 26.8 当前执行顺序

建议顺序：

1. 先跑 `Exp-R2.1-A`
2. 若平台明显推迟，再决定是否需要 `delta_i`
3. 若 `Exp-R2.1-A` 仍然过早塌缩，再跑 `Exp-R2.1-C`
4. 若仍不足，再跑 `Exp-R2.1-B`

### 26.9 当前版本的一句话总结

```text
当前最新主方案不是继续堆 norm 统计量，而是把 router 从“幅值驱动”
升级到“方向 + 强度 + compat”驱动；第一优先版本为 R2.1 / Exp-R2.1-A。
```
