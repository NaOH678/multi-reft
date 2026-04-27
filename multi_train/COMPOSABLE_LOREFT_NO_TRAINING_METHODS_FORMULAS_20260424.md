# Composable LoReFT No-Training 融合方法公式整理

日期：2026-04-24

本文档只整理我们已经尝试过的 `no-training` 融合方法，不讨论后续 router training。目标是把当前代码和实验里出现过的方法，统一写成一套符号和公式，便于后续讨论。

## 1. 统一记号

设：

- 一共有 `N` 个 specialist，索引为 `i = 1, ..., N`
- 当前层、当前位置的 base hidden state 为 `h \in R^d`
- 第 `i` 个 specialist 的 LoReFT 低秩维度为 `r`
- 第 `i` 个 specialist 的 rotation 矩阵为 `R_i \in R^{r \times d}`
- 第 `i` 个 specialist 的 source 参数记为 `W_i, b_i`
- 非线性记为 `f(\cdot)`，在代码里是 `act_fn`

对单个 specialist，代码中的中间量可以写成：

```text
z_i = R_i h
s_i = f(W_i h + b_i)
delta_i = s_i - z_i                  \in R^r
Delta_i = R_i^T delta_i             \in R^d
```

其中：

- `delta_i` 是低维 latent 空间中的改变量
- `Delta_i` 是 lift 回模型输出空间后的干预向量

代码里常用的两个打分量是：

```text
r_i^delta = ||delta_i||_2
r_i^int   = ||Delta_i||_2
```

两两兼容性用 lift 后向量的 cosine：

```text
c_{ij} = cos(Delta_i, Delta_j)
```

最终都会先得到一个权重 `alpha_i`，再做组合。若在 `output` 域组合，则：

```text
Delta_mix = sum_i alpha_i Delta_i
h' = h + Delta_mix
```

所以真正的区别主要在两部分：

1. `alpha` 怎么算
2. 在哪个 `compose_domain` 里做组合

## 2. Compose Domain 公式

### 2.1 `output`

这是当前主线，也是效果最稳定的分支。

```text
Delta_mix = sum_i alpha_i Delta_i
h' = h + Delta_mix
```

### 2.2 `projected_output`

先在 output 空间混合，再投影到共享子空间。

设共享 basis 为 `U = {u_1, ..., u_k}`，则：

```text
Delta_raw = sum_i alpha_i Delta_i
Delta_proj = sum_m <Delta_raw, u_m> u_m
h' = h + Delta_proj
```

若把 basis 行堆叠成矩阵 `B \in R^{k \times d}`，也可写成：

```text
Delta_proj = B^T B Delta_raw
```

### 2.3 `shared_latent`

先把各 specialist 的低维改变量 transport 到共享 latent，再 lift 回输出空间。

设第 `i` 个 specialist 的 transport 为 `T_i`，共享 basis 为 `B \in R^{k \times d}`：

```text
tilde_delta_i = T_i delta_i
delta_mix = sum_i alpha_i tilde_delta_i
Delta_mix = B^T delta_mix
h' = h + Delta_mix
```

若 `transport_type = identity`，则 `T_i` 是恒等映射，此时要求共享维度与原低秩维度一致。

## 3. 基础 Policy 公式

以下 policy 都是在单层、单 token 位置上独立计算，不做跨 token pooling。

### 3.1 `single`

只使用某一个 specialist。

```text
alpha_i = 1(i = i*)
```

### 3.2 `equal`

所有 specialist 等权。

```text
alpha_i = 1 / N
```

### 3.3 `residual_softmax`

用低维 residual norm 做 softmax。

```text
score_i = r_i^delta
alpha_i = softmax(score_i / tau)
```

这里 `tau` 是 `composition_temperature`。

### 3.4 `intervention_softmax`

用某种 score source 做 softmax。当前支持两种 raw score：

```text
score_i = r_i^delta
```

或

```text
score_i = r_i^int
```

于是统一写成：

```text
score_i = q_i,   q_i in {r_i^delta, r_i^int}
alpha_i = softmax(score_i / tau)
```

其中：

- `score_source = delta_norm` 时，`q_i = r_i^delta`
- `score_source = intervention_norm` 时，`q_i = r_i^int`

### 3.5 `topk_residual`

先按 `r_i^delta` 取 top-k，再只在 top-k 内做 softmax。

设：

```text
K = TopK({r_i^delta}, k)
```

则：

```text
alpha_i =
  exp(r_i^delta / tau) / sum_{j in K} exp(r_j^delta / tau),   i in K
  0,                                                          i not in K
```

### 3.6 `compat_filtered_topk`

这是当前 follow-up 里最重要的方法之一。

第一步，选 raw score：

```text
q_i in {r_i^delta, r_i^int}
```

第二步，若需要 normalization，得到 effective score `tilde_q_i`；否则 `tilde_q_i = q_i`。

第三步，先按 `tilde_q_i` 取 top-k：

```text
K = TopK({tilde_q_i}, k)
```

第四步，按分数从高到低做 greedy compatibility filtering。初始化 `S = []`，对 `K` 中候选 `i` 依次检查：

```text
若对所有 j in S 都满足 c_{ij} >= gamma，则保留 i
否则丢弃 i
```

其中 `gamma = compat_threshold`。

若最后 `S` 为空，则回退到 top-1。

最后只在保留下来的集合 `S` 上做 masked softmax：

```text
alpha_i =
  exp(tilde_q_i / tau) / sum_{j in S} exp(tilde_q_j / tau),   i in S
  0,                                                          i not in S
```

## 4. Score Normalization 公式

这些 normalization 本身不是独立 policy，而是加在某些 policy 的 score 上。

### 4.1 `none`

```text
tilde_q_i = q_i
```

### 4.2 `mean_ratio`

设该层该 specialist 在 calibration 数据上的均值为 `mu_i`，则：

```text
tilde_q_i = q_i / max(mu_i, eps)
```

对应代码中的：

- `residual_scaled_softmax`
- 或 `intervention_softmax / compat_filtered_topk` 配合 `score_normalizer=mean_ratio`

### 4.3 `log_zscore`

设该层该 specialist 的 calibration 统计为：

- `m_i = E[log(q_i + eps)]`
- `s_i = Std[log(q_i + eps)]`

则：

```text
tilde_q_i = (log(q_i + eps) - m_i) / max(s_i, eps)
```

对应代码中的：

- `residual_logz_softmax`
- 或 `intervention_softmax / compat_filtered_topk` 配合 `score_normalizer=log_zscore`

## 5. Residual-Normalized 旧方法

这几个名字本质上是把 score source 固定为 `delta_norm`，再把 normalization 固定下来。

### 5.1 `residual_scaled_softmax`

```text
q_i = r_i^delta
tilde_q_i = q_i / max(mu_i, eps)
alpha_i = softmax(tilde_q_i / tau)
```

### 5.2 `residual_logz_softmax`

```text
q_i = r_i^delta
tilde_q_i = (log(q_i + eps) - m_i) / max(s_i, eps)
alpha_i = softmax(tilde_q_i / tau)
```

## 6. Follow-up 八组配置的公式对应

当前 follow-up 文档里的 `E1` 到 `E8`，其实只是下面几种组合：

| 配置 | 方法公式 |
| --- | --- |
| `E1 compat_delta_none` | `compat_filtered_topk` + `q_i=r_i^delta` + `none` |
| `E2 compat_delta_mean` | `compat_filtered_topk` + `q_i=r_i^delta` + `mean_ratio` |
| `E3 compat_delta_logz` | `compat_filtered_topk` + `q_i=r_i^delta` + `log_zscore` |
| `E4 intervention_none` | `intervention_softmax` + `q_i=r_i^int` + `none` |
| `E5 compat_intervention_mean` | `compat_filtered_topk` + `q_i=r_i^int` + `mean_ratio` |
| `E6 compat_intervention_logz` | `compat_filtered_topk` + `q_i=r_i^int` + `log_zscore` |
| `E7 intervention_mean` | `intervention_softmax` + `q_i=r_i^int` + `mean_ratio` |
| `E8 intervention_logz` | `intervention_softmax` + `q_i=r_i^int` + `log_zscore` |

## 7. `shared` / `specialist` stats 的含义

这不是新的融合公式，而是 normalization 所用 calibration 统计的来源不同。

### 7.1 `shared`

同一套全局统计文件中，按层、按 specialist 存下来的均值和方差：

```text
{mu_i, m_i, s_i} from shared calibration stats
```

### 7.2 `specialist`

仍然是按层、按 specialist 的统计，但来源是更细粒度、按 specialist 构建的 calibration 文件：

```text
{mu_i, m_i, s_i} from specialist-specific calibration stats
```

因此：

- `shared residual_scaled(temp=1)`
- `special residual_scaled(temp=1)`
- `shared residual_logz(temp=1)`
- `special residual_logz(temp=1)`

它们的 policy 公式完全一样，差别只在用哪套统计量。

## 8. 当前可以怎样理解这些方法

从公式角度看，当前 no-training 方法大致分三类：

1. 纯幅值路由
   - `residual_softmax`
   - `topk_residual`

2. 幅值标准化后再路由
   - `residual_scaled_softmax`
   - `residual_logz_softmax`
   - normalized `intervention_softmax`
   - normalized `compat_filtered_topk`

3. 路由时加入结构约束
   - `topk_residual`
   - `compat_filtered_topk`

其中最关键的两个设计轴是：

1. 用什么 score
   - `||delta_i||`
   - `||Delta_i||`

2. 怎么把分数转成 `alpha`
   - 直接 softmax
   - top-k 后 softmax
   - top-k + compatibility filtering 后 softmax

## 9. 一句总结

我们目前尝试过的 no-training 融合，本质上都可以写成：

```text
1. 先对每个 specialist 计算 delta_i, Delta_i
2. 从 {||delta_i||, ||Delta_i||} 中选一个 raw score
3. 可选做 normalization
4. 用某个 routing policy 得到 alpha_i
5. 在某个 compose_domain 中把各 specialist 干预混合起来
6. 得到 h' = h + mixed_intervention
```

后续若讨论 router training，本质上就是把第 3 步和第 4 步的一部分，从手工 heuristic 改成可学习映射。
