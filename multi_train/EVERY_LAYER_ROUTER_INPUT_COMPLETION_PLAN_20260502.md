# Every-Layer Router Input Completion Plan

## 1. 问题背景

当前 every-layer router 的主线已经比较明确：

$$
\mathcal L_R = \mathcal L_{\text{LoReFT}}
$$

也就是：所有原本安装 LoReFT intervention 的层，都安装 trainable router。

因此，当前的核心问题已经不再是“non-router 层如何 fallback”，而是：

$$
\boxed{\text{router 在每一层、每一个 intervention token 上，到底看什么输入来决定 } \alpha_{i,t}^l}
$$

从目前实现和实验结果来看，只使用 intervention 几何量相关特征，例如：

- `rh_direction`
- `delta_direction`
- `rh_log_norm`
- `delta_log_norm`
- `mean_compat`
- `neg_compat_mass`

容易使 router 学成一种接近“固定层分工”的策略，而不是：

$$
\boxed{\text{基于当前样本、当前 token、当前 forward 状态的 conditional routing}}
$$

因此，下一版 router 的重点不是改 scorer 形式，而是：

$$
\boxed{\text{补齐 router input}}
$$

---

## 2. 总体结论

router input 的补齐分两步进行：

1. 先加入 **token-level hidden state feature**
2. 再视情况加入 **prompt-level global feature**

其中第一步优先级最高。

更具体地说，下一版 router input 建议写成：

$$
x_{i,t}^l
=
[
\text{local}_{i,t}^l;
\text{compat}_{i,t}^l;
g_t^l;
c
]
$$

其中：

- $\text{local}_{i,t}^l$：第 $i$ 个 specialist 在第 $l$ 层第 $t$ 个 token 上的局部 intervention 特征
- $\text{compat}_{i,t}^l$：第 $i$ 个 specialist 与其他 specialists 的兼容性特征
- $g_t^l$：当前层、当前 token、当前 composable forward 状态的 hidden feature
- $c$：整条 prompt 的全局语义 feature

下一版的核心思想是：

$$
\boxed{\text{保留现有 rh/delta/compat 特征，但让它们从“全部输入”降级为“局部几何提示”}}
$$

同时补上真正决定 routing 的状态量：

$$
\boxed{h_{\mathrm{pre},t}^l \text{ 和 } g_{\text{global}}}
$$

---

## 3. Local Feature

第 $i$ 个 specialist 在第 $l$ 层第 $t$ 个 token 上的 LoReFT 相关量定义为：

$$
r_{i,t}^l = R_i^l h_{\mathrm{pre},t}^l
$$

$$
\delta_{i,t}^l
=
W_i^l h_{\mathrm{pre},t}^l + b_i^l - R_i^l h_{\mathrm{pre},t}^l
$$

$$
\Delta h_{i,t}^l = (R_i^l)^\top \delta_{i,t}^l
$$

其中 $h_{\mathrm{pre},t}^l$ 表示当前第 $l$ 层 intervention 之前的 hidden state。

local feature 继续采用“方向 + 强度分离”的方式。

### 3.1 Directional Features

$$
u_{r,i,t}^l
=
\frac{r_{i,t}^l}{\|r_{i,t}^l\|_2+\epsilon}
$$

$$
u_{\delta,i,t}^l
=
\frac{\delta_{i,t}^l}{\|\delta_{i,t}^l\|_2+\epsilon}
$$

对应实现中的：

- `rh_direction`
- `delta_direction`

### 3.2 Scalar Features

$$
s_{r,i,t}^l
=
\log\left(\|r_{i,t}^l\|_2+\epsilon\right)
$$

$$
s_{\delta,i,t}^l
=
\log\left(\|\delta_{i,t}^l\|_2+\epsilon\right)
$$

对应实现中的：

- `rh_log_norm`
- `delta_log_norm`

### 3.3 Local Projection

将 directional local features 拼接后，做 specialist-specific projection：

$$
v_{i,t}^l
=
\left[
u_{r,i,t}^l;
u_{\delta,i,t}^l
\right]
$$

$$
a_{i,t}^l = P_{R,l,i}(v_{i,t}^l)
$$

其中：

$$
P_{R,l,i} : \mathbb R^{2r} \rightarrow \mathbb R^{d_r}
$$

$r$ 为 LoReFT low-rank dimension，$d_r$ 为 local projection dimension。

推荐主配置：

$$
d_r = 128
$$

---

## 4. Compatibility Feature

compatibility 继续保留，因为它能够表达多个 specialists 是否相互冲突。

令：

$$
\hat{\Delta h}_{i,t}^l
=
\frac{\Delta h_{i,t}^l}{\|\Delta h_{i,t}^l\|_2+\epsilon}
$$

则第 $i,j$ 个 specialists 的 pairwise compatibility 定义为：

$$
\text{cos}_{i,j,t}^l
=
\left\langle
\hat{\Delta h}_{i,t}^l,
\hat{\Delta h}_{j,t}^l
\right\rangle
$$

基于此构造两个标量：

$$
\text{mean\_compat}_{i,t}^l
=
\frac{1}{|\mathcal S|-1}
\sum_{j\neq i}\text{cos}_{i,j,t}^l
$$

$$
\text{neg\_compat\_mass}_{i,t}^l
=
\sum_{j\neq i}\max\left(0,-\text{cos}_{i,j,t}^l\right)
$$

对应实现中的：

- `mean_compat`
- `neg_compat_mass`

这两个量不应再被视为主决策信号，而应视为：

$$
\boxed{\text{对 local intervention 几何关系的辅助描述}}
$$

---

## 5. Hidden State Feature

这是下一版最重要的新增输入。

### 5.1 为什么必须加入

当前 router 若只看 `rh/delta/compat`，本质上主要是在看：

$$
\boxed{\text{各 specialist 在当前层上“可能施加多大、多朝哪个方向”的干预}}
$$

但它仍然缺少：

$$
\boxed{\text{当前样本、当前 token、当前 forward 状态到底是什么}}
$$

因此同一层很容易学成某种近似固定分工，而不是真正按状态 routing。

### 5.2 定义

hidden state feature 应当来自：

$$
h_{\mathrm{pre},t}^l
$$

即：

$$
\boxed{\text{当前 composable forward 中，第 } l \text{ 层 intervention 之前、第 } t \text{ 个 token 的 hidden state}}
$$

不能使用：

$$
h_{\mathrm{post},t}^l
=
h_{\mathrm{pre},t}^l
+
\sum_i \alpha_{i,t}^l \Delta h_{i,t}^l
$$

否则会产生循环依赖：

$$
\alpha^l
\rightarrow
h_{\mathrm{post}}^l
\rightarrow
\alpha^l
$$

### 5.3 Hidden Projection

建议先做 LayerNorm，再做 layer-specific projection：

$$
g_t^l
=
P_h^l\left(\operatorname{LN}(h_{\mathrm{pre},t}^l)\right)
$$

其中：

$$
P_h^l : \mathbb R^{d_{\text{model}}} \rightarrow \mathbb R^{d_h}
$$

并且 $P_h^l$ 在同一层内对所有 specialists 共享。

推荐主配置：

$$
d_h = 64
$$

加入这个分支后，router 才真正具备：

$$
\boxed{\text{同一层、同一 specialist，在不同样本/不同 token 上做不同决策}}
$$

的能力。

因此：

$$
\boxed{h_{\mathrm{pre}} \text{ 是下一版 router input 中优先级最高的新增特征}}
$$

---

## 6. Global Feature

global feature 的作用不是表达当前 token 的局部状态，而是表达：

$$
\boxed{\text{整条 prompt 的任务语义}}
$$

例如：

- 当前更像 truthful question
- 当前更像 ethics instruction
- 当前更像 toxicity-sensitive prompt

### 6.1 为什么不能直接从主 forward 里取

在 every-layer router 设定下，主 forward 的中间 hidden state 已经被前面层 router 和 LoReFT 改写：

$$
h_{\text{current}}^l \neq h_{\text{base-only}}^l
$$

因此 global feature 不应直接从主 composable forward 的任意中间 hidden state 中抽取。

更合理的做法是从 frozen base-only side branch 中抽取，或者离线预计算。

### 6.2 推荐定义

推荐从 frozen base-only 模型某个中层，取 prompt 最后一个 token 的 hidden state：

$$
g_{\text{global}}
=
h_{\text{base-only},T_p}^{l_g}
$$

其中：

- $T_p$：prompt 最后一个 token
- $l_g$：固定选择的一层，例如 16 或 20

然后做：

$$
c
=
P_c\left(\operatorname{LN}(g_{\text{global}})\right)
$$

其中：

$$
P_c : \mathbb R^{d_{\text{model}}} \rightarrow \mathbb R^{d_c}
$$

推荐主配置：

$$
d_c = 64
$$

这个向量对整条样本固定，然后广播到：

- 所有 router 层
- 所有 intervention tokens
- 所有 specialists

### 6.3 优先级

global feature 有价值，但优先级低于 hidden state feature。

原因是当前更缺的是：

$$
\boxed{\text{当前层、当前 token 的局部状态}}
$$

而不是整条 prompt 的粗粒度语义摘要。

因此建议：

1. 先实现 $h_{\mathrm{pre}}$
2. 再视实验结果决定是否加入 $c$

---

## 7. 最终 Router Input 形式

### 7.1 第一阶段推荐输入

第一阶段先加入 $h_{\mathrm{pre}}$，不加入 global feature：

$$
x_{i,t}^l
=
\left[
a_{i,t}^l;
s_{r,i,t}^l;
s_{\delta,i,t}^l;
\text{mean\_compat}_{i,t}^l;
\text{neg\_compat\_mass}_{i,t}^l;
g_t^l
\right]
$$

其中：

- $a_{i,t}^l \in \mathbb R^{d_r}$
- $g_t^l \in \mathbb R^{d_h}$

若取：

$$
d_r=128,\quad d_h=64
$$

则每个 specialist 的输入维度约为：

$$
128 + 4 + 64 = 196
$$

### 7.2 第二阶段推荐输入

如果第一阶段仍然出现明显平台期，或者 debug summary 显示后层 routing 仍然高度模板化，则加入 global feature：

$$
x_{i,t}^l
=
\left[
a_{i,t}^l;
s_{r,i,t}^l;
s_{\delta,i,t}^l;
\text{mean\_compat}_{i,t}^l;
\text{neg\_compat\_mass}_{i,t}^l;
g_t^l;
c
\right]
$$

若取：

$$
d_r=128,\quad d_h=64,\quad d_c=64
$$

则每个 specialist 的输入维度约为：

$$
128 + 4 + 64 + 64 = 260
$$

---

## 8. Scorer 结构

不建议改成 joint MLP，即不建议直接对：

$$
[x_1;x_2;x_3;x_4]
$$

做一次联合打分。

更推荐继续保持当前主结构：

$$
\boxed{\text{specialist-specific projection} + \text{layer-specific shared scorer}}
$$

也就是：

$$
q_{i,t}^l = s_l(x_{i,t}^l)
$$

$$
\alpha_{i,t}^l
=
\operatorname{softmax}_i\left(q_{i,t}^l\right)
$$

其中：

- $P_{R,l,i}$：layer-specific, specialist-specific
- $P_h^l$：layer-specific, shared across specialists
- $P_c$：shared across all layers
- $s_l$：layer-specific, shared across specialists

推荐继续使用两层 MLP scorer：

$$
s_l : \mathbb R^{d_{\text{in}}} \rightarrow \mathbb R
$$

隐藏层维度推荐：

$$
d_{\text{hidden}} = 128
$$

---

## 9. 归一化建议

### 9.1 Directional Features

以下向量继续做单位化：

- `rh_direction`
- `delta_direction`

即：

$$
u = \frac{v}{\|v\|_2+\epsilon}
$$

### 9.2 Scalar Features

以下标量继续做 z-score 或现有统计量归一化，并做 clip：

- `rh_log_norm`
- `delta_log_norm`
- `mean_compat`
- `neg_compat_mass`

### 9.3 Hidden Feature

对 $h_{\mathrm{pre},t}^l$ 不建议额外手工做 L2 normalize。更推荐：

$$
\operatorname{LN}(h_{\mathrm{pre},t}^l)
\rightarrow
P_h^l
$$

### 9.4 Global Feature

对 $g_{\text{global}}$ 同样建议：

$$
\operatorname{LN}(g_{\text{global}})
\rightarrow
P_c
$$

---

## 10. 推荐实验顺序

### Phase A

保留当前 local + compat 特征，新增：

$$
h_{\mathrm{pre}}
$$

即：

$$
x_{i,t}^l
=
[\text{local};\text{compat};g_t^l]
$$

这是下一版的最关键实验。

### Phase B

如果 Phase A 之后：

- train loss 仍然很快平台
- debug summary 里后层 alpha 仍然接近固定模板
- benchmark 几个 ckpt 之间 routing 差异仍然很小

则再加入：

$$
c
$$

即：

$$
x_{i,t}^l
=
[\text{local};\text{compat};g_t^l;c]
$$

### Phase C

如果加入 $h_{\mathrm{pre}}$ 和 $c$ 后，仍然没有明显改善，再考虑：

- null expert / off expert
- 更细的 layer-wise 参数共享设计
- 更强的 auxiliary monitor / regularization

但这些都应放在 input 补齐之后。

---

## 11. KL Regularization

这里按照 `multi_train/every_layer_router_design_dollar_math.md` 中的最小可归因版本，只引入 task-level soft routing supervision，不额外引入复杂 teacher。

当前只验证一个问题：

$$
\boxed{\text{显式 task-level routing supervision 是否帮助正确 routing？}}
$$

baseline：

$$
\mathcal L = \mathcal L_{\text{answer}}
$$

加 KL 后：

$$
\mathcal L
=
\mathcal L_{\text{answer}}
+
\lambda_{\text{KL}}
\mathcal L_{\text{KL}}
$$

### 11.1 KL 定义

设 trainable router 在第 $l$ 层、第 $t$ 个 intervention token 上输出：

$$
\alpha_t^l = [\alpha_{1,t}^l,\ldots,\alpha_{N,t}^l]
$$

则：

$$
\mathcal L_{\text{KL}}
=
\frac{1}{|\mathcal L_R||\mathcal I(x)|}
\sum_{l\in\mathcal L_R}
\sum_{t\in\mathcal I(x)}
\operatorname{KL}
\left(
p_y
\|\
\alpha_t^l
\right)
$$

其中：

- $N$ 为 specialist 数量
- $y$ 为当前样本对应的任务标签
- $p_y$ 为该任务对应的 soft target distribution

### 11.2 Soft Target

teacher distribution 使用 soft target：

$$
p_y(i)
=
\begin{cases}
0.8, & i=y \\
\frac{0.2}{N-1}, & i\neq y
\end{cases}
$$

如果 $N=4$，则非目标 specialist 权重为：

$$
\frac{0.2}{3}\approx 0.0667
$$

这里的关键点是：

$$
\boxed{\text{使用 soft target，而不是 hard one-hot target}}
$$

这样 KL 提供的是任务级方向性约束，但不会把 router 直接退化成硬分类器。

### 11.3 KL 方向

KL 方向使用：

$$
\operatorname{KL}(p_y\|\alpha)
$$

实现上等价于 soft-label cross entropy：

$$
\mathcal L_{\text{KL}}
=
-
\sum_i
p_y(i)\log \alpha_i
+
\text{const}
$$

### 11.4 推荐实验设置

第一轮只试：

$$
\lambda_{\text{KL}}\in\{0,0.02,0.05\}
$$

不额外加入：

- entropy
- margin
- front/back weighting
- layer weighting

避免不可归因。

### 11.5 解释

这一版 KL 的作用只是给 every-layer router 一个最小的 task-level prior：

$$
\boxed{
\text{让 router 知道“这个任务整体更应该偏向哪个 specialist”，但不直接规定细粒度 token-level routing。}
}
$$

因此它适合和当前的 input 补齐实验一起使用：

- `Exp-1`: every-layer router，no KL
- `Exp-2`: every-layer router + weak KL
- `Exp-3`: every-layer router + medium KL

---

## 12. 最终结论

下一版 every-layer router 的核心，不是推翻现有 `rh/delta/compat` 结构，而是：

$$
\boxed{
\text{让 router 在现有 intervention 几何特征之外，再显式看到当前 forward 状态}
}
$$

因此，推荐的优先级是：

1. 必加：

$$
h_{\mathrm{pre},t}^l
$$

2. 第二优先级：

$$
g_{\text{global}}
$$

3. 保留作为辅助提示：

- `rh_direction`
- `delta_direction`
- `rh_log_norm`
- `delta_log_norm`
- `mean_compat`
- `neg_compat_mass`

一句话总结：

$$
\boxed{
\text{现有 local/compat 特征负责描述“specialist 能做什么”，}
}
$$

$$
\boxed{
\text{而新增 } h_{\mathrm{pre}} \text{ 和 } g_{\text{global}} \text{ 负责描述“当前到底需要什么”。}
}
$$

另外，在训练目标中建议加入适度的 KL regularization：

$$
\mathcal L
=
\mathcal L_{\text{answer}}
+
\lambda_{\mathrm{KL}} \mathcal L_{\mathrm{KL}}
$$

使 router 在训练早期不要过快塌缩到单一模板化分工。
