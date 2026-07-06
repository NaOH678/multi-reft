# Every-Layer Composable LoReFT Router 方案整理

## 1. 当前问题重述

当前目标不是简单地“防止 truthful specialist 干扰 toxicity”，而是学习 **正确 routing**。

更准确地说，每一层、每个被干预的 prompt token 上，router 都应该判断当前状态下哪个 specialist 更应该主导，或者应该如何组合多个 specialists。

原来的 single specialist 训练方式是：

$$
\text{每一层都有 LoReFT intervention}
$$

因此，如果 composable 阶段只在少数层安装 router，而其他层关闭 intervention：

$$
l \notin \mathcal L_R \Rightarrow \alpha^l = 0
$$

会损害 single specialist 原本的基本能力。

如果其他层继续使用 E6：

$$
l \notin \mathcal L_R \Rightarrow \alpha^l = \alpha_{\text{E6}}^l
$$

又可能出现错误 routing。

所以当前更合理的主线是：

$$
\boxed{\mathcal L_R = \mathcal L_{\text{LoReFT}}}
$$

也就是：**所有原本有 LoReFT 的层都安装 trainable router**。

这样就不再需要决定 non-router 层用 E6 还是 off，因为没有 non-router 层。

---

## 2. 基本设定

第二阶段训练中：

$$
\theta_{\text{base}} \text{ frozen}
$$

$$
\theta_{\text{specialist}} \text{ frozen}
$$

$$
\theta_{\text{router}} \text{ trainable}
$$

也就是说，只训练 router 相关参数，不训练 base model，也不训练已经得到的 LoReFT specialists。

假设有四个 specialists：

$$
\mathcal S = \{\text{truthful},\text{moral},\text{bias},\text{toxicity}\}
$$

每个 specialist 在每一层都有自己的 LoReFT 参数。

---

## 3. 只干预 prompt token，不干预 answer token

LoReFT 标准做法是只干预 prompt 的前 7 个 token 和后 7 个 token。

设 prompt length 为 $T_p$，则 intervention positions 为：

$$
\mathcal I(x)
=
\{1,\ldots,7\}
\cup
\{T_p-6,\ldots,T_p\}
$$

answer tokens 不做 intervention。

因此：

$$
t\in \mathcal I(x)
\Rightarrow
\tilde h_t^l
=
h_t^l
+
\sum_i
\alpha_{i,t}^l
\Delta h_{i,t}^l
$$

$$
t\notin \mathcal I(x)
\Rightarrow
\tilde h_t^l = h_t^l
$$

训练时 answer-only language modeling loss 仍然只在 answer tokens 上计算：

$$
\mathcal L_{\text{answer}}
=
-
\sum_{t\in \text{answer}}
\log p(x_t\mid x_{<t})
$$

所以要区分：

$$
\boxed{\text{intervention positions} = \text{prompt front-7/back-7}}
$$

$$
\boxed{\text{LM loss positions} = \text{answer tokens}}
$$

---

## 4. LoReFT intervention 表达

对第 $l$ 层、第 $t$ 个 prompt token、第 $i$ 个 specialist：

$$
r_{i,t}^l = R_i^l h_{\mathrm{pre},t}^l
$$

$$
\delta_{i,t}^l
=
W_i^l h_{\mathrm{pre},t}^l
+
b_i^l
-
R_i^l h_{\mathrm{pre},t}^l
$$

$$
\Delta h_{i,t}^l
=
(R_i^l)^\top
\delta_{i,t}^l
$$

其中 $h_{\mathrm{pre},t}^l$ 表示当前第 $l$ 层 intervention 之前的 hidden state。

router 的任务是输出：

$$
\alpha_{i,t}^l
$$

最终更新：

$$
\tilde h_t^l
=
h_{\mathrm{pre},t}^l
+
\sum_i
\alpha_{i,t}^l
\Delta h_{i,t}^l
$$

---

## 5. Router 输出粒度

router 不是对整条样本只输出一次权重，也不是对所有 token 输出权重。

它对每一个：

$$
l\in \mathcal L_R
$$

$$
t\in \mathcal I(x)
$$

$$
i\in \mathcal S
$$

计算一个 score：

$$
q_{i,t}^l
=
s_l(x_{i,t}^l)
$$

然后对同一个 layer、同一个 token 上的四个 specialist 分数做 softmax：

$$
\alpha_{i,t}^l
=
\operatorname{softmax}_i
\left(
\frac{q_{i,t}^l}{T}
\right)
$$

如果 batch size 是 $B$，有 $L$ 个 router 层，干预 $14$ 个 prompt token，specialist 数是 $4$，则：

$$
q \in \mathbb R^{B\times L\times 14\times 4}
$$

$$
\alpha \in \mathbb R^{B\times L\times 14\times 4}
$$

softmax 沿 specialist 维度做。

---

## 6. Router 结构：逐 specialist 打分，而不是 joint MLP

推荐结构是：

$$
\boxed{\text{每个 specialist 单独构造输入，单独输出一个 score}}
$$

也就是：

$$
q_i = s_l(x_i)
$$

然后：

$$
\alpha = \operatorname{softmax}([q_1,q_2,q_3,q_4]/T)
$$

不推荐主方案用：

$$
[q_1,q_2,q_3,q_4]
=
\operatorname{MLP}([x_1;x_2;x_3;x_4])
$$

原因是 joint MLP 过度依赖 specialist 顺序，不方便扩展，也更难解释。

最推荐的折中是：

$$
\boxed{\text{specialist-specific projection} + \text{layer-specific shared scorer}}
$$

即：

$$
a_{i,t}^l = P_{R,l,i}(\text{local vector}_{i,t}^l)
$$

$$
q_{i,t}^l = s_l(x_{i,t}^l)
$$

其中 $P_{R,l,i}$ 是每层、每个 specialist 独立的投影，$s_l$ 是同一层内四个 specialists 共享的 scorer。

---

## 7. Router 输入由四类特征组成

每个 specialist 的 router input 建议写成：

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

四类信息分别是：

1. specialist-local feature
2. compatibility feature
3. token-level hidden state feature
4. prompt-level global feature

其中：

$$
g_t^l
$$

是当前 layer、当前 token 的 hidden feature，对所有 specialists 相同。

$$
c
$$

是整条 prompt 的 global feature，对所有 layers、tokens、specialists 相同。

不同 specialist 之间不同的是：

$$
\text{local}_{i,t}^l
$$

和：

$$
\text{compat}_{i,t}^l
$$

---

## 8. Global feature

如果采用 every-layer router，global feature 不能从主 forward 的任意中间层 hidden state 中取。

原因是 every-layer router 下，主 forward 中的中间 hidden state 已经被前面层的 LoReFT/router 改过：

$$
h_{\text{current}}^l \neq h_{\text{base-only}}^l
$$

所以 global feature 应该来自一条独立的 base-only side branch，或者离线预计算。

推荐定义：

$$
g_{\text{global}}
=
\operatorname{stopgrad}
\left(
h_{\text{base-only},T_p}^{16}
\right)
$$

即：base-only 模型第 16 层 prompt 最后一个 token 的 hidden state。

然后：

$$
c
=
\operatorname{LayerNorm}
\left(
P_{\text{global}}g_{\text{global}}
\right)
$$

其中：

$$
P_{\text{global}}:
\mathbb R^{d_{\text{model}}}
\rightarrow
\mathbb R^{d_c}
$$

推荐主配置：

$$
d_c = 64
$$

global feature 的含义是：

$$
\boxed{\text{原始 prompt 的任务语义}}
$$

它应该被广播给所有 router 层、所有 intervention tokens、所有 specialists。

实现上推荐离线预计算 $g_{\text{global}}$，因为 base model frozen，prompt 固定，该向量对每个样本是固定的。

---

## 9. Hidden state feature

hidden state feature 表示：

$$
\boxed{\text{当前层、当前 token、当前 forward 状态}}
$$

它不是 clean feature，而是 current composable forward 中当前层 intervention 之前的 hidden state。

第 $l$ 层 router 应该使用：

$$
h_{\mathrm{pre},t}^l
$$

而不是：

$$
h_{\mathrm{post},t}^l
=
h_{\mathrm{pre},t}^l
+
\sum_i
\alpha_{i,t}^l
\Delta h_{i,t}^l
$$

否则会产生循环依赖：

$$
\alpha^l
\rightarrow
h_{\mathrm{post}}^l
\rightarrow
\alpha^l
$$

因此：

$$
\boxed{\text{hidden feature 使用当前层 intervention 前的 }h_{\mathrm{pre},t}^l}
$$

只在 prompt front-7/back-7 token 上计算：

$$
t\in\mathcal I(x)
$$

推荐投影：

$$
g_t^l
=
\operatorname{LayerNorm}
\left(
P_h^l
\operatorname{stopgrad}(h_{\mathrm{pre},t}^l)
\right)
$$

其中：

$$
P_h^l:
\mathbb R^{d_{\text{model}}}
\rightarrow
\mathbb R^{d_h}
$$

推荐主配置：

$$
d_h = 64
$$

$P_h^l$ 建议 layer-specific：

$$
P_h^0,P_h^1,\ldots,P_h^{L-1}
$$

因为不同层 hidden state 分布不同。

hidden feature 对所有 specialists 共享：

$$
g_{1,t}^l
=
g_{2,t}^l
=
g_{3,t}^l
=
g_{4,t}^l
=
g_t^l
$$

---

## 10. Local feature

local feature 表示第 $i$ 个 specialist 在当前 hidden state 下会产生什么 intervention。

首先：

$$
r_{i,t}^l
=
R_i^l h_{\mathrm{pre},t}^l
$$

推荐使用方向和强度分离：

$$
u_{i,t}^l
=
\frac{
r_{i,t}^l
}{
\|r_{i,t}^l\|_2+\epsilon
}
$$

$$
s_{i,t}^l
=
\log
\left(
\|r_{i,t}^l\|_2+
\epsilon
\right)
$$

方向向量需要投影，标量统计量不需要投影。

$$
a_{i,t}^l
=
P_{R,l,i}u_{i,t}^l
$$

其中：

$$
P_{R,l,i}:\mathbb R^r\rightarrow\mathbb R^{d_R}
$$

推荐：

$$
d_R = 32
$$

$P_{R,l,i}$ 建议同时 layer-specific 和 specialist-specific。

原因是不同 specialist 的 $R_i^l$ 子空间坐标系不一致，不能假设 $R_{\text{truth}}h$ 和 $R_{\text{tox}}h$ 的坐标语义相同。

因此 $P_{R,l,i}$ 的作用是：

$$
\boxed{\text{把每个 specialist 自己的 local coordinate 转到 router common coordinate}}
$$

第一版 local feature 推荐只用：

$$
\text{local}_{i,t}^l
=
[a_{i,t}^l;s_{i,t}^l]
$$

暂时不加 $\delta_i$，方便归因。

---

## 11. Delta feature：后续增强项

如果第一版效果不够，可以加入 $\delta_i$ 方向。

$$
\delta_{i,t}^l
=
W_i^l h_{\mathrm{pre},t}^l
+
b_i^l
-
R_i^l h_{\mathrm{pre},t}^l
$$

方向归一化：

$$
v_{i,t}^l
=
\frac{
\delta_{i,t}^l
}{
\|\delta_{i,t}^l\|_2+
\epsilon
}
$$

投影：

$$
b_{i,t}^l
=
P_{\delta,l,i}v_{i,t}^l
$$

强度：

$$
\log(\|\delta_{i,t}^l\|_2+
\epsilon)
$$

以及：

$$
\log(\|\Delta h_{i,t}^l\|_2+
\epsilon)
$$

但这不建议第一版就加。建议等 $R_i h$ 版本跑完后再做 ablation。

---

## 12. Compatibility feature

compatibility feature 用来描述第 $i$ 个 specialist 的 intervention 与其他 specialists 是否一致。

先计算：

$$
\operatorname{cos}_{ij,t}^l
=
\cos
\left(
\Delta h_{i,t}^l,
\Delta h_{j,t}^l
\right)
$$

然后：

$$
\operatorname{mean\_compat}_{i,t}^l
=
\frac{1}{N-1}
\sum_{j\neq i}
\operatorname{cos}_{ij,t}^l
$$

$$
\operatorname{neg\_compat}_{i,t}^l
=
\sum_{j\neq i}
\max(0,-\operatorname{cos}_{ij,t}^l)
$$

compatibility 不再作为 hard filter，而是作为 router 输入特征。

第一版可以先不加 compatibility；第二版再加，用于判断 compatibility 是否提供增益。

---

## 13. 推荐 router input 分阶段设计

为了方便归因，不要第一版就把所有 feature 都加上。

### 13.1 Input-A：最小主方案

$$
x_{i,t}^l
=
[
 a_{i,t}^l;
 s_{i,t}^l;
 g_t^l;
 c
]
$$

其中：

$$
a_{i,t}^l
=
P_{R,l,i}
\frac{
R_i^l h_{\mathrm{pre},t}^l
}{
\|R_i^l h_{\mathrm{pre},t}^l\|_2+
\epsilon
}
$$

$$
s_{i,t}^l
=
\log
\left(
\|R_i^l h_{\mathrm{pre},t}^l\|_2+
\epsilon
\right)
$$

这组用于验证：

$$
\text{local }R_i h + \text{ hidden }g_t^l + \text{ global }c
$$

是否足够学习正确 routing。

### 13.2 Input-B：加入 compatibility

$$
x_{i,t}^l
=
[
 a_{i,t}^l;
 s_{i,t}^l;
 \operatorname{mean\_compat}_{i,t}^l;
 \operatorname{neg\_compat}_{i,t}^l;
 g_t^l;
 c
]
$$

这组用于验证 compatibility 是否帮助 router 做正确 routing。

### 13.3 Input-C：加入 delta

$$
x_{i,t}^l
=
[
 a_{i,t}^l;
 b_{i,t}^l;
 s_{i,t}^l;
 \operatorname{mean\_compat}_{i,t}^l;
 \operatorname{neg\_compat}_{i,t}^l;
 g_t^l;
 c
]
$$

这组表达力更强，但归因更复杂，建议后跑。

---

## 14. 推荐维度

当前建议主配置：

$$
d_R = 32
$$

$$
d_h = 64
$$

$$
d_c = 64
$$

Input-A 的维度约为：

$$
d_{\text{in}} = 32 + 1 + 64 + 64 = 161
$$

scorer 可以使用：

$$
s_l:
\mathbb R^{161}
\rightarrow
\mathbb R
$$

例如：

$$
161 \rightarrow 128 \rightarrow 1
$$

如果加入 compatibility，输入维度变成：

$$
161 + 2 = 163
$$

如果加入 delta，则再增加 $d_\delta$ 和若干 norm 标量。

---

## 15. KL loss：最小可归因版本

暂时不要引入复杂 loss。

只验证一个问题：

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

其中：

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

teacher distribution 用 soft target：

$$
p_y(i)
=
\begin{cases}
0.8, & i=y \\
\frac{0.2}{N-1}, & i\neq y
\end{cases}
$$

如果 $N=4$，非目标 specialist 权重为：

$$
\frac{0.2}{3}\approx 0.0667
$$

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
p_y(i)
\log \alpha_i
+
\text{const}
$$

第一轮只试：

$$
\lambda_{\text{KL}}\in\{0,0.02,0.05\}
$$

不要加 entropy、margin、front/back weighting、layer weighting，避免不可归因。

---

## 16. 推荐最小实验矩阵

### Exp-0：E6 baseline

所有层使用当前固定 E6 routing，不训练 router。

目的：确认当前 E6 的效果和错误 routing 现象。

### Exp-1：Router-all，no KL

所有 LoReFT 层都安装 trainable router：

$$
\mathcal L_R = \mathcal L_{\text{LoReFT}}
$$

训练目标：

$$
\mathcal L = \mathcal L_{\text{answer}}
$$

目的：验证只靠 answer loss，every-layer router 是否能学到正确 routing。

### Exp-2：Router-all + weak KL

$$
\mathcal L
=
\mathcal L_{\text{answer}}
+
0.02\mathcal L_{\text{KL}}
$$

目的：验证弱 task-level routing supervision 是否有帮助。

### Exp-3：Router-all + medium KL

$$
\mathcal L
=
\mathcal L_{\text{answer}}
+
0.05\mathcal L_{\text{KL}}
$$

目的：判断更强 KL 是否进一步提升，还是开始伤害生成效果。

---

## 17. 评估与 debug 指标

不要只看总 loss。

需要看每个任务上的 router 行为。

对于 toxicity 样本，看：

$$
\mathbb E[\alpha_{\text{tox}}]
$$

以及：

$$
\mathbb E[\alpha_{\text{tox}}]
-
\mathbb E[\alpha_{\text{truth}}]
$$

对于 ethics 样本，看：

$$
\mathbb E[\alpha_{\text{moral}}]
$$

以及：

$$
\mathbb E[\alpha_{\text{moral}}]
-
\mathbb E[\alpha_{\text{truth}}]
$$

但这里的解释应该是“routing 是否正确”，而不是“防 truthful”。truthful 只是一个 specialist，若它在某些任务中不该主导却主导了，就是错误 routing。

还应分别统计 front-7 和 back-7：

$$
\alpha_{\text{front}}^l
=
\frac{1}{7}
\sum_{t=1}^{7}
\alpha_t^l
$$

$$
\alpha_{\text{back}}^l
=
\frac{1}{7}
\sum_{t=T_p-6}^{T_p}
\alpha_t^l
$$

back-7 更接近 answer generation，通常更关键。

还应统计：

$$
\operatorname{dominant\_rate}_{i,l}
=
\Pr
\left[
 i=\arg\max_j\alpha_{j,t}^l
\right]
$$

以及：

$$
\mathcal H(\alpha_t^l)
=
-
\sum_i
\alpha_{i,t}^l
\log \alpha_{i,t}^l
$$

如果 KL 后 $\alpha$ 过早接近 one-hot，但生成指标变差，说明 KL 过强。

---

## 18. 当前推荐主方案

当前最推荐先实现：

$$
\boxed{\text{Every-layer router + Input-A + answer loss}}
$$

也就是：

$$
\mathcal L_R = \mathcal L_{\text{LoReFT}}
$$

$$
x_{i,t}^l
=
[
 a_{i,t}^l;
 s_{i,t}^l;
 g_t^l;
 c
]
$$

$$
q_{i,t}^l = s_l(x_{i,t}^l)
$$

$$
\alpha_{i,t}^l
=
\operatorname{softmax}_i
\left(
\frac{q_{i,t}^l}{T}
\right)
$$

$$
\mathcal L = \mathcal L_{\text{answer}}
$$

然后再做：

$$
\boxed{\text{Every-layer router + Input-A + }0.02\mathcal L_{\text{KL}}}
$$

这样可以清楚归因 KL 是否有效。

---

## 19. 一句话总结

当前路线应从：

$$
\text{少数层 router + non-router E6/off}
$$

改成：

$$
\boxed{\text{every-layer trainable router}}
$$

因为 single specialist 原本就是 every-layer LoReFT。

router 每一层、每个被干预 prompt token、每个 specialist 输出一个 score，再对 specialists 做 softmax。

router 输入不应只看 token hidden，而应包含：

$$
\boxed{\text{specialist-local feature} + \text{hidden state feature} + \text{global feature}}
$$

第一版先用：

$$
R_i h \text{ direction/norm} + h_{\mathrm{pre}}^l \text{ projection} + \text{base-only global feature}
$$

再逐步 ablate compatibility、delta feature 和 KL。
