# Composable LoReFT 总体结果分析（截至 2026-04-23）

本文档用于汇总此前的 `delta / residual` 系列方法、`delta normalization` 系列方法，以及最新一轮 `policy follow-up` 方法在四个 benchmark 上的表现：

- `truthfulqa_mc`
- `bbq`
- `ethics`
- `toxicity`

其中：

- `truthfulqa_mc / bbq / ethics` 指标越高越好
- `toxicity` 为 Detoxify 平均毒性分数，越低越好

但需要特别强调：

- `toxicity` 当前不能只看分数，还必须结合生成内容判断是否答非所问
- `ethics` 当前也存在明显的 truth-style QA 模板污染问题

## 1. 需要对比的方法族

### 1.1 旧方法

主要包括三类：

1. raw residual routing
   - `residual_softmax(temp=1)`
   - `residual_softmax(temp=64)` 等温度扫描

2. residual normalization
   - `residual_scaled_softmax`
   - `residual_logz_softmax`

3. calibration stats 形式
   - `shared`
   - `specialist`

### 1.2 新方法（policy follow-up）

本轮 follow-up 主要比较：

- `compat_filtered_topk`
- `intervention_softmax`
- score source:
  - `delta_norm`
  - `intervention_norm`
- score normalization:
  - `none`
  - `mean_ratio`
  - `log_zscore`

共 8 组：

- E1 `compat_delta_none`
- E2 `compat_delta_mean`
- E3 `compat_delta_logz`
- E4 `intervention_none`
- E5 `compat_intervention_mean`
- E6 `compat_intervention_logz`
- E7 `intervention_mean`
- E8 `intervention_logz`

## 2. 补录总表（结合旧服务器结果）

这一节用于补录你在旧服务器上已经确认过的结果。它不是替换后文原始分析，而是把此前分散在不同日志和文档中的关键结果统一拉平到一张表里，便于横向比较。

说明：

- `truthfulqa_mc / bbq / ethics` 越高越好
- `toxicity` 越低越好，但必须结合生成内容判断是否答非所问
- `single(task-matched)` 指每个 benchmark 使用与任务最匹配的单 specialist，不是统一 composable 配置
- `-` 表示当前没有完整可对齐结果

| 配置 | TruthfulQA MC | BBQ | Ethics | Toxicity |
| --- | ---: | ---: | ---: | ---: |
| `single(task-matched)` | 0.494492 | 0.555136 | 0.913387 | 0.002650 |
| `equal` | 0.495716 | 0.570557 | 0.884780 | 0.074915 |
| `residual_softmax(temp=1)` | 0.567931 | 0.549853 | 0.878141 | 0.012701 |
| `topk_residual(k=1,temp=1)` | 0.500612 | 0.507163 | 0.685949 | 0.004155 |
| `shared residual_scaled(temp=1)` | 0.608323 | 0.577139 | 0.858622 | 0.060326 |
| `special residual_scaled(temp=1)` | 0.602203 | 0.576130 | 0.857120 | 0.061494 |
| `shared residual_logz(temp=1)` | 0.536108 | 0.616922 | 0.862257 | 0.072427 |
| `special residual_logz(temp=1)` | 0.545900 | 0.616631 | 0.867157 | 0.070167 |
| `shared residual_scaled(temp=64)` | - | - | 0.881935 | 0.085214 |
| `special residual_scaled(temp=64)` | - | - | 0.879485 | 0.084324 |
| `shared residual_logz(temp=64)` | - | - | 0.888494 | 0.080370 |
| `special residual_logz(temp=32)` | - | - | 0.890153 | 0.081085 |
| `shared_latent equal` | 0.294982 | - | - | - |
| `shared_latent residual_softmax` | 0.007344 | - | - | - |
| `projected_output residual_softmax` | 0.073439 | - | - | - |
| `bbq raw residual_softmax(temp=64)` | - | 0.592679 | - | - |
| E1 `compat_delta_none` | 0.505508 | 0.523063 | 0.812233 | 0.007585 |
| E2 `compat_delta_mean` | 0.553244 | 0.579395 | 0.878853 | 0.014475 |
| E3 `compat_delta_logz` | 0.565483 | 0.579635 | 0.880828 | 0.018368 |
| E4 `intervention_none` | 0.571603 | 0.549665 | 0.875533 | 0.012447 |
| E5 `compat_intervention_mean` | 0.554468 | 0.578917 | 0.880354 | 0.014483 |
| E6 `compat_intervention_logz` | 0.566707 | 0.579908 | 0.883120 | 0.018250 |
| E7 `intervention_mean` | 0.607099 | 0.577549 | 0.856646 | 0.038437 |
| E8 `intervention_logz` | 0.543452 | 0.615896 | 0.865260 | 0.041803 |

直接从这张补录表看，当前最重要的结论有四点：

1. 旧 `residual/delta normalization` 方法并没有被 follow-up 全面超越。
2. `TruthfulQA MC` 最好的是 `shared residual_scaled(temp=1)=0.608323`，不是 follow-up。
3. `BBQ` 最好的是 `shared residual_logz(temp=1)=0.616922`，也不是 follow-up。
4. `Ethics` 的最佳 composable 结果仍是 `special residual_logz(temp=32)=0.890153`；follow-up 里最接近的是 `E6=0.883120`。

## 3. 各 benchmark 总表

说明：

- `-` 表示该方法没有完整覆盖该 benchmark，或当前没有可直接对齐的完整结果
- `toxicity` 的低分未必表示真正更好，后文会专门分析

| 配置 | truthfulqa_mc | bbq | ethics | toxicity |
| --- | ---: | ---: | ---: | ---: |
| `base model` | 0.4720 |  0.3352 | 0.3650 | 0.366 |
| `+sft` | 0.4517 |  0.5543 | 0.8836 | 0.0078 |
| `+LoRA` | 0.4418 | 0.5501 | 0.8845 | 0.0027 |
| raw `residual_softmax temp=1` | 0.5679 | 0.5499 | - | - |
| raw `residual_softmax temp=64` | - | 0.5927 | - | - |
| `shared residual_scaled temp=1` | 0.6083 | 0.5771 | 0.8586 | 0.0603 |
| `shared residual_logz temp=1` | 0.5361 | 0.6169 | 0.8623 | 0.0724 |
| `specialist residual_scaled temp=1` | 0.6022 | 0.5761 | 0.8571 | 0.0615 |
| `specialist residual_logz temp=1` | 0.5459 | 0.6166 | 0.8672 | 0.0702 |
| `shared residual_scaled temp=64` | - | - | 0.8819 | 0.0852 |
| `specialist residual_scaled temp=64` | - | - | 0.8795 | 0.0843 |
| `shared residual_logz temp=64` | - | - | 0.8885 | 0.0804 |
| `specialist residual_logz temp=32` | - | - | 0.8902 | 0.0811 |
| E1 `compat_delta_none` | 0.5055 | 0.5231 | 0.8122 | 0.0076 |
| E2 `compat_delta_mean` | 0.5532 | 0.5794 | 0.8789 | 0.0145 |
| E3 `compat_delta_logz` | 0.5655 | 0.5796 | 0.8808 | 0.0184 |
| E4 `intervention_none` | 0.5716 | 0.5497 | 0.8755 | 0.0124 |
| E5 `compat_intervention_mean` | 0.5545 | 0.5789 | 0.8804 | 0.0145 |
| E6 `compat_intervention_logz` | 0.5667 | 0.5799 | 0.8831 | 0.0183 |
| E7 `intervention_mean` | 0.6071 | 0.5775 | 0.8566 | 0.0384 |
| E8 `intervention_logz` | 0.5435 | 0.6159 | 0.8653 | 0.0418 |

这一节表格更强调“当前仓库内已统一整理的主干对比”；如果要看最完整版本，应以上一节“补录总表”为准。

## 4. 按 benchmark 的对比结论

### 3.1 TruthfulQA MC

当前最强结果：

- `shared residual_scaled temp=1`：`0.6083`

follow-up 中最强：

- `E7 intervention_mean`：`0.6071`

观察：

- 旧的 `residual_scaled_softmax` 仍然是 TruthfulQA 上最强的统一 composable 方法之一。
- `E7` 非常接近它，但 `E7` 在 `ethics` 和 `toxicity` 上退化较明显，因此不能仅凭 TruthfulQA 单项最好就作为主线。
- `E6/E3` 虽然弱于 `shared residual_scaled temp=1`，但仍保持在一个可接受区间，并换来了更好的整体均衡性。

### 3.2 BBQ

当前最强结果：

- `shared residual_logz temp=1`：`0.6169`

follow-up 中最强：

- `E8 intervention_logz`：`0.6159`

观察：

- BBQ 上最强方法仍然是旧的 `residual_logz_softmax`。
- `E8` 很接近，但它在 `toxicity` 上明显更差。
- `E6/E3` 在 BBQ 上只有 `0.5799/0.5796`，说明 follow-up 的“更均衡路由”并没有转化成更强的 BBQ 最终表现。

### 3.3 Ethics

当前最强结果：

- `specialist residual_logz temp=32`：`0.8902`

follow-up 中最强：

- `E6 compat_intervention_logz`：`0.8831`

观察：

- 旧的高温 `residual_logz_softmax` 仍然略强于本轮 follow-up。
- 但 `E6` 已经非常接近旧最优，只差约 `0.007`。
- 这说明 `compat_filtered_topk + intervention_norm + log_zscore` 确实接近了之前高温 residual 的效果。
- 不过两者都没有真正解决 ethics 的输出风格问题：模型仍然会大量输出 `the correct answer is wrong/not wrong` 这种 truth QA 模板。

### 3.4 Toxicity

从数值上看：

- follow-up 的 toxicity 分数显著更低，例如：
  - E1: `0.0076`
  - E2/E5: `0.0145`
  - E6: `0.0183`
- 旧 residual-normalized 方法整体更高，例如：
  - `shared residual_scaled temp=1`: `0.0603`
  - `specialist residual_logz temp=32`: `0.0811`

但这不能直接解释为 follow-up 真正更优。

原因是：

- follow-up 的 toxicity 输出里大量出现 truth QA 模板：
  - `the correct answer is false`
  - `the correct answer is true`
  - `the correct response is false`
  - `the correct solution is ...`
- 这类输出并不是对 incomplete prompt 的 continuation，而是答非所问
- Detoxify 对这种输出往往给出较低毒性分数，于是 summary 人工上变“更好”

因此：

- 当前 toxicity 最低分不应被解释成真正最优
- `toxicity` 必须结合生成内容一起分析，而不能只看平均分

## 5. 当前 follow-up 相比旧 delta / normalization 方法，到底带来了什么

### 4.1 没有带来的东西

1. 没有在 `truthfulqa_mc` 上超过最强旧方法
   - 最强旧方法仍是 `shared residual_scaled temp=1`

2. 没有在 `bbq` 上超过最强旧方法
   - 最强旧方法仍是 `shared residual_logz temp=1`

3. 没有在 `ethics` 上超过最强旧方法
   - 最强旧方法仍是 `specialist residual_logz temp=32`

4. 没有真正解决 `toxicity/ethics` 的 truth-template 污染
   - 输出风格仍被 truthful specialist 明显牵引

### 4.2 带来的东西

1. 提供了更可控的 routing 结构
   - `compat_filtered_topk` 不再是“所有专家都直接 softmax 混合”
   - 可以显式观察：
     - top-k 进入率
     - conflict filtering 拒绝率
     - selected rate

2. 显式削弱了 truth specialist 的原始强支配性
   - 相比 `none` 配置，`mean_ratio/log_zscore` 确实降低了 truth alpha
   - 也提升了 moral/stereotype/toxicity specialist 的参与度

3. 在“整体均衡性”上给出了更可继续推进的主线
   - `E6 compat_intervention_logz`
   - `E3 compat_delta_logz`
   - `E5 compat_intervention_mean`
   - `E2 compat_delta_mean`

也就是说，follow-up 的价值主要不在于“直接刷出了新的单项 SOTA”，而在于：

- 给了一个更可分析、更可控、也更适合后续加约束的统一路线

## 6. Debug Summary 角度的对比

旧 residual / delta normalization 的主要问题，在之前文档中已经很明确：

- `truthful` 仍然拿到最高平均 alpha
- late layers 往往仍然 truth-heavy
- 温度升高的收益主要来自“压平极端 truth domination”
- 而不是让 task-matched specialist 真正主导

这一点在本轮 follow-up 中也没有彻底改变。

本轮 follow-up 的变化是：

- `compat_filtered_topk + normalization` 能把 truth alpha 从极端高值拉下来
- 例如在多个 benchmark 上，从约 `0.87~0.88` 降到约 `0.58~0.63`
- 这比 raw `none` 配置更健康

但问题仍在：

- `ethics` 输出依旧全部被拉成 `the correct answer is wrong/not wrong`
- `toxicity` 输出依旧大量被拉成 `the correct answer is true/false`

这说明现在的主要瓶颈，已经不只是“score scale mismatch”，而是：

- truthful specialist 诱导了错误的任务格式

## 7. 综合判断

### 6.1 如果按单项最优看

- `truthfulqa_mc`：旧 `shared residual_scaled temp=1`
- `bbq`：旧 `shared residual_logz temp=1`
- `ethics`：旧 `specialist residual_logz temp=32`
- `toxicity`：当前 follow-up 数值最低，但不可信

### 6.2 如果按统一路线、整体均衡、后续可继续推进看

当前最推荐继续推进的是：

- `E6 compat_intervention_logz`

它的特点：

- `truthfulqa_mc` 没有太差：`0.5667`
- `bbq` 没有太差：`0.5799`
- `ethics` 是本轮最好：`0.8831`
- `toxicity` 分数看起来不差：`0.0183`
- 有 compatibility filtering，可分析性最好

对照配置：

- `E3 compat_delta_logz`
- `E5 compat_intervention_mean`
- `E2 compat_delta_mean`

### 6.3 当前真正的问题

当前统一 no-training 路线的核心问题已经比较清楚：

1. 单纯调 score 和 normalization，不足以解决任务格式污染
2. truthful specialist 太强，不只是“拿走 routing mass”
3. 它还会把输出形式一起拉成 truth QA 风格

因此：

- `ethics` 上，虽然 accuracy 高，但输出格式已经偏成 truth QA
- `toxicity` 上，这种偏移更致命，因为它直接导致 continuation 任务答非所问

## 8. 后续建议

在当前阶段，不建议再仅仅围绕旧 delta / normalization 或 follow-up score 做大规模盲扫。

更值得继续做的是：

1. 以 `E6 compat_intervention_logz` 为主线继续分析
2. 在 toxicity/ethics 上显式加入“任务格式保持”约束
3. 在 routing 或 decoding 层面抑制 truth QA 模板污染
4. 把“是否输出 `the correct answer is ...` / `true/false` / `answer1/2/3`”纳入失败指标

## 9. 最终结论

截至目前，可以用一句话概括：

旧的 `delta / residual normalization` 方法在 `truthfulqa_mc`、`bbq`、高温 `ethics` 上仍然更强；新的 follow-up 方法并没有全面超越它们，但 `compat_filtered_topk + normalization` 提供了一个更均衡、可控、可继续改进的统一方向。当前最大的未解决问题不是单纯 routing 分数不够，而是 `truthful` specialist 仍在系统性地把 `ethics/toxicity` 输出拉成 truth-style QA 模板。
