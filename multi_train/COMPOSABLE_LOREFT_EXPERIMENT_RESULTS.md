# Composable LoReFT Experiment Results

## 1. Scope

本文件整理当前已经跑完并可以直接用于判断方向的实验结果，主要包含：

- TruthfulQA MC 上的 `output / shared_latent / projected_output` 对比
- BBQ 上的 `output` 分支对比
- BBQ 上 `residual_softmax` 的温度扫描
- 当前 debug summary 给出的主要机制结论
- 后续最值得继续补的实验

## 2. 当前最重要的结论

### 2.1 TruthfulQA 上，当前最有效的是 `output + residual_softmax(temp=1)`

在 TruthfulQA MC 上：

| Method | Accuracy |
| --- | ---: |
| `single(idx0 = truthful_4)` | `0.4944920441` |
| `equal` | `0.4957160343` |
| `topk_residual(k=1, temp=1)` | `0.5006119951` |
| `residual_softmax(temp=1)` | `0.5679314565` |
| `shared_latent + equal` | `0.2949816401` |
| `shared_latent + residual_softmax` | `0.0073439412` |
| `projected_output + residual_softmax` | `0.0734394125` |

当前可直接下结论：

- `output` 分支明显优于 `shared_latent / projected_output`
- `residual_softmax(temp=1)` 是当前 TruthfulQA 上的最佳组合方法
- `equal` 和 `single` 都明显不如 `residual_softmax`
- `topk1` 也不如 soft mixture

对应 summary：

- `multi_train/eval_truth/llama3-8b-composable-output-residual_softmax-temp1-...-truthfulqa_mc_summary.json`
- `multi_train/eval_truth/llama3-8b-composable-output-equal-...-truthfulqa_mc_summary.json`
- `multi_train/eval_truth/llama3-8b-composable-output-single-idx0-...-truthfulqa_mc_summary.json`

### 2.2 BBQ 上，低温 residual 不好，但高温 residual 明显有效

在 BBQ 上，初始对比是：

| Method | Accuracy |
| --- | ---: |
| `single(idx2 = stereotype_1)` | `0.5551357451` |
| `equal` | `0.5705566573` |
| `residual_softmax(temp=1)` | `0.5498529713` |
| `topk_residual(k=1, temp=1)` | `0.5071633728` |

这说明：

- BBQ 上低温 `residual_softmax` 不如 `equal`
- BBQ 上 `topk1` 更差
- BBQ 比起强 routing，更适合平滑组合

## 3. BBQ Temperature Scan

目前已经完成的 `residual_softmax` 温度扫描如下：

| Temperature | Accuracy |
| --- | ---: |
| `1` | `0.5498529713` |
| `2` | `0.5654277508` |
| `4` | `0.5548451070` |
| `8` | `0.5671544827` |
| `16` | `0.5794125692` |
| `18` | `0.5814983246` |
| `20` | `0.5833105382` |
| `24` | `0.5866785201` |
| `32` | `0.5898755385` |
| `64` | `0.5926793408` |

对应日志目录：

- `multi_train/logs/composable_truth_20260421_123726`
- `multi_train/logs/composable_truth_20260421_125621`
- `multi_train/logs/composable_truth_20260421_151934`

当前最佳点：

- `residual_softmax(temp=64)` with `accuracy = 0.5926793408`

这个结果已经：

- 明显超过 `equal = 0.5705566573`
- 明显超过 `single(idx2) = 0.5551357451`
- 说明 BBQ 上 residual signal 不是没用，而是需要 **高温 soft arbitration**

## 4. Debug Summary Insights

### 4.1 TruthfulQA: `temp=1` 的 residual routing 是“浅层混合 + 深层 truthful 收口”

TruthfulQA 的 `output + residual_softmax(temp=1)` debug summary 显示：

- `truthful_4` 主导 `31 / 32` 层
- 只有 `layer_27` 被 `stereotype_1` 主导
- 早层仍有明显 soft mixture
- 中后层快速塌缩到 `truthful_4`

这解释了为什么 TruthfulQA 上：

- `residual_softmax` 最好
- `equal` 不够好
- `topk1` 又太硬

当前最合理的机制解释是：

- 低层允许多专家共同参与
- 深层让 `truthful_4` 统一收口

### 4.2 BBQ: 低温 residual 失败，是因为 routing 几乎塌到 `truthful_4`

在 BBQ 上：

- `temp=1` 时，平均 alpha 大约是：
  - truthful `0.7508`
  - moral `0.0623`
  - stereotype `0.1043`
  - toxicity `0.0826`

也就是说，`residual_softmax(temp=1)` 本质上接近：

- 一个被 `truthful_4` 强主导的 routing

这对 BBQ 并不理想，因为 BBQ 任务更需要：

- `stereotype_1`
- `toxicity`
- 以及更平滑的多专家组合

### 4.3 BBQ: 温度升高后，routing 逐步变成“接近 equal，但仍保留结构”

平均 alpha 的变化：

| Temp | truthful | moral | stereotype | toxicity |
| --- | ---: | ---: | ---: | ---: |
| `1` | `0.7508` | `0.0623` | `0.1043` | `0.0826` |
| `16` | `0.3832` | `0.1847` | `0.2126` | `0.2194` |
| `32` | `0.3438` | `0.2034` | `0.2247` | `0.2282` |
| `64` | `0.3081` | `0.2189` | `0.2360` | `0.2370` |

这说明：

- 温度升高后，routing 不再被 `truthful_4` 强行主导
- 但它也不是直接退化成纯 `equal`
- 更像是：
  - 浅层接近均匀
  - 中层轻微偏置
  - 深层仍保留一些结构信息

这正是 BBQ 上当前最有效的组合形态。

### 4.4 BBQ 中几个稳定出现的特殊层

当前有两个比较稳定的模式：

- `layer_24` 常由 `toxicity` 主导
- `layer_27` 常由 `stereotype_1` 主导

随着温度升高：

- 这两个层的主导专家仍保持不变
- 但 alpha 逐步变平，不再极端 one-hot

这说明高温不是在“破坏结构”，而是在“削弱过强的错误偏置”。

### 4.5 `layer_31` 仍然可疑

在多次 debug summary 中，`layer_31` 仍然经常表现为：

- `alpha = 0.25 / 0.25 / 0.25 / 0.25`

这不自然，当前仍然更像：

- 统计问题
- 或最后一层某种广播/缓存异常

因此：

- 暂时不要用 `layer_31` 做机制解释
- 后续应单独排查

## 5. Current Status by Compose Domain

### 5.1 `output`

当前最成熟，也是真正可用于继续做研究判断的分支。

已知：

- TruthfulQA 上效果最好的是 `residual_softmax(temp=1)`
- BBQ 上效果最好的是 `residual_softmax(temp=64)`（当前已扫范围内）

### 5.2 `shared_latent`

当前仍是第一版实现，不稳定。

问题包括：

- TruthfulQA 指标极差
- 生成格式容易崩坏
- 在 BBQ 上还暴露出文件名过长问题，导致部分实验没能正常落盘

因此：

- 当前不适合对 `shared_latent` 下机制结论

### 5.3 `projected_output`

当前也不稳定，整体效果很差。

和 `shared_latent` 一样：

- 先不要做强结论
- 优先修评测与输出稳定性

## 6. Engineering Issues Found So Far

### 6.1 BBQ 下长文件名问题

`shared_latent / projected_output` 的若干 BBQ 实验出现：

- `OSError: [Errno 36] File name too long`

影响：

- 部分实验在生成或保存 summary/debug_summary 时中断
- 这些结果不能直接拿来比较

### 6.2 `layer_31` debug anomaly

如上所述，最后一层经常出现不可信的均匀 alpha。

### 6.3 生成格式问题

在 `shared_latent / projected_output` 下，模型更容易输出：

- `"The correct answer is ..."`
- `"the answer is correct"`
- 重复说明文本

这会污染自动抽取 `answer1/answer2/answer3` 的评测逻辑。

## 7. What Has Been Verified

当前已经可以认为基本成立的经验结论：

1. `output` 分支明显比 `shared_latent / projected_output` 更稳。
2. 不同任务需要的 routing sharpness 不同。
3. TruthfulQA 更适合低温 residual routing。
4. BBQ 更适合高温 residual routing。
5. `topk1` 在当前实验里并不优。
6. BBQ 上最优 routing 不是硬选择，而是高温 soft composition。

## 8. Next Experiments

下面是当前最值得继续做的实验顺序。

### 8.1 First Priority

1. 继续补 BBQ 的高温扫描：
   - `temp = 96`
   - `temp = 128`
   - 目的：确认 `temp=64` 是否仍在上升区间，还是接近最优

2. 做 residual score normalization：
   - global per-specialist z-score
   - layer-wise per-specialist z-score
   - 目的：判断当前问题是否来自 residual norm 在 specialist 间不可比

3. 对比 raw residual 与 normalized residual：
   - TruthfulQA
   - BBQ
   - 目的：验证“raw residual 偏向 truthful_4”是否真是核心原因

### 8.2 Second Priority

4. 在 BBQ 上比较：
   - `equal`
   - `residual_softmax(temp=64)`
   - `residual_softmax(normalized)`
   - 重点看 debug summary 而不只是 accuracy

5. 在 TruthfulQA 上补温度小范围扫描：
   - `temp = 0.5, 1, 2`
   - 目的：确认 TruthfulQA 当前最佳点是不是确实在低温附近

6. 做 `compat_filtered_topk`：
   - 优先在 `output` 分支上做
   - 先跑少量对照，不要立刻全量铺开

### 8.3 Engineering Fixes

7. 修复 `evaluate_truth.py` 中 BBQ 的长文件名问题。

8. 单独排查 `layer_31` 的 debug 统计逻辑。

9. 改进 BBQ / TruthfulQA 的 answer extraction：
   - 减少 `"The correct answer is..."` 这类格式对评测的污染

### 8.4 Later

10. 在 `output` 分支稳定之后，再回头重做：
    - `shared_latent`
    - `projected_output`

11. 只有当前面的 rule-based routing 已经显示稳定趋势后，再考虑：
    - freeze specialists
    - 训练第二阶段 arbitration

## 9. Short Takeaway

当前实验已经说明两件事：

- `output` 分支是可行的，而且已经能跑出有意义的任务差异
- `residual_softmax` 本身不是固定好坏，而是高度依赖任务和温度

更具体地说：

- TruthfulQA 倾向于低温 residual routing
- BBQ 倾向于高温 residual routing

因此下一阶段最值得做的，不是直接上第二阶段训练，而是：

- 继续把 routing score 的可比性和温度行为摸清楚
- 尤其是 residual normalization
