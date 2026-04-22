# E3 Residual Softmax Debug Analysis

## 1. 分析对象

- summary:
  [multi_train/eval_truth/llama3-8b-composable-output-residual_softmax-temp1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-truthfulqa_mc_summary.json](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_softmax-temp1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-truthfulqa_mc_summary.json)
- debug summary:
  [multi_train/eval_truth/llama3-8b-composable-output-residual_softmax-temp1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-truthfulqa_mc_debug_summary.json](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/llama3-8b-composable-output-residual_softmax-temp1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-truthfulqa_mc_debug_summary.json)

实验配置：

- compose domain: `output`
- composition method: `residual_softmax`
- temperature: `1.0`
- 4 个 specialists:
  - `truthful_4`
  - `moral`
  - `stereotype_1`
  - `toxicity`

最终 `accuracy = 0.5679314565483476`，是当前 truth 组合实验里效果最好的一个。

## 2. 主要结论

### 2.1 整体上是“浅层混合，深层塌缩到 truthful”

按每层 `alpha_mean` 的最大值统计，32 层里：

- `truthful_4` 主导 31 层
- `stereotype_1` 主导 1 层
- `moral` 主导 0 层
- `toxicity` 主导 0 层

这说明 `residual_softmax` 虽然表面上是多 specialist soft routing，但在 truth 任务上，实际已经非常明显地偏向 `truthful_4`，尤其是中后层。

### 2.2 早期层不是纯 single，而是存在有效软组合

前几层的 `alpha_mean` 不是 one-hot，说明早期层仍然允许其它 specialist 参与：

- layer 0:
  - truthful `0.3708`
  - moral `0.1864`
  - stereotype `0.2108`
  - toxicity `0.2320`
- layer 4:
  - truthful `0.4790`
  - moral `0.1470`
  - stereotype `0.1819`
  - toxicity `0.1921`

这意味着 `E3` 的提升，不像是“所有层都只选 truthful”，而更像是：

- 低层允许多个 rewrite 共同参与
- 中后层逐渐把控制权收敛到 `truthful_4`

这也是为什么它可能优于 `topk1`。`topk1` 会把这种浅层软混合直接砍掉。

### 2.3 中后层快速塌缩到 truthful_4

一些代表层：

- layer 8: truthful `0.8050`
- layer 12: truthful `0.7473`
- layer 16: truthful `0.8899`
- layer 20: truthful `0.9944`
- layer 24: truthful `0.9223`
- layer 28: truthful `1.0000`
- layer 30: truthful `1.0000`

这说明 residual score 在 deeper layers 上几乎总是把 `truthful_4` 的 residual 推到最强，从而把组合退化成“truthful 主导，其他只在部分层做补充”。

这个行为本身不一定是坏事。对 TruthfulQA 来说，它可能正是有效机制：

- 前层保留少量多专家修正能力
- 后层让 truth specialist 统一收口

## 3. 特殊层

### 3.1 layer 27 是唯一被 stereotype_1 主导的层

`layer_27` 的 `alpha_mean`：

- truthful `0.0000178`
- moral `0.0002438`
- stereotype `0.9522672`
- toxicity `0.0474712`

对应的 `delta_norm_mean` 也明显是 `stereotype_1` 最大：

- truthful `277.09`
- moral `519.89`
- stereotype `592.92`
- toxicity `538.24`

这说明这里不是 softmax 偶然偏了，而是该层该 specialist 的 residual 确实压过其它分支。

当前解释是：

- `stereotype_1` 在这层学到了一类与 truth 任务局部兼容的 rewrite
- residual-based routing 在该层把它识别出来并放大

这个点值得后续继续看，因为它可能是“非 truth specialist 也能对 truth 任务提供局部收益”的直接证据。

### 3.2 layer 31 很可疑

`layer_31` 的 4 个 specialists 完全一致：

- `alpha_mean = 0.25`
- `delta_norm_mean = 1100.6212`
- `intervention_norm_mean = 1100.6213`

4 个 specialist 的这三组统计都完全一样，这非常不自然。

这更像以下几种情况之一：

- 最后一层的 debug 统计写错了，取到了同一份 tensor
- 最后一层 routing score 退化成了完全相同的值
- 某个 broadcast / stack / reshape 逻辑让 4 个 specialist 的统计在 layer 31 被覆盖成同一份结果

因此，`layer_31` 目前不建议直接拿来做机制解释。它首先是一个需要单独排查的实现或统计异常点。

## 4. 对 E3 为什么有效的当前解释

当前最合理的解释是：

1. `residual_softmax` 的确提供了有用 routing 信号。
2. 这个 routing 信号不是把 4 个 specialists 均匀融合，而是自动学出一种“truth-oriented asymmetric composition”。
3. 在 truth 任务里，最有用的模式不是平均组合，也不是每层硬选 top-1，而是：
   - 早层保留 soft mixture
   - 深层主要依赖 `truthful_4`

因此：

- `equal` 效果一般，说明“全都开且均分”不对。
- `topk1` 也一般，说明“每层只留一个”太硬。
- `residual_softmax` 最好，说明 token-local residual 作为 arbitration signal 是有信息量的。

## 5. 现阶段不该过度解读的点

- 不能把 `E3` 解读成“多专家长期协同工作”。从统计上看，它更接近“truthful 主导 + 少量层级补充”。
- 不能用 `layer_31` 支持任何机制结论，因为它的 debug 结果明显异常。
- 目前也还不能说 `moral` 和 `toxicity` 没用。它们虽然没有主导任何层，但可能在早层通过非零 alpha 提供了局部修正。

## 6. 后续建议

最值得继续做的不是马上改方法，而是先把这个分析补齐：

1. 对比 `E1 single truthful` 和 `E3 residual_softmax` 的每层 `alpha / delta_norm / intervention_norm`。
2. 单独检查 `layer_31` 的 forward debug，确认是不是统计写错。
3. 统计每层 top-2 specialist 的 gap，看 `E3` 的收益是否主要来自“少数层的软混合”。
4. 重点看 `layer_27` 这类非-truth specialist 主导层，确认它们是否稳定出现。

## 7. 当前结论的压缩版

`E3` 的核心不是“4 个 specialists 被平衡组合”，而是：

- 浅层存在 soft mixture
- 深层几乎塌缩到 `truthful_4`
- 这种“软混合 + truth 主导收口”的模式，比 `equal` 和 `topk1` 都更适合当前 truth 任务

其中 `layer_27` 是一个值得深挖的有效异常点，`layer_31` 是一个需要先排查的可疑异常点。
