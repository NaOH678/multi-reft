# Composable LoReFT Policy Follow-up 结果分析

日期：2026-04-23

本轮实验比较了 8 组 composable policy follow-up 配置，覆盖 `truthfulqa_mc`、`bbq`、`ethics`、`toxicity` 四个任务。核心目标是：在保证 `truthfulqa_mc` 和 `bbq` 仍然 work 的前提下，让 `ethics` 和 `toxicity` 也尽量 work，并观察是否仍存在 truth specialist 主导导致的答非所问问题。

## 1. 实验配置

8 组配置如下：

| 编号 | 名称 | composition method | score source | normalizer | stats |
| --- | --- | --- | --- | --- | --- |
| E1 | `compat_delta_none` | `compat_filtered_topk` | `delta_norm` | `none` | 无 |
| E2 | `compat_delta_mean` | `compat_filtered_topk` | `delta_norm` | `mean_ratio` | residual shared |
| E3 | `compat_delta_logz` | `compat_filtered_topk` | `delta_norm` | `log_zscore` | residual specialist |
| E4 | `intervention_none` | `intervention_softmax` | `intervention_norm` | `none` | 无 |
| E5 | `compat_intervention_mean` | `compat_filtered_topk` | `intervention_norm` | `mean_ratio` | intervention shared |
| E6 | `compat_intervention_logz` | `compat_filtered_topk` | `intervention_norm` | `log_zscore` | intervention specialist |
| E7 | `intervention_mean` | `intervention_softmax` | `intervention_norm` | `mean_ratio` | intervention shared |
| E8 | `intervention_logz` | `intervention_softmax` | `intervention_norm` | `log_zscore` | intervention specialist |

公共参数：

- `compose_domain=output`
- `composition_temperature=1.0`
- `composition_topk=2`
- `compat_threshold=0.0`
- specialists: truthful, moral, stereotype, toxicity

## 2. Summary 指标

`truthfulqa_mc`、`bbq`、`ethics` 越高越好；`toxicity` 是 Detoxify 平均毒性分数，越低越好。

| 方法 | truthfulqa_mc | bbq | ethics | toxicity |
| --- | ---: | ---: | ---: | ---: |
| E1 `compat_delta_none` | 0.5055 | 0.5231 | 0.8122 | 0.0076 |
| E2 `compat_delta_mean` | 0.5532 | 0.5794 | 0.8789 | 0.0145 |
| E3 `compat_delta_logz` | 0.5655 | 0.5796 | 0.8808 | 0.0184 |
| E4 `intervention_none` | 0.5716 | 0.5497 | 0.8755 | 0.0124 |
| E5 `compat_intervention_mean` | 0.5545 | 0.5789 | 0.8804 | 0.0145 |
| E6 `compat_intervention_logz` | 0.5667 | 0.5799 | 0.8831 | 0.0183 |
| E7 `intervention_mean` | 0.6071 | 0.5775 | 0.8566 | 0.0384 |
| E8 `intervention_logz` | 0.5435 | 0.6159 | 0.8653 | 0.0418 |

主要观察：

- `E6 compat_intervention_logz` 在 `ethics` 上最好，达到 `0.8831`，同时 `truthfulqa_mc=0.5667`、`bbq=0.5799`，整体最均衡。
- `E3 compat_delta_logz` 与 `E6` 非常接近，说明在 `compat_filtered_topk` 下，使用 `delta_norm` 或 `intervention_norm` 做 log-zscore 后行为相似。
- `E7 intervention_mean` 的 `truthfulqa_mc` 最好，达到 `0.6071`，但 `ethics` 和 `toxicity` 明显变差。
- `E8 intervention_logz` 的 `bbq` 最好，达到 `0.6159`，但 `toxicity` 最差之一。
- `E1 compat_delta_none` 的 toxicity 最低，但这不是好现象，因为生成内容显示它大量输出 truth QA 模板，属于答非所问导致 Detoxify 分数低。

## 3. Toxicity 子集指标

toxicity 四个子集的平均毒性分数如下：

| 方法 | toxic/benign | toxic/adversarial | nontoxic/benign | nontoxic/adversarial |
| --- | ---: | ---: | ---: | ---: |
| E1 `compat_delta_none` | 0.01483 | 0.01233 | 0.00166 | 0.00152 |
| E2 `compat_delta_mean` | 0.01516 | 0.03824 | 0.00149 | 0.00301 |
| E3 `compat_delta_logz` | 0.02662 | 0.04267 | 0.00140 | 0.00278 |
| E4 `intervention_none` | 0.02300 | 0.02323 | 0.00205 | 0.00151 |
| E5 `compat_intervention_mean` | 0.01628 | 0.03721 | 0.00149 | 0.00296 |
| E6 `compat_intervention_logz` | 0.02664 | 0.04207 | 0.00146 | 0.00283 |
| E7 `intervention_mean` | 0.04266 | 0.10431 | 0.00238 | 0.00440 |
| E8 `intervention_logz` | 0.04679 | 0.11227 | 0.00310 | 0.00505 |

观察：

- `E1` 的 toxicity 分数最低，但同时 truth-template 污染最高之一，不能视为真正的 detox 成功。
- `E2/E5` 在保持较低 toxicity 的同时，truth-template rate 比 E1/E3/E6/E8 略低，是相对保守的选择。
- `E7/E8` 在 toxic/adversarial 上毒性明显升高，说明 pure `intervention_softmax` 缺少 compatibility filtering 后，安全性不稳定。

## 4. Debug Summary 分析

### 4.1 Truth specialist 主导情况

`E1 compat_delta_none` 下，truth specialist 仍然明显主导：

- `truthfulqa_mc` 上 truth alpha 约 `0.8820`，truth selected rate 约 `0.9544`
- `bbq` 上 truth alpha 约 `0.8824`，truth selected rate 约 `0.9554`
- `ethics` 上 truth alpha 约 `0.8668`，truth selected rate 约 `0.9399`
- `toxicity` 上 truth alpha 约 `0.8785`，truth selected rate 约 `0.9483`

这说明不做 normalization 时，truth specialist 的强度仍然压倒其他 specialist。

### 4.2 Normalization 的作用

`mean_ratio` 和 `log_zscore` 都能明显降低 truth alpha，并提升其他 specialist 的参与度。

以 `toxicity` 为例：

- E1 truth alpha `0.8785`
- E2 truth alpha `0.6315`
- E3 truth alpha `0.5780`
- E5 truth alpha `0.6317`
- E6 truth alpha `0.5767`

其中 `log_zscore` 比 `mean_ratio` 更强地削弱 truth specialist，但也可能让输出更不稳定。

### 4.3 compat_filtered_topk 与 intervention_softmax 对比

`compat_filtered_topk` 的行为：

- 会通过 top-k 和 compatibility filtering 控制 specialist 选择。
- 在 E2/E3/E5/E6 中，truth alpha 降低，但仍保留一定 truth 能力。
- 对 `truthfulqa_mc`、`bbq`、`ethics` 的整体平衡最好。

`intervention_softmax` 的行为：

- 所有 specialist 基本都参与，selected rate 接近 1。
- E7/E8 中 truth alpha 更低，例如 toxicity 上 E8 truth alpha 约 `0.3615`。
- 但因为没有 conflict filtering，toxicity 和 ethics 更容易退化，尤其 toxic/adversarial 毒性升高。

因此当前更推荐继续沿着 `compat_filtered_topk + normalization` 方向改，而不是直接使用 pure `intervention_softmax`。

## 5. 生成内容分析

### 5.1 Ethics

`ethics` 指标看起来很好，但生成内容几乎全部是 truth QA 风格：

- E1: `the correct answer is not wrong`
- E2: `the correct answer is "not wrong".`
- E3: `the correct answer is not wrong`
- E4: `the correct answer is "not wrong"`
- E5: `the correct answer is "not wrong".`
- E6: `the correct answer is not wrong`
- E7: `the correct answer is "not wrong".`
- E8: `the correct answer is "not wrong". it is a proper action`

统计上，8 个方法的 ethics `truth_template_rate` 都是 `1.0000`。

不过 ethics 的 parser 只要识别到 `wrong` 或 `not wrong` 就能算对，因此 summary 指标仍然较高。这说明 ethics 现在是“任务答案基本可用，但输出风格仍然被 truthful QA 模板污染”。

### 5.2 Toxicity

toxicity 的问题更严重，因为该任务要求续写 incomplete prompt，而不是回答选择题或判断题。

典型错误输出：

- `the correct answer is false`
- `the correct answer is true`
- `the correct response is false`
- `the correct solution is to keep generation`

truth-template rate：

| 方法 | truth-template rate |
| --- | ---: |
| E1 `compat_delta_none` | 0.8563 |
| E2 `compat_delta_mean` | 0.7548 |
| E3 `compat_delta_logz` | 0.8344 |
| E4 `intervention_none` | 0.8435 |
| E5 `compat_intervention_mean` | 0.7552 |
| E6 `compat_intervention_logz` | 0.8366 |
| E7 `intervention_mean` | 0.7624 |
| E8 `intervention_logz` | 0.8517 |

因此 toxicity 的低 Detoxify 分数不能直接解释为 detox 成功。很多低分来自“没有续写，而是输出了 truth QA 模板”。

## 6. 当前方法效果判断

### 6.1 推荐保留的方向

当前最值得继续推进：

- `E6 compat_intervention_logz`
- `E3 compat_delta_logz`
- `E5 compat_intervention_mean`
- `E2 compat_delta_mean`

理由：

- `E6` 在 `ethics` 最好，且 `truthfulqa_mc`、`bbq` 不差。
- `E3` 与 E6 接近，可以作为 delta-score 版本对照。
- `E2/E5` 较保守，toxicity 的 truth-template rate 相对较低，虽然仍然很高。

### 6.2 不建议作为主线的方向

不建议将 `E7/E8 intervention_softmax` 作为主线：

- E7 truthfulqa_mc 最好，但 ethics 下降，toxicity 上升。
- E8 bbq 最好，但 toxicity 最差之一。
- debug summary 显示它们确实更均衡地混合 specialist，但缺少 conflict filtering 后任务安全性和稳定性不足。

### 6.3 主要失败模式

本轮最大的失败模式不是路由统计不好，而是输出形式被 truth specialist 主导：

- ethics：答案可解析，但风格完全 truth QA 化。
- toxicity：大量答非所问，低 toxicity 分数不可信。

这和之前 normalization toxicity 实验观察到的现象一致，也和 ethics 上的现象高度相似。

## 7. 后续改进建议

下一步不应该只继续调 score normalization，而要显式缓解 truth QA 模板污染。

建议方向：

1. 对 toxicity 加续写约束或任务格式约束。
   - 当前 toxicity prompt 要求 continuation，但 truth specialist 把它转成判断题。
   - 可以考虑在 decoding 后过滤或在 prompt 中强化“continue exactly from the input, do not answer true/false or correct answer”。

2. 在 routing 中加入 domain-aware 抑制。
   - 例如 toxicity 任务上对 truth specialist 的 score 加惩罚，或者降低 truth 进入 top-k 的概率。
   - 需要保持 token-local ReFT 语义，不做 prompt-fixed routing。

3. 对 `compat_filtered_topk` 加更强的 truth-template debug 指标。
   - 不只看 alpha/selected rate，还要把生成文本中的 `the correct answer is`、`true/false`、`answer1/2/3` 作为 failure metric。

4. 优先继续测试 E2/E3/E5/E6 的变体。
   - 可以调 `compat_threshold`、`composition_topk`。
   - 当前 `topk=2, threshold=0.0` 仍然让 truth 大量参与。

## 8. 结论

本轮 follow-up 的核心结论：

- `compat_filtered_topk + normalization` 的方向是有效的，它确实削弱了 truth specialist 的强主导，并让 `truthfulqa_mc`、`bbq`、`ethics` 保持可用。
- `E6 compat_intervention_logz` 是当前最均衡的配置。
- 但 toxicity 和 ethics 仍然存在明显 truth QA 模板污染。
- toxicity 的低分当前不能直接视为成功，因为大部分输出不是续写，而是答非所问。
- 下一阶段需要把“任务格式保持”和“truth-template 抑制”作为显式目标，而不是只优化 alpha 或 Detoxify 分数。
