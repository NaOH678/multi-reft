# Codex Handoff

## Goal

本项目当前在做：

- 多个独立 LoReFT specialist 的 `no-training` 组合
- 保持原始 ReFT 语义：
  - token-local
  - layer-local
  - 不做 prompt-fixed
  - 不做跨 token pooling
  - 不让前面 token 看到后面 token

当前 4 个 specialist 统一视为 LoReFT：

- `truthful`
- `moral`
- `bias` (`stereotype`)
- `toxicity`

忽略：

- `multi_train/combine_subspace_weights.py`
- `subNodireft` / `SubNodireftIntervention`

## Core Implementation

主要实现是：

- `pyreft/pyreft/interventions.py`
- `multi_train/eval_common/composable_loreft.py`

核心类：

- `ComposableLoreftIntervention`

支持的 `compose_domain`：

- `output`
- `projected_output`
- `shared_latent`

支持的 `policy_type`：

- `single`
- `equal`
- `residual_softmax`
- `intervention_softmax`
- `topk_residual`
- `compat_filtered_topk`
- `residual_scaled_softmax`
- `residual_logz_softmax`

已经接入 composable 的评测：

- truth / bbq
- ethics
- bias
- toxicity

## Important Design Constraints

新会话继续开发时，必须保持这些约束：

- 不引入 prompt-fixed routing
- 不使用 sample-level pooled hidden state 决定 token-level intervention
- 不做“前 token 看后 token”的设计
- 组合策略必须与原始 ReFT 干预语义兼容

## Key Docs To Read First

按这个顺序读：

1. `multi_train/COMPOSABLE_LOREFT_IMPLEMENTATION_STATUS.md`
2. `multi_train/COMPOSABLE_LOREFT_RESIDUAL_STATS_RESULTS.md`
3. `multi_train/COMPOSABLE_LOREFT_ETHICS_TEMP_ANALYSIS.md`
4. `multi_train/COMPOSABLE_LOREFT_EXPERIMENT_RESULTS.md`
5. `multi_train/COMPOSABLE_LOREFT_RESIDUAL_NORMALIZATION_PLAN.md`

## Main Experimental Conclusions

### 1. Truth / BBQ

- `output` 分支明显优于 `shared_latent / projected_output`
- `shared_latent` 之前的尝试整体很差，不是当前优先级
- `truthfulqa_mc` 上，residual-based composable 方法明显有效
- `bbq` 上，composable 也明显有效，而且比单 specialist 更有说服力

当前判断：

- `truth` 基本够了
- `bbq` 基本够了
- 不需要优先再跑 `eval_bias`

原因：

- `bbq` 本质上已经是更有说服力的 bias benchmark

### 2. Ethics

这是当前 no-training 路线里最关键的困难 benchmark。

已经明确的结论：

- raw residual routing 不够
- residual normalization 有帮助
- 调温度有帮助
- 但最佳 composable 仍然低于 `single moral`

最重要的专题文档：

- `multi_train/COMPOSABLE_LOREFT_ETHICS_TEMP_ANALYSIS.md`

核心结论：

- 高温度的收益主要来自“把 truthful 的 extreme dominance 压平”
- 不是因为模型真正学会了 ethics-aware routing
- 最优仍低于 `single moral`

当前判断：

- ethics 上继续盲扫温度已经收益不高

### 3. Toxicity

当前 toxicity 的结论还没有完全补齐。

已经知道：

- raw `residual_softmax(temp=1)` 明显被 `truthful` 主导
- late layers 会塌到 `truthful`
- 生成内容会带 truth-style QA 表面形式

因此 toxicity 仍值得做一轮小规模补测：

- 重点不是大扫参数
- 而是检查 residual normalization 能不能把 toxicity 从这个坏状态里拉回来

## Residual Normalization Status

已经实现：

- `residual_scaled_softmax`
- `residual_logz_softmax`

已经构建的 calibration stats：

- `multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json`
- `multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json`

相关脚本说明：

- `multi_train/COMPOSABLE_LOREFT_RESIDUAL_STATS_USAGE.md`

## Already Tried And Deprioritized

这些方向已经基本不该再作为优先项：

- `shared_latent`
- `projected_output`
- 继续大规模盲扫 raw residual temperature
- 指望纯 `no-training + raw-space heuristic` 自动解决 ethics

## No-Training Methods Still Potentially Worth Trying

如果还要继续补少量 no-training，对新会话来说，只剩这些还有一点价值：

- `intervention_softmax`
- `compat_filtered_topk` on normalized scores
- toxicity 上的小规模 residual-normalized sanity check

其中优先级最高的是：

- toxicity residual normalization sanity check

## Immediate Next Step Recommendation

新会话继续时，优先做这个，不要先发散：

1. 补 toxicity residual normalization 小规模实验
2. 分析：
   - `overall_score`
   - debug summary 的 late-layer alpha
   - 是否仍输出 truth-style QA 格式
3. 如果 toxicity 也救不回来，则认为统一 no-training 路线基本到头
4. 之后再考虑：
   - `intervention_softmax`
   - normalized `compat_filtered_topk`
   - 或直接进入第二阶段 router training

## Where To Look

新 Codex 需要优先知道证据都放在哪里。

### 1. Logs

运行脚本的终端日志一般在：

- `multi_train/logs`

常见用途：

- 看脚本是否真正跑完
- 看哪些任务被跳过 / 复用已有结果
- 看报错、CUDA 分配、文件覆盖等问题

### 2. Truth / BBQ Outputs

truth 和 bbq 的 summary / debug summary 主要在：

- `multi_train/eval_truth`

这里会直接出现：

- `*truthfulqa_mc_summary.json`
- `*truthfulqa_mc_debug_summary.json`
- `*bbq_summary.json`
- `*bbq_debug_summary.json`

### 3. Ethics Outputs

ethics 的主要结果在：

- `multi_train/eval_ethics/data/generations`

重点看：

- `*ethics_summary.json`
- `*ethics_debug_summary.json`
- 对应生成出来的 csv / 文本输出

### 4. Bias Outputs

独立 `eval_bias` 的输出在：

- `multi_train/eval_bias/data/outputs`

但当前研究判断里，更优先使用 `bbq` 作为 bias 证据。

### 5. Toxicity Outputs

toxicity 的 summary 在：

- `multi_train/eval_toxicity/data/summaries`

toxicity 的 debug summary / 生成内容在：

- `multi_train/eval_toxicity/data/generations`

toxicity 目前特别要看：

- `overall_score`
- `late-layer alpha`
- 文本输出是否退化成 truth-style QA 格式

### 6. Calibration Stats

residual normalization 用到的统计量在：

- `multi_train/calibration/train_input/stats/residual_stats_shared_train_input.json`
- `multi_train/calibration/train_input/stats/residual_stats_specialist_train_input.json`

### 7. Scripts

常用脚本目录：

- `multi_train/script`

重点关注：

- truth / bbq / ethics / toxicity 的 composable eval 脚本
- residual stats 构建脚本
- temperature sweep 脚本

## Suggested Toxicity Commands

优先使用：

- `multi_train/script/evaluate_composable_toxicity_residual_stats.sh`

目标配置：

- `shared + residual_scaled_softmax + temp=1`
- `shared + residual_logz_softmax + temp=1`
- `specialist + residual_scaled_softmax + temp=1`
- `specialist + residual_logz_softmax + temp=1`

如需再补一小步：

- `shared + residual_logz_softmax + temp=32`
- `specialist + residual_logz_softmax + temp=32`

## How To Continue In A Fresh Codex Session

在新会话里，先给 Codex 这一段：

```text
继续 multi_train 里的 composable LoReFT 工作。先读：
1. multi_train/CODEX_HANDOFF.md
2. multi_train/COMPOSABLE_LOREFT_IMPLEMENTATION_STATUS.md
3. multi_train/COMPOSABLE_LOREFT_RESIDUAL_STATS_RESULTS.md
4. multi_train/COMPOSABLE_LOREFT_ETHICS_TEMP_ANALYSIS.md

重要约束：
- 保持原始 ReFT 语义
- token-local only
- no prompt-fixed
- no cross-token leakage
- 忽略 multi_train/combine_subspace_weights.py
- 忽略 subNodireft

当前优先级：
- 先补 toxicity residual normalization sanity check
- 去 multi_train/logs 看最近运行日志
- 去 eval_truth / eval_ethics / eval_toxicity 对应目录看 summary 和 debug summary
- 不要先扩展 shared-latent
- 不要先盲扫更多 raw residual temp
```

## Notes

- 用户更偏好把实验过程、结论、分析写进 `md` 文档，方便后续接手和追踪。
- 用户对“统一方法”有偏好，但也明确接受先用实验排除 low-value no-training 方向。
