# Composable LoReFT Ethics Temperature Analysis

## 1. Results

从这两轮温度扫描看，`temp=64` 确实比之前明显更好，但它不是全局最优。

- `shared + residual_scaled_softmax`：`temp=64` 最好，`0.881935`
  - [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_scaled_softmax-temp64-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)
- `specialist + residual_scaled_softmax`：`temp=64` 最好，`0.879485`
- `shared + residual_logz_softmax`：`temp=64` 最好，`0.888494`
- `specialist + residual_logz_softmax`：`temp=32` 最好，`0.890153`
  - [summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp32-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_summary.json)

和之前 `temp=1` 比，提升是实打实的：

- `shared_scaled`：`0.858622 -> 0.881935`，`+0.0233`
- `special_scaled`：`0.857120 -> 0.879485`，`+0.0224`
- `shared_logz`：`0.862257 -> 0.888494`，`+0.0262`
- `special_logz`：`0.867157 -> 0.890153`，`+0.0230`
  - 最优在 `temp=32`
  - `temp=64` 反而回落到 `0.886755`

但和 `single moral = 0.913387` 相比，最好配置仍差 `0.0232`。

- [single moral summary](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-single-idx1-Llama3_8b_Loreft_truthful_4_checkpoint_3330+Llama3_8b_Loreft_moral_checkpoint_330+Llama3_8b_Loreft_stereotype_1_checkpoint_160+Llama3_8b_Loreft_toxicity_checkpoint_235-ethics_summary.json)

## 2. Debug Summary

关键结论是：高温度的收益主要来自“把 extreme routing 压平”，不是因为 `moral` 真正变成了主导 specialist。

以之前较差的 `specialist + logz + temp=1` 为例：

- [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp1-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_debug_summary.json)
- 全层平均 alpha：`truth 0.3653`, `moral 0.1766`
- 第 `29/30` 层：`truth 0.5992 / 0.6492`
- 第 `29/30` 层：`moral 0.0114 / 0.0287`

这说明 late layers 基本被 `truthful` 吃掉了。

到最优的 `specialist + logz + temp=32`：

- [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp32-stats_sp_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_debug_summary.json)
- 全层平均 alpha 已接近均匀：`truth 0.2510`, `moral 0.2472`, `bias 0.2504`, `tox 0.2514`
- 第 `29/30` 层也被压平：`truth 0.2575 / 0.2572`
- `moral` 提到：`0.2345 / 0.2385`

到你提到的 `temp=64`，例如 `shared + logz`：

- [debug](/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/data/generations/llama3-8b-composable-output-residual_logz_softmax-temp64-stats_sh_tr_in-truthful_4_3330+moral_330+stereotype_1_160+toxicity_235-ethics_debug_summary.json)
- 第 `29/30` 层：`moral 0.2422 / 0.2442`

所以 `temp=64` 的改善本质上是：把 `truthful` 的强支配性软化掉，让 4 个 specialist 更接近均分。它不是在“识别 ethics 样本并把权重给 `moral`”，而是在“避免某一个 specialist 过分霸占”。

## 3. How To Read This Result

结论比较明确：

- 温度对 `ethics` 有用，尤其在 residual normalization 之后。
- 有用的机制是“削弱 truth domination”，不是“学会 ethics-aware routing”。
- `logz` 明显比 `scaled` 更稳。
- `scaled` 是一个很明显的 U 型曲线，`temp=8~16` 最差，升到 `32/64` 才回来。
- 最优仍低于 `single moral`，说明仅靠统一 soft routing 还不够。

再看分项，`specialist + logz + temp=32` 相比 `single moral` 主要掉在 `prompt type 2/5`：

- `single`：`0.917496 / 0.914651`
- `composable best`：`0.868658 / 0.880512`

这说明不是少数 prompt type 出现偶然波动，而是 ethics 核心能力本身还被混合稀释了。

## 4. Next Directions

如果下一步继续全力搞 `ethics`，优先考虑两条统一路线：

- 先在 `residual_logz_softmax` 上继续做更强的 residual normalization，而不是再盲扫更高温度。
- 或者引入不破坏统一性的 “soft anchor / bias correction”，目标不是让 `moral` 独占，而是把 ethics 场景下 `truthful` 的系统性偏高再压一点。
