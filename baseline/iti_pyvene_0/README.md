# ITI Pyvene Baseline

这个目录保留 `ITI` 的原始 probe / intervention vector 训练方式，但把推理期注入改成 `pyvene`。

当前实现：

- 训练：复用 [`baseline/iti_0/train_iti.py`](/mnt/hwfile/shichaojian/multi-reft/baseline/iti_0/train_iti.py)
- 推理注入：`pyvene` head-level intervention
- 目标位置：`head_attention_value_output`
- 干预 token：
  - `truth` 默认只干预 prompt 最后一个 token
  - `bias / ethics / toxicity` 默认对生成 token 生效
- 干预形式：`attn_out = attn_out + alpha * vector`

说明：

- `ITI` 训练出来的 layer vector 仍然来自 head probe 聚合，但评测时按原始 ITI 语义落到 `head_attention_value_output`。

日志与产物结构：

- 训练日志：`baseline/iti_pyvene_0/runs/train/...`
- 训练产物：`baseline/iti_pyvene_0/artifacts/interventions/...`
- 测试日志：`baseline/iti_pyvene_0/runs/eval/...`
