# RepE Pyvene Baseline

这个目录保留 `RepE` 的原始 contrastive activation / PCA 向量训练方式，但把推理期注入改成 `pyvene`。

当前实现：

- 训练：复用 [`baseline/repe_0/train_vectors.py`](/mnt/hwfile/shichaojian/multi-reft/baseline/repe_0/train_vectors.py)
- 推理注入：`pyvene` additive intervention
- 目标位置：`block_output`
- 干预 token：对所有生成 token 持续生效
- 干预形式：`h = h + alpha * v`

日志与产物结构：

- 训练日志：`baseline/repe_pyvene_0/runs/train/...`
- 训练向量：`baseline/repe_pyvene_0/artifacts/vectors/...`
- 测试日志：`baseline/repe_pyvene_0/runs/eval/...`
