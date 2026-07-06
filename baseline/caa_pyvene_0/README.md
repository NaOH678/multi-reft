# CAA Pyvene Baseline

这个目录保留 `CAA` 的原始训练数据与向量训练方式，但把推理时的注入后端改成 `pyvene`。

当前实现约束：

- 干预模块：`residual stream`
- 干预位置：`block_output`
- 干预 token：只对 `prompt` 结束后的所有生成 token 生效
- 干预形式：`h = h + alpha * steering_vec`

对应到代码：

- 训练：复用 [`baseline/CAA_0/train_vectors.py`](/mnt/hwfile/shichaojian/multi-reft/baseline/CAA_0/train_vectors.py)
- 推理注入：复用 [`baseline/pyvene_additive_common.py`](/mnt/hwfile/shichaojian/multi-reft/baseline/pyvene_additive_common.py)
- 评测入口：`run_eval_experiment.sh`

日志与产物结构：

- 训练日志：`baseline/caa_pyvene_0/runs/train/...`
- 训练向量：`baseline/caa_pyvene_0/artifacts/vectors/...`
- 测试日志：`baseline/caa_pyvene_0/runs/eval/...`

说明：

- 这里只替换注入后端，不改你现在已经能跑的手搓 `CAA_0`。
- 如果要做 naive composition，给多个 `CAA_VECTOR_DIRS`，再设 `CAA_COMPOSITION=sum|mean|norm_mean|weighted_sum`。

