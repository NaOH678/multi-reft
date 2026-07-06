# repe_0

`repe_0` 是为本项目整理的一版 `RepE` baseline。它沿用你们已经固定下来的 `CAA_0 / iti_0` 实验范式：

- 同样的四个训练数据集
- 同样的四个测试入口
- 同样的日志和结果目录结构
- 同样支持单目标和 naive composition

## 方法定义

这里的 `RepE` 实现采用标准的差分表示方向做法：

1. 对每条样本分别构造 positive / negative 文本
2. 在目标层提取最后一个非 padding token 的隐藏状态
3. 计算每条样本的差分：
   - `d_i = h_i^+ - h_i^-`
4. 对一层内所有 `d_i` 做 `PCA(n_components=1)`
5. 取第一主方向作为该层 steering vector
6. 用均值差 `mean(d_i)` 做符号修正
7. 将向量做单位范数归一化
8. 推理时像 `CAA` 一样注入到 decoder layer output

也就是说：

- `CAA`：均值差
- `ITI`：head probe
- `RepE`：差分样本的第一主方向

## 目录布局

```text
baseline/repe_0/
├── artifacts/
│   └── vectors/
│       └── <model_name>/<task>/<run_name>/
│           ├── layer_<idx>.pt
│           └── summary.json
├── runs/
│   ├── train/
│   │   └── <model_name>/<task>/<run_name>/<run_ts>/
│   │       ├── train.log
│   │       ├── run_manifest.env
│   │       └── feature_cache/   # 默认训练后清掉
│   └── eval/
│       └── <model_name>/<task>/<run_name>/<run_ts>/
│           ├── eval.log
│           ├── run_manifest.env
│           ├── summary.json
│           └── outputs/
│               ├── results.json
│               └── generations.csv
```

## 训练数据

固定复用：

- `dataset/caa/truthful_10k/train.json`
- `dataset/caa/bias/train.json`
- `dataset/caa/ethics/train.json`
- `dataset/caa/toxicity_10k/train.json`

输入模式也与 `CAA_0 / iti_0` 保持一致：

- `truth / ethics`
  - `qa_response`
  - 输入是 `question + answer`
- `bias / toxicity`
  - `raw_contrastive`
  - 输入是句子本身

## 训练产物

训练后的真正产物是每层一个向量：

- `layer_<idx>.pt`

另有：

- `summary.json`
  - 记录 layer 指标、PCA 解释方差、数据 schema 等

## 推理注入位置

当前 `RepE` 与 `CAA` 对齐，注入点是：

- `decoder_layer_output`

支持：

- `token_strategy=last`
- `token_strategy=all`

## 组合方式

支持：

- `single`
- `sum`
- `mean`
- `norm_mean`
- `weighted_sum`

## 训练入口

推荐用：

```bash
RUN_NAME=truth_default \
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/xxx \
BATCH_SIZE=8 \
bash baseline/repe_0/run_train_experiment.sh
```

## 评测入口

推荐用：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/xxx \
REPE_VECTOR_DIR=baseline/repe_0/artifacts/vectors/llama3-8b/truth/truth_default \
REPE_ALPHA=1.0 \
bash baseline/repe_0/run_eval_experiment.sh
```

## 当前接线状态

已经接入：

- `multi_train/eval_truth/evaluate_truth.py`
- `multi_train/eval_ethics/machine_ethics_exp.py`
- `multi_train/eval_toxicity/toxicity_exp.py`

因此 `truth / bias / ethics / toxicity` 四个任务都可以直接用同一套项目评测逻辑。
