# iti_0

`iti_0` 是为本项目单独整理的一版 `ITI` baseline。它不是直接照搬 `baseline/honest_llama`，而是保留其核心方法，再接到你们现有的四个评测脚本、日志结构和结果汇总逻辑里。

## 方法定义

这里实现的是标准 `ITI` 思路，而不是把它偷换成 `CAA`：

1. 用正负对比样本提取 activation
2. 在每个 `(layer, head)` 上训练一个 logistic probe
3. 用验证集准确率选择 top-K heads
4. 对每个被选 head，使用：
   - `normalized probe coefficient`
   - 乘以该 head 在训练激活上的 `projection std`
5. 将这些 head-level direction 拼成每层一个 `hidden_size` 维 intervention vector
6. 推理时把该向量注入到 `self_attn.o_proj` 的输入位置

也就是说：

- `CAA_0` 学的是每层均值差向量
- `iti_0` 学的是 head-level probe，再聚合成层向量

## 训练数据

当前固定复用已经整理好的四个 `CAA` 格式训练集：

- `truth`: `dataset/caa/truthful_10k/train.json`
- `bias`: `dataset/caa/bias/train.json`
- `ethics`: `dataset/caa/ethics/train.json`
- `toxicity`: `dataset/caa/toxicity_10k/train.json`

四个任务仍然分成两类输入模式：

- `truth / ethics`
  - `qa_response`
  - 输入是 `question + answer`
- `bias / toxicity`
  - `raw_contrastive`
  - 输入是正负文本本身

训练脚本会做 schema 校验：

- `truth / ethics` 要求 `question` 非空
- `bias / toxicity` 要求 `question` 为空
- 四个任务都要求正负文本非空且不能完全相同

## 代码结构

- `train_iti.py`
  - 训练 ITI probes 并导出 intervention artifact
- `train_iti_intervention.sh`
  - 训练脚本的直接 shell 封装
- `run_train_experiment.sh`
  - 训练实验入口，统一组织日志和产物
- `steering.py`
  - 推理时的 ITI hook 逻辑
  - 也支持 naive composition
- `evaluate_iti_task.sh`
  - 统一评测入口
  - 底层复用本项目已有 truth / bias / ethics / toxicity evaluator
- `run_eval_experiment.sh`
  - 评测实验入口，统一组织日志和结果

## 目录布局

```text
baseline/iti_0/
├── artifacts/
│   └── interventions/
│       └── <model_name>/<task>/<run_name>/
│           ├── layer_<idx>.pt
│           ├── head_val_acc.npy
│           ├── probe_coefficients.npz
│           └── summary.json
├── runs/
│   ├── train/
│   │   └── <model_name>/<task>/<run_name>/<run_ts>/
│   │       ├── train.log
│   │       └── run_manifest.env
│   └── eval/
│       └── <model_name>/<task>/<run_name>/<run_ts>/
│           ├── eval.log
│           ├── run_manifest.env
│           ├── summary.json
│           └── outputs/
│               ├── results.json
│               └── generations.csv
```

约定和 `CAA_0` 一样：

- 训练产物只保存在 `artifacts/`
- 运行日志只保存在 `runs/`
- 不再把同一份 summary 重复拷很多处

## 训练产物

`ITI` 训练后不会生成一个新的 HuggingFace 模型 checkpoint。真正的训练产物是：

- `layer_<idx>.pt`
  - 每层最终 intervention vector
- `head_val_acc.npy`
  - 每层每个 attention head 的验证集 probe 准确率
- `probe_coefficients.npz`
  - 每层每个 head 的 logistic probe 参数
- `summary.json`
  - 本次训练的配置、数据、top heads 和 artifact 信息

## 推理注入位置

当前实现固定注入：

- `self_attn.o_proj.input`

默认行为：

- `token_strategy=last`
- `include_prompt=0`

这意味着默认只对生成阶段的新 token 做 intervention，更贴近 `honest_llama` 的 decode-time ITI 风格。

## 支持的 baseline 形式

### Single-objective

- `ITI-truth`
- `ITI-bias`
- `ITI-ethics`
- `ITI-toxicity`

每个目标单独训练、单独评测。

### Naive composition

`steering.py` 已支持以下组合方式：

- `single`
- `sum`
- `mean`
- `norm_mean`
- `weighted_sum`

因此后续可以直接把多个单目标 artifact 做朴素组合，作为你们组合方法的对照组。

## 训练命令

推荐入口：

```bash
RUN_NAME=truth_default \
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
BATCH_SIZE=8 \
TOP_K_HEADS=48 \
bash baseline/iti_0/run_train_experiment.sh
```

可选参数：

- `MAX_SAMPLES`
- `SOURCE_JSON`
- `VAL_RATIO`
- `TOP_K_HEADS`
- `MAX_ITER`
- `DEVICE`
- `TORCH_DTYPE`
- `KEEP_FEATURE_CACHE`

## 评测命令

单目标评测示例：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
ITI_ARTIFACT_DIR=baseline/iti_0/artifacts/interventions/llama3-8b/truth/truth_default \
ITI_ALPHA=1.0 \
bash baseline/iti_0/run_eval_experiment.sh
```

naive 组合示例：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
ITI_ARTIFACT_DIRS="baseline/iti_0/artifacts/interventions/llama3-8b/truth/truth_default baseline/iti_0/artifacts/interventions/llama3-8b/ethics/ethics_default" \
ITI_COMPOSITION=mean \
ITI_ALPHA=1.0 \
bash baseline/iti_0/run_eval_experiment.sh
```

## 与现有评测脚本的衔接

`iti_0` 现在已经接入：

- `multi_train/eval_truth/evaluate_truth.py`
- `multi_train/eval_ethics/machine_ethics_exp.py`
- `multi_train/eval_toxicity/toxicity_exp.py`

其中：

- `truth` 和 `bias` 继续复用 `evaluate_truth.py`
- `ethics` 继续复用 `machine_ethics_exp.py`
- `toxicity` 继续复用 `toxicity_exp.py`

因此：

- 测试集
- 打分逻辑
- summary 字段

都尽量和现有 `ReFT` / `CAA` 路线保持一致。

## 当前状态

已经完成：

- ITI 训练代码
- ITI 推理 hook
- 训练日志和 artifact 组织
- 评测入口接线
- truth / bias / ethics / toxicity 的 evaluator 参数接入

已做静态校验：

- `py_compile`
- shell `bash -n`
- 容器内 `--help` 导入检查

还没有做的事：

- 真正跑一轮模型训练和评测 smoke test
- 针对你们具体模型做默认 layer / alpha 的系统调参
