# CAA_0

`CAA_0` 是为本项目单独整理的一版 `CAA` baseline。它不是原始仓库的全量搬运，而是针对本项目实验问题做的可运行实现，目标是和现有 `ReFT` 路线在相同测试集、相同评分逻辑下进行可比实验。

当前这套代码支持两类 baseline：

1. `Single-objective baseline`
   - 每个目标单独训练一个 steering vector
   - 然后在本目标测试集上评估，也可以拿去其他目标测试集上测 cross-objective transfer / interference
2. `Naive composition baseline`
   - 将多个单目标 steering vector 直接做朴素组合
   - 再在现有四个测试集上评估
   - 用来对比你们自己的组合方法是否比简单线性组合更稳定

## 代码结构

- `train_vectors.py`
  - 从 `dataset/caa` 的训练集提取每个任务的 CAA vector
- `train_caa_vector.sh`
  - `train_vectors.py` 的 shell 封装
- `run_train_experiment.sh`
  - 训练实验入口
  - 负责把训练日志、manifest、vector 产物统一放到 `baseline/CAA_0` 下面
- `steering.py`
  - 推理时的 hook 逻辑
  - 也负责 naive composition 的向量组合
- `evaluate_caa_task.sh`
  - 统一评测入口
  - 底层复用本项目已有四个任务的评测脚本
- `run_eval_experiment.sh`
  - 评测实验入口
  - 负责把 generation、summary、日志统一放到 `baseline/CAA_0` 下面

## 目录布局

为了避免像前面 ReFT 实验那样日志过散、同一信息重复太多，当前 `CAA_0` 使用如下较简洁的目录组织：

```text
baseline/CAA_0/
├── artifacts/
│   └── vectors/
│       └── <model_name>/<task>/<run_name>/
│           ├── layer_<idx>.pt
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
│               ├── results.json            # truth / bias
│               └── generations.csv         # ethics / toxicity
```

其中：

- `artifacts/vectors/...`
  - 是 CAA 的“训练产物”
  - 对 CAA 来说，这个就是后续推理真正需要加载的东西
- `runs/train/...`
  - 只保存本次训练运行日志和 manifest
  - 不重复拷贝 vector 文件
- `runs/eval/...`
  - 只保存本次评测的 generation、summary、日志
  - 不额外再拷贝一份同内容 summary 到别的地方

## 训练产物说明

这里需要特别说明：

- ReFT 的训练产物是一个 intervention checkpoint
- CAA 的训练产物不是一个新的 HuggingFace 模型 checkpoint
- CAA 的训练产物是每层的 steering vector

所以如果你问“训练好的模型保存在哪里”，在 `CAA_0` 这套 baseline 里，更准确地说是：

- 训练好的 `CAA vector artifact` 保存到：
  - `baseline/CAA_0/artifacts/vectors/...`

后面评测加载的也是这些向量，而不是新的完整模型权重。

## 训练数据

当前固定使用以下四个训练集：

- `truth`: `dataset/caa/truthful_10k/train.json`
- `bias`: `dataset/caa/bias/train.json`
- `ethics`: `dataset/caa/ethics/train.json`
- `toxicity`: `dataset/caa/toxicity_10k/train.json`

这四个数据集都已经被整理成 CAA 所需的对比格式：

- `question`
- `answer_matching_behavior`
- `answer_not_matching_behavior`

## 数据适配策略

### truth / ethics

这两个任务大多数样本有显式问题，因此训练时采用本项目 ReFT 风格模板，把 `question + answer` 拼起来后再提激活：

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
{answer}
```

这样做的原因是尽量让 `CAA` 的输入分布贴近你们项目里原本的指令跟随格式。

### bias / toxicity

这两个任务迁移后的 CAA 数据里，`question` 为空，核心信息都在正负文本本身。因此这里不强行套原版 CAA 的 `prompt + 正负回答` 形式，而是直接对文本本身取激活。

这个选择主要参考了 `baseline/2602.17560v2.pdf` 这一类做法里的 detoxification / sentence-only contrastive 处理方式：没有明确 question 时，直接用文本本身提行为方向，而不是伪造一个问答 prompt。

代码里现在也是**显式区分**这两类输入模式，而不是依赖 `question=""` 的隐式行为：

- `truth / ethics`
  - `task_input_mode = qa_response`
  - 输入给模型的是 `question + answer`
- `bias / toxicity`
  - `task_input_mode = raw_contrastive`
  - 输入给模型的是正负文本本身

也就是说，后两者不会再被当成和前两者同一种 `prompt+response` 任务来处理。

此外，训练脚本现在会做严格 schema 校验：

- `truth / ethics`
  - 要求 `question` 必须非空
- `bias / toxicity`
  - 要求 `question` 必须为空
- 四个任务都要求：
  - `answer_matching_behavior` 和 `answer_not_matching_behavior` 必须都非空
  - 正负文本不能完全相同

如果数据格式不符合当前任务的预期，`train_vectors.py` 会直接报错，而不会静默继续训练。

## 向量训练方法

当前实现的向量训练逻辑是：

1. 对每条正例文本和负例文本分别编码
2. 抽取每条序列最后一个非 padding token 的隐藏状态
3. 对每层分别求：

```text
v_layer = mean(h_positive) - mean(h_negative)
```

4. 每层保存一个向量文件：

- `layer_<idx>.pt`
- `summary.json`

也就是说，这里实现的是项目内最直接、最透明的一版 CAA，不引入额外分类头或额外训练器。

## Single-objective baseline

这类 baseline 的含义是：

- `CAA-truth`
- `CAA-bias`
- `CAA-ethics`
- `CAA-toxicity`

每次只训练一个任务的 vector，然后单独评测。

它主要回答两个问题：

1. 单目标 steering 在自己的目标上能达到什么水平
2. 单目标 steering 会不会干扰其他目标

这类实验非常适合拿来展示：

- on-target 可能有效
- off-target 可能不稳定
- 存在 cross-objective interference

### 训练示例

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
LAYERS="14 15 16" \
BATCH_SIZE=8 \
bash baseline/CAA_0/train_caa_vector.sh
```

默认输出目录：

```text
baseline/CAA_0/artifacts/vectors/<base_model_name>/<task>/<run_name>/
```

### 单目标评测示例

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/truth/default \
CAA_LAYERS="14 15 16" \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/evaluate_caa_task.sh
```

### 推荐训练入口

如果你希望日志、manifest、训练产物都自动整理到 `baseline/CAA_0` 下，推荐不要直接调用 `train_caa_vector.sh`，而是调用：

```bash
RUN_NAME=truth_l14_16 \
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
LAYERS="14 15 16" \
BATCH_SIZE=8 \
bash baseline/CAA_0/run_train_experiment.sh
```

这会自动生成：

- `baseline/CAA_0/artifacts/vectors/...`
- `baseline/CAA_0/runs/train/...`

包装脚本的行为是：

- 终端只打印本次运行的目录和日志路径
- 详细训练过程只写入 `train.log`

### 推荐评测入口

如果你希望 generation、summary、日志都自动整理到 `baseline/CAA_0` 下，推荐调用：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/truth/truth_l14_16 \
CAA_LAYERS="14 15 16" \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/run_eval_experiment.sh
```

这会自动生成：

- `baseline/CAA_0/runs/eval/.../eval.log`
- `baseline/CAA_0/runs/eval/.../summary.json`
- `baseline/CAA_0/runs/eval/.../outputs/...`

包装脚本的行为是：

- 终端只打印本次运行的目录、日志、summary、输出文件路径
- 逐条样本输出和底层评测信息只写入 `eval.log`
- 真正的模型回答保存在 `outputs/` 下，不会再同步刷到终端

### 自动选层

如果你不想手动猜 `CAA_LAYERS`，现在可以让 `CAA_0` 自动做两阶段单层 sweep，并选择当前任务上分数最高的那一层。

默认逻辑是：

1. 先做 `coarse sweep`
   - 从完整候选层里均匀挑出一小组代表层
2. 再做 `fine sweep`
   - 围绕 coarse stage 的最佳层，在邻域内细扫
3. 在验证集上选出最终最佳层
4. 用这个最终最佳层在正式测试集上只跑一次最终评测

如果候选层本来就很少，或者 fine 邻域里没有新增层可测，这个流程会自动退化成接近单阶段的行为，不会额外浪费评测时间。

当前已经接入“选择集”和“最终测试集”分离逻辑：

- `selection phase`
  - 专门用于选 layer
- `final phase`
  - 专门用于最终正式结果

对 `truth` 任务，如果本地存在：

```text
dataset/truthfulqa_mc_dev/test.json
```

那么在 `AUTO_SELECT_LAYER=1` 时：

- 选层阶段默认会用 `truthfulqa_mc_dev`
- 最终阶段仍然用正式测试集 `truthfulqa_mc`

也就是说，truth 已经默认按“验证集选层，测试集报最终结果”的方式工作。

开启方式：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/truth/truth_l14_16 \
CAA_ALPHA=0.2 \
AUTO_SELECT_LAYER=1 \
bash baseline/CAA_0/run_eval_experiment.sh
```

如果你想显式指定选层集，也可以写：

```bash
TASK=truth \
TRUTH_DATASET=truthfulqa_mc \
LAYER_SELECT_TRUTH_DATASET=truthfulqa_mc_dev \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/truth/truth_default \
CAA_ALPHA=0.2 \
AUTO_SELECT_LAYER=1 \
bash baseline/CAA_0/run_eval_experiment.sh
```

可选参数：

- `CAA_LAYER_CANDIDATES="12 16 20 24"`
  - 完整候选层池
- `CAA_COARSE_LAYER_CANDIDATES="12 20 24"`
  - 手动指定 coarse stage 的候选层
  - 如果不设，脚本会从完整候选层池里自动均匀抽样
- `CAA_COARSE_SWEEP_POINTS=6`
  - 自动 coarse sweep 时默认抽多少个代表层
- `CAA_FINE_SWEEP_RADIUS=2`
  - fine stage 默认在 coarse 最佳层的前后各扩几层
- `LAYER_SELECT_TRUTH_DATASET=truthfulqa_mc_dev`
  - truth 任务选层阶段使用的数据集
- `LAYER_SELECT_BIAS_DATASET=bbq`
  - bias 任务选层阶段使用的数据集
- `LAYER_SELECT_ETHICS_DATASET_FILE=...`
  - ethics 任务选层阶段使用的数据文件
- `LAYER_SELECT_TOXICITY_DATASET_ROOT=...`
  - toxicity 任务选层阶段使用的数据根目录
- `LAYER_SELECT_TOXICITY_DATASETS="toxic nontoxic"`
  - toxicity 任务选层阶段使用的数据子集
- `LAYER_SELECT_TOXICITY_PROMPTS="benign adversarial"`
  - toxicity 任务选层阶段使用的 prompt 类型

输出目录里会额外生成：

- `layer_selection/coarse/`
  - coarse stage 每层的中间结果
- `layer_selection/fine/`
  - fine stage 每层的中间结果
- `layer_selection_summary.json`
  - 两阶段 sweep 的完整汇总，包括：
    - 全量候选层
    - coarse 候选层与分数
    - fine 候选层与分数
    - 最终选中的 layer

## 其余三个子任务

`bias / ethics / toxicity` 现在已经接成和 `truth` 同一套训练、评测、日志框架，不再需要单独写临时脚本。

当前默认行为是：

- `truth`
  - 不自动指定固定单层
  - 仍然建议手动指定或走自动选层
- `bias / ethics / toxicity`
  - 如果你没有显式传 `LAYERS` 或 `CAA_LAYERS`
  - 会自动使用一个默认单层

当前默认单层映射参考 `ODESTEER` 的 `LLaMA3.1-8B -> layer 14` 设定：

- `llama3-8b` 默认 `layer 14`
- `mistral-7b` 默认 `layer 16`
- `qwen25-7b` 默认 `layer 14`
- 其他未显式登记模型先回退到 `layer 14`

也就是说，对你现在的 `llama3-8b`，后面三个任务如果不手动传 layer，就会直接走 `layer 14`。

### bias 训练

```bash
RUN_NAME=bias_default \
TASK=bias \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
BATCH_SIZE=8 \
bash baseline/CAA_0/run_train_experiment.sh
```

### bias 测试

```bash
TASK=bias \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/bias/bias_default \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/run_eval_experiment.sh
```

### ethics 训练

```bash
RUN_NAME=ethics_default \
TASK=ethics \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
BATCH_SIZE=8 \
bash baseline/CAA_0/run_train_experiment.sh
```

### ethics 测试

```bash
TASK=ethics \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/ethics/ethics_default \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/run_eval_experiment.sh
```

### toxicity 训练

```bash
RUN_NAME=toxicity_default \
TASK=toxicity \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
BATCH_SIZE=8 \
bash baseline/CAA_0/run_train_experiment.sh
```

### toxicity 测试

```bash
TASK=toxicity \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIR=baseline/CAA_0/artifacts/vectors/llama3-8b/toxicity/toxicity_default \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/run_eval_experiment.sh
```

### 日志与结果位置

三个任务和 `truth` 完全一样：

- 训练日志：
  - `baseline/CAA_0/runs/train/<model>/<task>/<run_name>/<run_ts>/`
- 向量产物：
  - `baseline/CAA_0/artifacts/vectors/<model>/<task>/<run_name>/`
- 评测日志与结果：
  - `baseline/CAA_0/runs/eval/<model>/<task>/<run_name>/<run_ts>/`

其中重点文件仍然是：

- `train.log`
- `eval.log`
- `run_manifest.env`
- `summary.json`
- `outputs/results.json` 或 `outputs/generations.csv`

### 注意

- `toxicity` 评测仍依赖 `detoxify`
- 如果环境里没装，测试阶段会在分析环节报错
- 当前这三个任务先走固定默认层，不做验证集选层
- 如果不设 `CAA_LAYER_CANDIDATES`
  - 就会自动读取当前 vector 目录里已有的全部层

自动选层运行后，会在当前 run 目录下额外写出：

- `layer_selection_summary.json`
- `layer_selection/`
  - 每个候选层各自的 summary 和输出文件

当前自动选层策略是：

- `truth / bias`
  - 比较 `accuracy`
- `ethics / toxicity`
  - 比较 `overall_score`

注意：

- 当前实现是基于“当前任务评测数据”做 layer sweep，再选出最好层
- 所以它更像自动调参版 baseline
- 如果你后面想完全对齐论文式 protocol，最好再单独准备 dev / validation split 来选层

## Naive composition baseline

现在这套代码也支持你要求的 naive 组合实验。

核心思想是：先训练出多个单任务 vector，然后在推理时直接组合，不做你们自己方法里的更复杂融合。

### 支持的组合方式

当前 `steering.py` 支持以下几种组合规则：

- `single`
  - 单个 vector，等价于 single-objective
- `sum`
  - 直接逐层相加
- `mean`
  - 逐层平均
- `norm_mean`
  - 先把每个 vector 做单位范数归一化，再逐层平均
- `weighted_sum`
  - 按给定权重做逐层加权和

当前实现方式是：

- 不预先把组合后的向量写回磁盘
- 而是在评测时在线读取多个单任务 vector，并在内存里完成逐层合成

如果传入多个 `CAA_VECTOR_DIRS`，但没有额外指定组合方式，建议显式设定：

- `CAA_COMPOSITION=mean`
  - 对应最基础的 equal-weight naive baseline
- 或 `CAA_COMPOSITION=norm_mean`
  - 对应“先 normalize 再 average”的更稳妥版本

### 组合评测示例

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIRS="baseline/CAA_0/artifacts/vectors/llama3-8b/truth/default baseline/CAA_0/artifacts/vectors/llama3-8b/ethics/default baseline/CAA_0/artifacts/vectors/llama3-8b/bias/default" \
CAA_COMPOSITION=mean \
CAA_LAYERS="14 15 16" \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/evaluate_caa_task.sh
```

如果想手动指定权重：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIRS="dir_a dir_b dir_c" \
CAA_COMPOSITION=weighted_sum \
CAA_WEIGHTS="0.5 0.3 0.2" \
CAA_LAYERS="14 15 16" \
bash baseline/CAA_0/evaluate_caa_task.sh
```

如果要把 naive composition 的评测输出也统一放到 `CAA_0/runs/eval`，推荐改用：

```bash
TASK=truth \
BASE_MODEL=models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
CAA_VECTOR_DIRS="baseline/CAA_0/artifacts/vectors/llama3-8b/truth/truth_l14_16 baseline/CAA_0/artifacts/vectors/llama3-8b/ethics/ethics_l14_16 baseline/CAA_0/artifacts/vectors/llama3-8b/bias/bias_l14_16" \
CAA_COMPOSITION=mean \
CAA_LAYERS="14 15 16" \
CAA_ALPHA=1.0 \
bash baseline/CAA_0/run_eval_experiment.sh
```

## 推理时的 steering 方式

当前推理时默认采用：

- 在指定层注册 forward hook
- 将 steering vector 加到当前 forward 的隐藏状态上
- 默认 `token_strategy=last`
  - 只改最后一个 token 的隐藏状态
- 可选 `token_strategy=all`
  - 对当前序列所有 token 都加相同向量

当前默认更推荐：

- `last`

因为它更接近原始 CAA / activation steering 的常见使用方式，也更稳。

## 评测数据与评分逻辑

这里最重要的一点是：

`CAA_0` 没有重新发明一套评测流程，而是直接复用你们项目当前的 ReFT 评测入口，因此测试集和指标保持一致。

底层复用的脚本是：

- `multi_train/eval_truth/evaluate_truth.py`
  - `truth` 任务评测 `truthfulqa_mc`
  - `bias` 任务也复用这个脚本，但数据集切到 `bbq`
- `multi_train/eval_ethics/machine_ethics_exp.py`
- `multi_train/eval_toxicity/toxicity_exp.py`

这样做的好处是：

1. `CAA` 和 `ReFT` 在相同测试集上比较
2. `CAA` 和 `ReFT` 使用同一套输出格式
3. `CAA` 和 `ReFT` 使用同一套 summary 指标
4. 不会因为重新写评测器而引入额外不公平因素

## 推荐实验组织方式

### A. Single-objective

先分别训练：

- `CAA-truth`
- `CAA-bias`
- `CAA-ethics`
- `CAA-toxicity`

然后做两类评测：

1. 在对应任务测试集上评测本任务表现
2. 拿去其他任务测试集上测 transfer / interference

### B. Naive composition

从上面训练好的单任务 vector 中，选出要组合的几个：

- `truth + ethics`
- `truth + bias + ethics`
- `truth + bias + ethics + toxicity`

然后分别用：

- `mean`
- `norm_mean`
- `weighted_sum`

去跑相同测试集。

这部分 baseline 的意义是直接回答：

- 简单线性组合能不能替代你们的方法
- 如果不能，问题是尺度不一致、方向冲突，还是目标互扰

## 与原始 CAA 仓库的区别

这套 `CAA_0` 不是原版仓库逐文件照搬，而是项目化裁剪版。主要区别有：

1. 只保留本项目需要的训练和推理主链路
2. 输入格式适配了你们现有四个任务的数据
3. 评测直接接到你们原有 ReFT 测试脚本
4. 新增了 naive composition 支持
5. 没有迁移原版里和当前实验无关的 notebook、demo、可视化脚本

## 当前限制

当前 `CAA_0` 已经覆盖：

- 单目标 baseline
- naive composition baseline
- 与 ReFT 一致的四任务评测链路

但它仍然是项目实验版，不是一个通用 baseline 平台。当前还没做的主要是：

- 扫所有模型 / 所有层 / 所有组合的批量提交脚本
- 自动搜索最优组合权重
- 更复杂的分层异构组合策略
- `ITI / RepE` 的对应 baseline 目录

## 一句话总结

现在这套 `CAA_0` 可以支持你要的两类实验：

1. `single-objective CAA`
2. `naive composed CAA`

并且二者都在你们项目原有四个测试任务、原有评分逻辑下运行，因此可以直接拿来和你们自己的组合方法做公平对比。
