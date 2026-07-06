---
title: CAA Baseline 数据构建方案
status: 方案设计
language: zh-CN
scope:
  - truthful
  - ethics
  - bias
  - toxicity
sources:
  - baseline/CAA
  - baseline/honest_llama
  - multi_train/train.py
  - multi_train/train_composable_router.py
  - distill_data/code/distill_data_stereotype.py
  - distill_data/code/distill_data_toxicity.py
---

## 一、目标

为 `multi-reft` 新增一个可与 ReFT 直接对比的 `CAA` baseline 数据方案。

这里的目标不是沿用你们现在的 SFT 三字段格式，而是构造成 `CAA` 仓库需要的**对比式回答对**格式：

```json
{
  "question": "...",
  "answer_matching_behavior": "...",
  "answer_not_matching_behavior": "..."
}
```

其中：

- `question`：模型看到的同一个问题 / 指令
- `answer_matching_behavior`：符合目标行为的回答
- `answer_not_matching_behavior`：不符合目标行为的回答

这是 `CAA` 的核心假设。`generate_vectors.py` 会对**同一个问题**分别拼接正回答和负回答，提取激活差，然后取平均作为 steering vector。

所以最重要的一点是：

- **负例必须是“模型回答”**
- **不能直接把用户的 instruction 当成负例回答**

这对 `bias` 和 `toxicity` 尤其关键。

## 二、CAA 对数据的真实要求

从 `baseline/CAA` 代码看，数据要求非常具体：

### 1. 训练 / 向量构建数据

来源：

- [generate_vectors.py](/mnt/hwfile/shichaojian/multi-reft/baseline/CAA/generate_vectors.py)
- [process_raw_datasets.py](/mnt/hwfile/shichaojian/multi-reft/baseline/CAA/process_raw_datasets.py)

要求：

- 每条样本必须有：
  - `question`
  - `answer_matching_behavior`
  - `answer_not_matching_behavior`
- 正负回答都应该是**短而明确的 completion**
- 同一个 `question` 下，正负回答最好只在“行为方向”上相反，不要同时改变太多别的属性

### 2. 测试数据

`CAA` 自带两种：

- `ab`：多选式或二选一式评测
- `open_ended`：开放生成评测

对你们当前项目来说，第一阶段不必完全照搬它原来的 test data，只需要先把**训练侧的 CAA 数据**构对。

## 三、与当前四个子任务的适配结论

### 结论先说

- `truthful`：**可以直接从现有训练数据改造**
- `ethics`：**可以直接从现有训练数据改造**
- `bias`：**可以直接用现有 `instruction + output` 做 question/正例，但必须补负回答**
- `toxicity`：**可以直接用现有 `instruction + output` 做 question/正例，但必须补负回答**

核心原因不是数据量，而是语义结构。

## 四、truthful 的构造方案

### 4.1 当前现有数据情况

当前训练使用：

- `dataset/alignment_truthful_format`

`dataset_info` 表明它是三字段：

- `instruction`
- `input`
- `output`

并且从真实训练样本和训练代码可以确认，这个任务本质上是：

- 多选 commonsense / truthful 风格题
- `instruction` 里通常已经包含：
  - 问题
  - `Answer1/Answer2/...`
  - `Answer format: answer1/answer2/...`
- `output` 常见风格是：
  - `the correct answer is answer2`
  - 或至少能还原出正确选项

训练代码里也明确做了：

- truthful trigger：`the correct answer is `

### 4.2 推荐转换规则

对每一条样本，构造成：

- `question = instruction`
- `answer_matching_behavior = output`
  - 若 `output` 没有前缀，则补成 `the correct answer is answerX`
- `answer_not_matching_behavior = the correct answer is answerY`
  - 其中 `answerY` 是同题的一个错误选项

### 4.3 负例怎么来

推荐优先级：

1. 从同题其它候选答案中随机选一个错误选项
2. 若有 `answer` 字段，则避开正确答案后随机采一个错误答案
3. 若题目无法解析出候选项，则直接丢弃

### 4.4 为什么这样做是合理的

因为这正好符合 `CAA` 的设定：

- 问题完全一样
- 正负回答只差“是否给出正确答案”

这个任务不需要额外找新数据。

### 4.5 建议过滤

- 只保留能稳定解析出 `Answer1...AnswerN` 结构的样本
- 去掉没有显式答案格式的样本
- 去掉 `output` 为空或不合法的样本

## 五、ethics 的构造方案

### 5.1 当前现有数据情况

当前训练实际使用的是：

- `dataset/alignment_moral_cls`

不是 `alignment_moral_augmented`。

`alignment_moral_cls` 的元数据表明它有：

- `instruction`
- `input`
- `output`

真实训练样本可见其风格大致是：

- `instruction`: `Given the following sentence. Evaluate the action with wrong or not wrong from the perspective of moral values.`
- `input`: 某个行为描述
- `output`: `wrong` 或 `not wrong`

训练代码会再统一格式化为：

- ethics trigger：`the action is `

### 5.2 推荐转换规则

对每一条样本，构造成：

- `question = instruction + "\n\n" + input`
- `answer_matching_behavior = the action is <gold_label>`
- `answer_not_matching_behavior = the action is <flipped_label>`

其中：

- `gold_label = output`
- `flipped_label`：
  - `wrong -> not wrong`
  - `not wrong -> wrong`

### 5.3 为什么 ethics 可以直接改造

因为这个任务天然是二分类判断：

- 同一行为描述
- 一个回答符合伦理判断
- 一个回答不符合伦理判断

这和 `CAA` 的正负对要求高度一致。

### 5.4 建议过滤

- 只保留 `output in {"wrong", "not wrong"}` 的样本
- 去掉 `instruction` 或 `input` 为空的样本
- 如存在更复杂标签文本，先标准化再使用

### 5.5 关于 `alignment_moral_augmented`

它可以作为后续扩充，但不建议第一版就混入。

原因：

- 第一版 baseline 先保证标签空间清晰
- `moral_cls` 已经足够适合作为二元 CAA 对比数据
- 先避免 augmentation 引入额外风格噪声

## 六、bias 的构造方案

### 6.1 当前现有数据情况

当前训练使用：

- `dataset/alignment_stereotype_format`

我已经直接读取了真实训练集样本，字段语义如下：

- `instruction`: **原始带偏见/刻板印象的文本**
- `output`: 一个“纠偏回答”
- `input`: 基本为空字符串

真实样本形态大致是：

```text
instruction: ... most black kids don't have the best education growing up.
output: It's important to avoid making generalizations about any group of people...
input:
```

所以这里不是“模板 instruction + 样本 input”的结构，而是：

- `instruction` 本身就是样本正文
- `output` 是针对这条偏见文本的纠偏回答

这说明现有数据已经天然提供了：

- 问题
- 一个好的纠偏回答

但仍然**没有天然的坏回答**。

### 6.2 关键判断：能不能把 instruction 当负例回答

不能直接这样做。

这里要分清两件事：

- `instruction` 可以直接当 `question`
- `instruction` 不能直接当 `answer_not_matching_behavior`

原因是 `CAA` 的 `answer_not_matching_behavior` 在代码里会被当成：

- **同一个问题下的一条模型回答**

而不是用户原始输入。

如果你把当前这条带偏见的 `instruction` 直接塞到 `answer_not_matching_behavior`：

- 语义角色错了
- 模型看到的会变成“偏见文本 + 偏见文本”
- 这不是“坏回答”，而是“重复问题/重复输入”
- 学到的激活差会混入 prompt 结构噪声，而不是回答行为差异

这会把 baseline 搞偏。

### 6.3 正确做法

`bias` 需要额外补一份**不纠偏 / 延续刻板印象 / 接受偏见前提**的负回答。

推荐构造成：

- `question`: 原始带偏见的 `instruction`
- `answer_matching_behavior`: 现有数据中的纠偏回答
- `answer_not_matching_behavior`: 一条坏回答

坏回答的要求：

- 不要写得特别极端脏
- 但必须体现“接受偏见前提”或“顺着 stereotype 往下说”
- 最好和正例长度接近
- 不要变成另一种任务，例如安全拒绝、元评论、空泛说教

### 6.4 bias 额外需要找什么数据

最理想的是下面两类之一：

#### 方案 A：同题双回答数据

每条样本直接有：

- `question`
- `safe_debiased_answer`
- `biased_answer`

这是最接近 `CAA` 原始设定的。

#### 方案 B：只有偏见 prompt 的数据

如果你只能拿到像 `CrowS-Pairs`、`BBQ` 这种主要提供：

- 问题 / 陈述
- 偏见方向标签

那也可以，但你还需要额外生成或人工构造：

- 一条 `answer_matching_behavior`
- 一条 `answer_not_matching_behavior`

也就是说，**原始数据至少要给你 prompt 和偏见类型标签**，后续再批量补回答。

### 6.5 bias 数据格式建议

如果你去找外部数据，建议最低统一成：

```json
{
  "question": "...",
  "bias_target": "...",
  "bias_type": "...",
  "answer_matching_behavior": "...",
  "answer_not_matching_behavior": "...",
  "source": "..."
}
```

其中：

- `bias_target` 和 `bias_type` 不是 CAA 必需字段
- 但后续做清洗、平衡采样、错误分析会非常有用

### 6.6 对你队长想法的判断

“instruction 是一些有偏见的回答，output 是对这些偏见的纠正”

这个判断是对的。

更准确地说，在当前 `bias` 数据里：

- `instruction` = 原始偏见文本
- `output` = 纠偏后的好回答

所以：

- **正例可以直接用 `output`**
- **question 可以直接用 `instruction`**
- **但负例仍然不能直接把 `instruction` 当回答**

## 七、toxicity 的构造方案

### 7.1 当前现有数据情况

当前训练使用：

- `dataset/alignment_toxic_format`

我已经直接读取了真实训练集样本，字段语义如下：

- `instruction`: **原始 toxic / offensive 文本**
- `output`: 去毒化、缓和后的回答
- `input`: 基本为空字符串

真实样本形态大致是：

```text
instruction: haha you guys are a bunch of losers.
output: I'm sorry to hear that you feel that way. Let's try to keep the conversation respectful...
input:
```

所以这里和 `bias` 一样，也不是“模板 instruction + 样本 input”的结构，而是：

- `instruction` 本身就是 toxic 文本
- `output` 是对这条 toxic 文本的去毒化回应

因此现有数据已经天然提供了：

- question
- 一个好的去毒化回答

但没有成对的坏回答。

### 7.2 关键判断：能不能把 instruction 当负例回答

同样不能直接这样做。

这里也要分清两件事：

- `instruction` 可以直接当 `question`
- `instruction` 不能直接当 `answer_not_matching_behavior`

原因完全一样：

- 当前 toxic `instruction` 是用户输入
- `CAA` 负例位置需要的是模型回答

### 7.3 正确做法

`toxicity` 需要补一份真正的 toxic / aggressive / offensive 回复作为负回答。

推荐构造成：

- `question`: 原始 toxic `instruction`
- `answer_matching_behavior`: 现有 detoxified / respectful answer
- `answer_not_matching_behavior`: 一条 toxic continuation / toxic reply

### 7.4 toxicity 额外需要找什么数据

最理想的是下面两类之一：

#### 方案 A：同题双回答数据

每条样本直接有：

- `question`
- `detoxified_answer`
- `toxic_answer`

#### 方案 B：只有 toxic prompt / toxic utterance 的数据

如果你拿到的是：

- toxic 用户发言
- 或 toxic instruction

那么你还需要额外补：

- 一条合格的去毒化回答
- 一条坏回答

也就是说，和 `bias` 一样，**只有 prompt 本身还不够**。

### 7.5 toxicity 数据格式建议

建议至少统一成：

```json
{
  "question": "...",
  "toxicity_type": "...",
  "severity": "...",
  "answer_matching_behavior": "...",
  "answer_not_matching_behavior": "...",
  "source": "..."
}
```

其中：

- `toxicity_type` 可选，例如 insult / hate / threat / profanity
- `severity` 可选，但很有用

## 八、对 bias / toxicity 的具体补数建议

### 8.1 最省事但相对可控的做法

对你们现有 `alignment_stereotype_format` 和 `alignment_toxic_format`：

1. 保留原来的 `instruction` 作为 `question`
2. 保留原来的 `output` 作为 `answer_matching_behavior`
3. 额外批量生成一条 `answer_not_matching_behavior`

生成负回答时要有专门 prompt，要求：

- 站在错误前提上继续回答
- 保持像普通 assistant completion
- 不要生成免责声明
- 不要输出空白
- 不要只复述问题

### 8.2 比较稳的做法

不要完全依赖生成负例。

推荐组合：

- 一部分用外部真实坏回答数据
- 一部分用模型生成坏回答

这样可以减少：

- 模板化负例
- 风格过于统一
- “一眼就是机器凑出来的坏回答”

## 九、推荐的第一版落地方案

### 9.1 truthful

直接从现有训练集转：

- `question = instruction`
- `answer_matching_behavior = gold output`
- `answer_not_matching_behavior = 随机错误选项`

### 9.2 ethics

直接从 `alignment_moral_cls` 转：

- `question = instruction + input`
- `answer_matching_behavior = the action is gold_label`
- `answer_not_matching_behavior = the action is flipped_label`

### 9.3 bias

不要直接用当前三字段硬转。

第一版应该：

- 用当前 `alignment_stereotype_format` 里的 `instruction` 直接做 `question`
- 用当前 `output` 做正例
- 新补一条坏回答

### 9.4 toxicity

不要直接用当前三字段硬转。

第一版应该：

- 用当前 `alignment_toxic_format` 里的 `instruction` 直接做 `question`
- 用当前 `output` 做正例
- 新补一条坏回答

## 十、外部新数据你需要找成什么格式

如果你去找 `bias` / `toxicity` 外部数据，我建议你尽量找下面两种之一：

### 最优格式

```json
{
  "question": "...",
  "answer_matching_behavior": "...",
  "answer_not_matching_behavior": "..."
}
```

这是可以直接喂给 `CAA` 的。

### 次优格式

```json
{
  "question": "...",
  "label": "...",
  "meta": "..."
}
```

这种也能用，但你必须再补两条回答中的至少一条。

### 不建议只找这种

```json
{
  "toxic_text": "...",
  "stereotype_text": "..."
}
```

因为它只有有害文本，没有回答对，离 `CAA` 的目标格式还差一步。

## 十一、最终建议

### 可以直接开始做的

- `truthful`
- `ethics`

这两个任务都可以直接从现有 ReFT 训练数据迁移成 `CAA` 数据。

### 不要偷懒直接硬转的

- `bias`
- `toxicity`

这两个任务里：

- 现有数据的 `output` 可以直接拿来当正例
- 现有数据的 `instruction` 可以直接拿来当 `question`
- 但 `instruction` 不能直接当负例回答

### 你接下来真正要补的东西

对 `bias` 和 `toxicity`，你要补的是：

- **和同一个问题配对的坏回答**

而不是再去找更多单独的坏问题。

---

如果下一步开始实现，我建议顺序是：

1. 先写 `truthful` 的转换脚本
2. 再写 `ethics` 的转换脚本
3. 然后单独设计 `bias/toxicity` 的负回答生成 prompt 和质检规则

这样不会把最容易做对的两项和最麻烦的两项混在一起。
