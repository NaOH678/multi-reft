# Composable LoReFT Usage Guide

## Scope

这份文档说明当前已经实现的组合推理如何使用，重点覆盖：

- 当前支持的 CLI 入口
- 需要传入哪些参数
- 当前实现的约束
- 当前应该优先验证哪些问题
- 一套建议的实验表格
- 对应的命令行脚本模板

## Current Supported Entry

当前已经接入组合推理的评测入口只有：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`

也就是说：

- 目前可以先在 `eval_truth` 上跑 composable inference
- `eval_bias / eval_ethics / eval_toxicity` 还没有接上新的 composable loader

## Current Inference Modes

当前 `eval_truth/evaluate_truth.py` 支持三种模型加载方式：

1. 单个 LoReFT
- `--reft_weights`

2. 多 specialist 组合推理
- `--reft_specialists ...`

3. LoRA
- `--lora_weights`

组合推理走的是第 2 种。

## Current Constraints

使用组合推理前需要注意：

1. 只能在 `eval_truth` 上直接跑

2. `--reft_specialists` 里的多个 checkpoint 必须层级 shape 一致
- 同层 `rotate_weight` 必须同 shape
- 同层 `source_weight` 必须同 shape
- 同层 `source_bias` 必须同 shape

这在实践里通常等价于：

- 这些 specialists 的 LoReFT rank 要一致

例如当前仓库里已知：

- `Llama3-8b-Loreft_truthful_4` 是 rank `8`
- `Llama3-8b-Loreft_moral` 是 rank `8`
- `Llama3-8b-Loreft_stereotype` 是 rank `16`
- `Llama3-8b-Loreft_combined` 是 rank `32`

所以：

- `truthful_4 + moral` 可以直接组合
- `truthful_4 + stereotype` 目前不行

3. 若使用：
- `--compose_domain shared_latent`
- `--transport_type identity`

则要求：

- `--shared_basis_rank == specialist rank`

否则会报错。

## CLI Parameters

### Required Parameters

这些参数是最基础的：

- `--dataset`
- `--base_model`
- `--batch_size`

示例：

```bash
--dataset truthfulqa_mc \
--base_model /path/to/base/model \
--batch_size 8
```

### Standard ReFT Inference Parameters

- `--target_layers`
  - 默认 `[-1]`，表示所有层
- `--subspace_rank`
  - 当前 composable 路径主要用于单 specialist 路径兼容
  - 对 composable 路径本身，真正使用的是 checkpoint 中读出的 rank
- `--positions`
  - prompt intervention 的位置数设置
- `--device`
  - 如 `cuda:0`
- `--greedy_decoding`
  - 是否贪心解码

### Composable Inference Parameters

#### Specialist Inputs

- `--reft_specialists path1 path2 ...`

表示要组合的多个 LoReFT specialist checkpoint。

这里传的是：

- `checkpoint-xxx`
- 或 `checkpoint-xxx/intervenable_model`

都可以，当前 loader 会自动解析。

#### Composition Backend

- `--compose_domain`

当前可选：

- `output`
- `projected_output`
- `shared_latent`

含义：

- `output`
  - 直接在原始表征空间组合各 specialist 的 `Delta`
- `projected_output`
  - 先做 output-space combine，再投影到 shared basis
- `shared_latent`
  - 先在共享 latent 中组合 `delta`，再 lift 回原空间

#### Policy

- `--composition_method`

当前可选：

- `single`
- `equal`
- `residual_softmax`
- `intervention_softmax`
- `topk_residual`
- `compat_filtered_topk`

#### Policy Hyperparameters

- `--composition_temperature`
  - softmax temperature
- `--composition_topk`
  - 用于 `topk_residual` / `compat_filtered_topk`
- `--compat_threshold`
  - 用于 `compat_filtered_topk`
- `--single_index`
  - 用于 `single`

#### Shared-Latent Parameters

- `--shared_basis_type`
  - `orth_mean`
  - `svd_union`

- `--shared_basis_rank`
  - 共享 basis 的 rank

- `--transport_type`
  - `identity`
  - `overlap`

## Minimal Usage Patterns

### 1. 单个 LoReFT Baseline

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset truthfulqa_mc \
  --base_model /path/to/base/model \
  --batch_size 8 \
  --reft_weights /path/to/checkpoint/intervenable_model \
  --target_layers -1 \
  --positions 7 \
  --device cuda:0
```

### 2. Output-Space Equal Mixture

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset truthfulqa_mc \
  --base_model /path/to/base/model \
  --batch_size 8 \
  --reft_specialists \
    /path/to/spec1/intervenable_model \
    /path/to/spec2/intervenable_model \
  --compose_domain output \
  --composition_method equal \
  --target_layers -1 \
  --positions 7 \
  --device cuda:0
```

### 3. Output-Space Residual Softmax

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset truthfulqa_mc \
  --base_model /path/to/base/model \
  --batch_size 8 \
  --reft_specialists \
    /path/to/spec1/intervenable_model \
    /path/to/spec2/intervenable_model \
  --compose_domain output \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --target_layers -1 \
  --positions 7 \
  --device cuda:0
```

### 4. Output-Space Top-k Residual

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset truthfulqa_mc \
  --base_model /path/to/base/model \
  --batch_size 8 \
  --reft_specialists \
    /path/to/spec1/intervenable_model \
    /path/to/spec2/intervenable_model \
  --compose_domain output \
  --composition_method topk_residual \
  --composition_topk 1 \
  --composition_temperature 1.0 \
  --target_layers -1 \
  --positions 7 \
  --device cuda:0
```

### 5. Shared-Latent With `svd_union + overlap`

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset truthfulqa_mc \
  --base_model /path/to/base/model \
  --batch_size 8 \
  --reft_specialists \
    /path/to/spec1/intervenable_model \
    /path/to/spec2/intervenable_model \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank 8 \
  --transport_type overlap \
  --target_layers -1 \
  --positions 7 \
  --device cuda:0
```

## Recommended Current Validation Goals

当前阶段最应该优先验证的是下面这些问题。

### A. 组合信号本身是否有用

先比较：

- `single`
- `equal`
- `residual_softmax`
- `topk_residual`

要回答的是：

- 多 specialist 组合是否比单 specialist 有收益
- rule-based arbitration 是否比 naive equal mixture 更靠谱

### B. output-space 是否已经足够

先在 `output` 分支下建立稳固 baseline。

要回答的是：

- 原空间组合是否已经能带来有意义收益
- 如果 output-space 已经足够强，shared-latent 就不是最高优先级

### C. shared-latent 是否真的有信号

再比较：

- `output`
- `projected_output`
- `shared_latent`

要回答的是：

- 问题是否真的出在“组合发生得太晚”
- 还是 output-space mixture 已经足够

### D. basis 和 transport 哪个更关键

在 `shared_latent` 下重点比较：

- `orth_mean + identity`
- `svd_union + identity`
- `svd_union + overlap`

要回答的是：

- shared-latent 的改善是否主要来自更好的 basis
- 还是主要来自更好的 latent transport

## Current Recommended Experiment Matrix

下面这组实验是当前最值得跑的第一批。

说明：

- 默认假设你选用的是两个 rank 相同的 specialists
- 例如：
  - `Llama3-8b-Loreft_truthful_4/checkpoint-666/intervenable_model`
  - `Llama3-8b-Loreft_moral/checkpoint-55/intervenable_model`

| Exp ID | Goal | compose_domain | composition_method | shared_basis_type | transport_type | Extra Params |
|---|---|---|---|---|---|---|
| E1 | 单 specialist baseline | output | single | - | - | `--single_index 0` |
| E2 | naive mixture baseline | output | equal | - | - | - |
| E3 | DII-native rule | output | residual_softmax | - | - | `--composition_temperature 1.0` |
| E4 | sparse arbitration | output | topk_residual | - | - | `--composition_topk 1 --composition_temperature 1.0` |
| E5 | 共享子空间最粗 baseline | shared_latent | equal | orth_mean | identity | `--shared_basis_rank r` |
| E6 | shared-latent + residual score | shared_latent | residual_softmax | orth_mean | identity | `--shared_basis_rank r --composition_temperature 1.0` |
| E7 | 更合理 shared basis | shared_latent | residual_softmax | svd_union | identity | `--shared_basis_rank r --composition_temperature 1.0` |
| E8 | 更合理 latent transport | shared_latent | residual_softmax | svd_union | overlap | `--shared_basis_rank r --composition_temperature 1.0` |
| E9 | output-space 的过渡对照 | projected_output | residual_softmax | svd_union | - | `--shared_basis_rank r --composition_temperature 1.0` |

这里的 `r` 是 specialist 的 rank，比如当前若使用：

- `truthful_4 + moral`

则可以用：

- `--shared_basis_rank 8`

## Script Templates

下面给一组可以直接改路径后运行的脚本模板。

### Shared Variables

```bash
BASE_MODEL=../weightsft/models/llama3-8b/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920
DATASET=truthfulqa_mc
BATCH_SIZE=256
DEVICE=cuda:0

SPEC1=/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_truthful_4/checkpoint-3330/intervenable_model
SPEC2=/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_moral/checkpoint-330/intervenable_model/
SPEC3=/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_stereotype_1/checkpoint-160/intervenable_model/
SPEC4=/mnt/shared-storage-user/zhoujiawei/fusion/multi-reft/multi_train/trainer_output/Llama3-8b-Loreft_toxicity/checkpoint-235/intervenable_model
```

### Script: E1

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4}\
  --compose_domain output \
  --composition_method single \
  --single_index 0 \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E2

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain output \
  --composition_method equal \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E3

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain output \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E4

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain output \
  --composition_method topk_residual \
  --composition_topk 1 \
  --composition_temperature 1.0 \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E5

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain shared_latent \
  --composition_method equal \
  --shared_basis_type orth_mean \
  --shared_basis_rank 8 \
  --transport_type identity \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E6

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type orth_mean \
  --shared_basis_rank 8 \
  --transport_type identity \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E7

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank 8 \
  --transport_type identity \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E8

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain shared_latent \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank 8 \
  --transport_type overlap \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

### Script: E9

```bash
python multi_train/eval_truth/evaluate_truth.py \
  --dataset ${DATASET} \
  --base_model ${BASE_MODEL} \
  --batch_size ${BATCH_SIZE} \
  --reft_specialists ${SPEC1} ${SPEC2} ${SPEC3} ${SPEC4} \
  --compose_domain projected_output \
  --composition_method residual_softmax \
  --composition_temperature 1.0 \
  --shared_basis_type svd_union \
  --shared_basis_rank 8 \
  --target_layers -1 \
  --positions 7 \
  --device ${DEVICE}
```

## What Should Be Verified First

建议先按这个顺序验证。

### Phase 1

先验证 output-space 路线：

- E1
- E2
- E3
- E4

重点看：

- `single` 和 `equal` 的差距
- `residual_softmax` 是否优于 `equal`
- `topk_residual` 是否优于 `equal`

### Phase 2

再验证 shared-latent 是否有信号：

- E5
- E6
- E7
- E8
- E9

重点看：

- `shared_latent` 是否优于 `output`
- `svd_union` 是否优于 `orth_mean`
- `overlap` 是否优于 `identity`
- `projected_output` 是否介于两者之间

## Current Missing Pieces

使用这套指南时还要记住：

- 当前只在 `eval_truth` 上支持 composable 推理
- 当前还没有为这些实验专门写成独立 `.sh` 文件
- 当前脚本模板都写在本文档中

后续若继续推进，建议再补：

- `multi_train/script/evaluate_composable_truth.sh`
- `multi_train/script/evaluate_composable_ethics.sh`
- `multi_train/script/evaluate_composable_bias.sh`
- `multi_train/script/evaluate_composable_toxicity.sh`

但当前阶段，先用本文档里的命令模板即可。
