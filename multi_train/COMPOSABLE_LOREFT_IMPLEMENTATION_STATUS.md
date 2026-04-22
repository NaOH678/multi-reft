# Composable LoReFT Implementation Status

## Purpose

这份文档记录当前已经完成的实现、尚未完成的部分、改动过的文件以及已做过的验证，作为后续继续开发时的保障。

当前实现对应的目标是：

- 为多个独立 LoReFT specialist 提供统一的 token-local 组合框架
- 先打通最小可行路径：
  - 新 intervention
  - 公共 loader
  - `eval_truth` 接入
- 为后续 `eval_bias / eval_ethics / eval_toxicity` 和第二阶段训练预留接口

## What Has Been Implemented

### 1. `ComposableLoreftIntervention` 已实现

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/interventions.py`

已实现内容：

- 新增 `ComposableLoreftIntervention`
- 保持 token-local 语义
- 前向拆成：
  - 计算每个 specialist 的 `delta`
  - 计算每个 specialist 的 `Delta`
  - 提取 policy feature
  - 根据 `policy_type` 计算 `alpha`
  - 根据 `compose_domain` 组合输出

当前支持的 `compose_domain`：

- `output`
- `projected_output`
- `shared_latent`

当前支持的 `policy_type`：

- `single`
- `equal`
- `residual_softmax`
- `intervention_softmax`
- `topk_residual`
- `compat_filtered_topk`
- `trainable` 的接口骨架已预留

当前支持的辅助能力：

- `freeze_specialists()`
- `freeze_policy()`
- `unfreeze_policy()`
- debug tensor 缓存：
  - `latest_alpha`
  - `latest_scores`
  - `latest_delta_norm`
  - `latest_intervention_norm`
  - `latest_pairwise_cos`
  - `latest_selected_mask`
  - `latest_conflict_score`
  - `latest_compose_domain`
  - `latest_shared_basis_stats`
  - `latest_transport_stats`

### 2. `pyreft` 已导出新 intervention

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/__init__.py`

已实现内容：

- 导出 `ComposableLoreftIntervention`

### 3. 公共 composable loader 已实现

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`

已实现内容：

- 解析多个 Loreft checkpoint 的 `intervenable_model`
- 直接读取每层：
  - `weight`
  - `bias`
  - `rotate_layer`
- 构造 stacked specialist 参数
- 构造 shared basis
- 构造 transport
- 构造 `ReftConfig`
- 返回 composable reft model

当前已实现函数：

- `_resolve_intervenable_dir`
- `normalize_specialist_label`
- `build_composable_model_tag`
- `load_loreft_specialist_state`
- `build_shared_basis_for_layer`
- `build_transport_for_layer`
- `_collect_target_layers`
- `_validate_layer_shapes`
- `build_composable_reft_config`
- `load_composed_reft_model`

当前支持的 shared basis：

- `orth_mean`
- `svd_union`

当前支持的 transport：

- `identity`
- `overlap`

### 4. `eval_truth` 已接入 composable 路线

文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`

已实现内容：

- 新增 `--reft_specialists`
- 新增 `--compose_domain`
- 新增 `--composition_method`
- 新增：
  - `--composition_temperature`
  - `--composition_topk`
  - `--compat_threshold`
  - `--single_index`
  - `--shared_basis_type`
  - `--shared_basis_rank`
  - `--transport_type`
- 支持三种加载路径：
  - `--reft_specialists`：走 composable loader
  - `--reft_weights`：走单个 Loreft
  - `--lora_weights`：走 LoRA
- summary 中已记录 composable 相关字段
- 输出文件命名对 composable 模式单独加了 `build_composable_model_tag`

### 5. import fallback 已处理

涉及文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`

已实现内容：

- 兼容当前仓库里的 `pyreft` namespace/package 结构
- 当 `from pyreft import ...` 失败时，fallback 到 `from pyreft.pyreft import ...`

这是必要的，因为当前环境中顶层 `pyreft` 是 namespace package。

## What Has Been Verified

### 1. Python 语法编译通过

已检查文件：

- `pyreft/pyreft/interventions.py`
- `pyreft/pyreft/__init__.py`
- `multi_train/eval_common/composable_loreft.py`
- `multi_train/eval_truth/evaluate_truth.py`

### 2. `ComposableLoreftIntervention` 前向 shape 已验证

已验证：

- `compose_domain=output`
- `compose_domain=projected_output`
- `compose_domain=shared_latent`

均能对合成输入跑通，并得到正确 shape：

- 输出 `out: [B, S, d]`
- `latest_alpha: [B, S, T]`

### 3. 真实 Loreft checkpoint 读取已验证

已确认能够从真实 checkpoint 的 `intervenable_model` 中读到：

- `weight`
- `bias`
- `rotate_layer`

已确认真实文件格式与 `LoreftIntervention.state_dict()` 一致。

### 4. shared basis / transport 构造已做真实 checkpoint 验证

使用真实 Loreft checkpoint 验证过：

- `build_shared_basis_for_layer(..., basis_type="svd_union")`
- `build_transport_for_layer(..., transport_type="overlap")`

能够返回正确 shape 的：

- `shared_basis`
- `transport_weight`

### 5. rank 不一致情况已显式防御

真实 checkpoint 检查中发现不同实验的 LoReFT rank 可能不同，例如：

- `truthful_4`: rank 8
- `stereotype`: rank 16
- `combined`: rank 32

因此 loader 已加入 `_validate_layer_shapes`：

- 如果要组合的 specialists 在同一层 shape 不一致，会直接报错
- 避免后续在 stack / transport 时 silent failure

## What Has NOT Been Implemented Yet

### 1. 其余三个 eval 入口还没有接入

尚未接入 composable loader 的文件：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_bias/stereotype_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_ethics/machine_ethics_exp.py`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_toxicity/toxicity_exp.py`

### 2. `eval_toxicity` 还没统一回 Loreft 体系

当前 `eval_toxicity/toxicity_exp.py` 还是按 `NodireftIntervention` 在加载。

接入 composable 之前，应先统一到 Loreft 口径。

### 3. `train_composer.py` 还没实现

文件尚未创建：

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/train_composer.py`

当前仅在 `ComposableLoreftIntervention` 内预留了：

- `trainable policy` 的结构入口
- freeze / unfreeze 方法

但还没有真正训练脚本。

### 4. metadata 独立保存还没实现

当前：

- `ComposableLoreftIntervention.state_dict()` 只保存张量参数
- 非张量 metadata 没有单独落盘

后续应补：

- 组合配置 json
- specialist 名称列表
- policy / compose_domain / basis / transport 配置

### 5. `compat_filtered_topk` 只是第一版 greedy 实现

当前实现：

- 先按 residual 取 top-k
- 再按 pairwise cosine greedy filter

尚未实现：

- 更强的 active-set 搜索
- latent-side compatibility
- 更精细的 conflict-aware 打分

### 6. shared-latent 的实现仍是第一版

当前支持：

- `orth_mean`
- `svd_union`
- `identity`
- `overlap`

但尚未实现：

- learned transport
- learned shared basis
- 更严格的 latent alignment 机制

## Files Changed So Far

### Modified

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/interventions.py`
  - 新增 `ComposableLoreftIntervention`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/pyreft/pyreft/__init__.py`
  - 导出新类
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_truth/evaluate_truth.py`
  - 接入 composable 路线

### Added

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/eval_common/composable_loreft.py`
  - 公共 loader

### Existing Planning / Design Docs

- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/COMPOSABLE_LOREFT_PLAN.md`
- `/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft/multi_train/COMPOSABLE_LOREFT_CODE_DESIGN.md`

## Important Current Constraints

### 1. 组合的 specialists 需要层级 shape 一致

当前实现要求：

- 每层 `rotate_weight` shape 一致
- 每层 `source_weight` shape 一致
- 每层 `source_bias` shape 一致

否则 loader 会直接报错。

这意味着：

- 第一版实验应优先组合 rank 一致的 specialists

### 2. `shared_latent + identity transport` 需要 `shared_basis_rank == specialist rank`

因为 `identity` transport 不做维度变化，所以要求：

- `k == r`

否则 `_compose_shared_latent` 会报错。

### 3. 当前只在 `eval_truth` 上接通

所以目前可视为：

- “核心组合模块已实现”
- “一个 eval 入口已完成接线”

还不能算四个任务的完整实现。

## Recommended Next Steps

建议后续按下面顺序继续：

1. 接入 `eval_ethics`
2. 接入 `eval_bias`
3. 统一 `eval_toxicity` 到 Loreft，再接入 composable loader
4. 抽取 composable metadata 持久化
5. 实现 `train_composer.py`

## Minimal Working State Summary

当前已经具备的最小能力是：

- 读取多个 rank 一致的 Loreft specialist checkpoint
- 构造 composable reft intervention
- 在 `eval_truth` 中以 `output / projected_output / shared_latent` 三种组合后端运行
- 使用 token-local 的：
  - `single`
  - `equal`
  - `residual_softmax`
  - `intervention_softmax`
  - `topk_residual`
  - `compat_filtered_topk`

这就是当前实现的可用边界。
