import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import torch
import transformers
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, interleave_datasets, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainerCallback, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from pyreft import ReftDataCollator, ReftSupervisedDataset, ReftTrainerForCausalLMDistributed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from multi_train.eval_common.composable_loreft import (
    load_mixed_composed_reft_model,
    normalize_specialist_label,
)


DEFAULT_ROUTER_FEATURE_NAMES = [
    "rh_direction",
    "delta_direction",
    "mean_compat",
    "neg_compat_mass",
]

DEFAULT_TASK_SAMPLING = {
    "toxicity": 0.35,
    "ethics": 0.25,
    "truthfulqa": 0.20,
    "bbq": 0.20,
}

TRUTHFUL_TRIGGER = "the correct answer is "
ETHICS_TRIGGER = "the action is "


@dataclass
class RouterModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model path"})
    reft_specialists: List[str] = field(default_factory=list, metadata={"help": "Specialist intervention dirs"})
    compose_domain: str = field(default="output", metadata={"help": "Composable domain"})
    target_layers: List[int] = field(default_factory=lambda: [-1], metadata={"help": "Target layers"})
    router_layers: List[int] = field(default_factory=lambda: [28, 29, 30], metadata={"help": "Layers that use trainable router"})
    router_layer_policy: str = field(default="trainable", metadata={"help": "Router layer policy: trainable or no_training"})
    router_feature_names: List[str] = field(default_factory=lambda: list(DEFAULT_ROUTER_FEATURE_NAMES))
    router_feature_stats_path: str = field(default="", metadata={"help": "Optional path to scalar router feature stats json"})
    base_score_stats_path: str = field(default="", metadata={"help": "Path to E6 score stats json"})
    positions: str = field(default="f5+l5", metadata={"help": "Intervention positions string"})
    base_composition_temperature: float = field(default=1.0)
    base_composition_topk: int = field(default=2)
    base_compat_threshold: float = field(default=0.0)
    base_score_source: str = field(default="intervention_norm")
    base_score_normalizer: str = field(default="log_zscore")
    router_temperature: float = field(default=1.5)
    policy_hidden_dim: int = field(default=128)
    policy_projection_dim: int = field(default=128)
    use_pre_hidden_state_feature: bool = field(default=True)
    pre_hidden_state_dim: int = field(default=64)
    router_feature_clip: float = field(default=5.0)
    router_feature_eps: float = field(default=1e-6)
    compat_impl: str = field(default="optimized")


@dataclass
class RouterDataArguments:
    task_dataset_map_json: str = field(
        metadata={
            "help": "JSON file or inline JSON mapping task -> dataset spec. Spec supports path, split, format, field_map."
        }
    )
    sampling_probs_json: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON file or inline JSON mapping task -> sampling probability."},
    )
    task_train_quota_json: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON file or inline JSON mapping task -> post-holdout train quota."},
    )
    max_train_samples_per_task: Optional[int] = field(default=None)
    max_eval_samples_per_task: int = field(default=128)
    eval_holdout_ratio: float = field(default=0.02)
    bbq_qa_bridge_count: int = field(default=0)
    bbq_qa_bridge_sampling_prob: Optional[float] = field(default=None)


@dataclass
class RouterTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=512)
    dropout: float = field(default=0.0)
    sanity_eval_only: bool = field(default=False)
    sanity_eval_splits: str = field(default="both")
    router_kl_weight: float = field(default=0.0)
    router_kl_target_prob: float = field(default=0.8)
    bias_router_kl_stereotype_prob: float = field(default=0.6)
    bias_router_kl_truth_prob: float = field(default=0.25)
    bbq_qa_bridge_kl_truth_prob: float = field(default=0.45)
    bbq_qa_bridge_kl_stereotype_prob: float = field(default=0.35)


def _unwrap_intervention(intervention_value):
    if isinstance(intervention_value, (list, tuple)):
        return intervention_value[0]
    return intervention_value


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _load_json_value(path_or_json: str) -> Dict:
    candidate = Path(path_or_json)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path_or_json)


def _normalize_task_name(task_name: str) -> str:
    return str(task_name).strip().lower()


def _format_truthful_output(example: Dict) -> str:
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(TRUTHFUL_TRIGGER):
        output = f"{TRUTHFUL_TRIGGER}{answer}"
    return output


def _format_ethics_output(example: Dict) -> str:
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(ETHICS_TRIGGER):
        output = f"{ETHICS_TRIGGER}{answer}"
    return output


def _truncate_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    sliced = dataset[: max_samples]
    return Dataset.from_dict(sliced, features=dataset.features)


def _slice_dataset_by_indices(dataset: Dataset, indices: List[int]) -> Dataset:
    if not indices:
        empty_columns = {name: [] for name in dataset.column_names}
        return Dataset.from_dict(empty_columns, features=dataset.features)
    sliced = dataset[indices]
    return Dataset.from_dict(sliced, features=dataset.features)


def _shuffle_dataset(dataset: Dataset, seed: int) -> Dataset:
    if len(dataset) <= 1:
        return dataset
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return _slice_dataset_by_indices(dataset, indices)


def _auto_load_dataset(path: str, split: Optional[str]) -> Dataset:
    dataset_path = Path(path)
    if dataset_path.exists() and dataset_path.is_dir():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            chosen_split = split or "train"
            return loaded[chosen_split]
        return loaded

    suffix = dataset_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        loaded = load_dataset("json", data_files=str(dataset_path))
        return loaded[split or "train"]
    if suffix == ".csv":
        loaded = load_dataset("csv", data_files=str(dataset_path))
        return loaded[split or "train"]
    raise ValueError(f"Cannot auto-load dataset from path: {path}")


def _load_task_dataset(task_name: str, spec: Dict, max_samples: Optional[int], seed: int) -> Dataset:
    if isinstance(spec, str):
        spec = {"path": spec}
    if "path" not in spec:
        raise ValueError(f"Dataset spec for task={task_name} is missing `path`.")

    dataset = _auto_load_dataset(spec["path"], spec.get("split"))
    field_map = spec.get("field_map") or {}
    rename_ops = {}
    for canonical in ["instruction", "input", "output"]:
        src = field_map.get(canonical, canonical)
        if src not in dataset.column_names:
            raise ValueError(
                f"Task `{task_name}` dataset is missing required column `{src}` "
                f"(mapped from canonical `{canonical}`). Available columns: {dataset.column_names}"
            )
        if src != canonical:
            rename_ops[src] = canonical
    if rename_ops:
        dataset = dataset.rename_columns(rename_ops)

    normalized_task_name = _normalize_task_name(task_name)
    if normalized_task_name in {"truthful", "truthfulqa"}:
        dataset = dataset.map(lambda example: {"output": _format_truthful_output(example)})
    elif normalized_task_name in {"moral", "ethics"}:
        dataset = dataset.map(lambda example: {"output": _format_ethics_output(example)})

    keep_columns = ["instruction", "input", "output"]
    extra_columns = [col for col in dataset.column_names if col not in keep_columns]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    dataset = _shuffle_dataset(dataset, seed=seed)
    dataset = _truncate_dataset(dataset, max_samples)
    dataset = dataset.add_column("task_name", [_normalize_task_name(task_name)] * len(dataset))
    return dataset


def _split_train_eval(dataset: Dataset, eval_holdout_ratio: float, max_eval_samples: int, seed: int):
    if len(dataset) <= 1 or eval_holdout_ratio <= 0:
        return dataset, _truncate_dataset(dataset, max_eval_samples)
    eval_size = max(1, int(len(dataset) * eval_holdout_ratio))
    if eval_size >= len(dataset):
        eval_size = max(1, len(dataset) - 1)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_indices = indices[:eval_size]
    train_indices = indices[eval_size:]
    train_dataset = _slice_dataset_by_indices(dataset, train_indices)
    eval_dataset = _slice_dataset_by_indices(dataset, eval_indices)
    eval_dataset = _truncate_dataset(eval_dataset, max_eval_samples)
    return train_dataset, eval_dataset


def _build_sampling_probabilities(task_names: List[str], override_json: Optional[str]) -> List[float]:
    if override_json:
        raw = _load_json_value(override_json)
    else:
        raw = DEFAULT_TASK_SAMPLING
    probs = []
    for task_name in task_names:
        if task_name not in raw:
            raise ValueError(f"Missing sampling probability for task `{task_name}`.")
        probs.append(float(raw[task_name]))
    total = sum(probs)
    if total <= 0:
        raise ValueError("Sampling probabilities must sum to a positive value.")
    return [value / total for value in probs]


def _truncate_eval_for_distributed(dataset: Dataset, per_device_eval_batch_size: int) -> Dataset:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_eval_batch = max(1, world_size * max(1, int(per_device_eval_batch_size)))
    if len(dataset) < global_eval_batch:
        return dataset
    truncated_length = (len(dataset) // global_eval_batch) * global_eval_batch
    if truncated_length == len(dataset):
        return dataset
    return _truncate_dataset(dataset, truncated_length)


def _build_task_train_quotas(task_names: List[str], quota_json: Optional[str]) -> Optional[Dict[str, int]]:
    if not quota_json:
        return None
    raw = _load_json_value(quota_json)
    quotas: Dict[str, int] = {}
    for task_name in task_names:
        if task_name not in raw:
            raise ValueError(f"Missing train quota for task `{task_name}`.")
        quota_value = int(raw[task_name])
        if quota_value <= 0:
            raise ValueError(f"Train quota for task `{task_name}` must be positive, got {quota_value}.")
        quotas[task_name] = quota_value
    return quotas


def _assign_task_name(dataset: Dataset, task_name: str) -> Dataset:
    normalized_task_name = _normalize_task_name(task_name)
    if "task_name" in dataset.column_names:
        dataset = dataset.remove_columns(["task_name"])
    return dataset.add_column("task_name", [normalized_task_name] * len(dataset))


def _build_bbq_qa_bridge_datasets(
    truthful_train_dataset: Dataset,
    truthful_eval_dataset: Dataset,
    bridge_count: int,
    max_eval_samples: int,
) -> tuple[Dataset, Dataset]:
    bridge_train_dataset = _assign_task_name(
        _truncate_dataset(truthful_train_dataset, bridge_count),
        "bbq_qa_bridge",
    )
    bridge_eval_dataset = _assign_task_name(
        _truncate_dataset(truthful_eval_dataset, max_eval_samples),
        "bbq_qa_bridge",
    )
    return bridge_train_dataset, bridge_eval_dataset


def _parse_sanity_eval_splits(raw_value: str) -> List[str]:
    normalized = str(raw_value).strip().lower()
    if normalized == "both":
        return ["train", "eval"]
    if normalized in {"train", "eval"}:
        return [normalized]
    raise ValueError(f"Unsupported sanity eval split setting: {raw_value}")


def _build_model_and_tokenizer(model_args: RouterModelArguments, training_args: RouterTrainingArguments):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # Avoid using EOS as PAD; otherwise EOS tokens may be masked out in attention.
    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    # Keep model/generation configs aligned with tokenizer to avoid repeated warnings.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer


def _freeze_for_router_training(reft_model):
    for param in reft_model.model.parameters():
        param.requires_grad = False
    for intervention in reft_model.interventions.values():
        if hasattr(intervention, "freeze_specialists"):
            intervention.freeze_specialists()
        if getattr(intervention, "policy_type", None) == "trainable":
            intervention.unfreeze_policy()
        else:
            intervention.freeze_policy()


def _build_layer_specs(model_args: RouterModelArguments, dropout: float):
    default_layer_spec = {
        "compose_domain": model_args.compose_domain,
        "composition_method": "compat_filtered_topk",
        "composition_temperature": model_args.base_composition_temperature,
        "composition_topk": model_args.base_composition_topk,
        "compat_threshold": model_args.base_compat_threshold,
        "score_source": model_args.base_score_source,
        "score_normalizer": model_args.base_score_normalizer,
        "score_stats_path": model_args.base_score_stats_path,
        "compat_impl": model_args.compat_impl,
        "dropout": dropout,
        "enable_debug_cache": False,
    }
    if model_args.router_layer_policy == "trainable":
        router_layer_spec = {
            "compose_domain": model_args.compose_domain,
            "composition_method": "trainable",
            "composition_temperature": model_args.router_temperature,
            "score_source": model_args.base_score_source,
            "score_normalizer": "none",
            "score_stats_path": None,
            "use_trainable_policy": True,
            "policy_hidden_dim": model_args.policy_hidden_dim,
            "policy_projection_dim": model_args.policy_projection_dim,
            "use_pre_hidden_state_feature": model_args.use_pre_hidden_state_feature,
            "pre_hidden_state_dim": model_args.pre_hidden_state_dim,
            "router_feature_names": model_args.router_feature_names,
            "router_feature_stats_path": model_args.router_feature_stats_path,
            "router_feature_clip": model_args.router_feature_clip,
            "router_feature_eps": model_args.router_feature_eps,
            "compat_impl": model_args.compat_impl,
            "dropout": dropout,
            "enable_monitor_cache": True,
            "enable_debug_cache": False,
        }
    elif model_args.router_layer_policy == "no_training":
        router_layer_spec = dict(default_layer_spec)
        router_layer_spec.update(
            {
                "enable_monitor_cache": True,
                "enable_debug_cache": False,
            }
        )
    else:
        raise ValueError(f"Unsupported router_layer_policy: {model_args.router_layer_policy}")
    layer_overrides = {int(layer): dict(router_layer_spec) for layer in model_args.router_layers}
    return default_layer_spec, layer_overrides


def _collect_router_policy_dims(reft_model, router_layers: List[int]) -> Dict[int, Dict[str, Optional[int]]]:
    model = _unwrap_model(reft_model)
    collected: Dict[int, Dict[str, Optional[int]]] = {}

    for layer_key, intervention_value in model.interventions.items():
        layer_idx = RouterMonitorTrainer._extract_layer_idx(layer_key)
        if layer_idx is None or layer_idx not in router_layers:
            continue

        intervention = _unwrap_intervention(intervention_value)
        if getattr(intervention, "policy_type", None) != "trainable":
            continue

        hidden_dim = None
        policy_head = getattr(intervention, "policy_head", None)
        if policy_head is not None and len(policy_head) > 0 and hasattr(policy_head[0], "out_features"):
            hidden_dim = int(policy_head[0].out_features)

        projection_dim = None
        specialist_proj_weight = getattr(intervention, "specialist_proj_weight", None)
        if specialist_proj_weight is not None:
            projection_dim = int(specialist_proj_weight.shape[1])
        else:
            specialist_proj_bias = getattr(intervention, "specialist_proj_bias", None)
            if specialist_proj_bias is not None:
                projection_dim = int(specialist_proj_bias.shape[1])

        pre_hidden_state_dim = None
        pre_hidden_proj = getattr(intervention, "pre_hidden_proj", None)
        if pre_hidden_proj is not None:
            pre_hidden_state_dim = int(pre_hidden_proj.out_features)

        collected[int(layer_idx)] = {
            "policy_hidden_dim": hidden_dim,
            "policy_projection_dim": projection_dim,
            "pre_hidden_state_dim": pre_hidden_state_dim,
        }

    return collected


def _validate_router_policy_dims(
    reft_model,
    router_layers: List[int],
    expected_hidden_dim: int,
    expected_projection_dim: int,
    expected_use_pre_hidden_state_feature: bool,
    expected_pre_hidden_state_dim: int,
) -> Dict[int, Dict[str, Optional[int]]]:
    actual_dims = _collect_router_policy_dims(reft_model, router_layers)
    missing_layers = [int(layer) for layer in router_layers if int(layer) not in actual_dims]
    if missing_layers:
        raise ValueError(f"Missing trainable router interventions for layers: {missing_layers}")

    mismatches = []
    for layer in sorted(actual_dims):
        actual_hidden_dim = actual_dims[layer]["policy_hidden_dim"]
        actual_projection_dim = actual_dims[layer]["policy_projection_dim"]
        actual_pre_hidden_state_dim = actual_dims[layer].get("pre_hidden_state_dim")
        pre_hidden_state_mismatch = (
            actual_pre_hidden_state_dim != int(expected_pre_hidden_state_dim)
            if expected_use_pre_hidden_state_feature
            else actual_pre_hidden_state_dim is not None
        )
        if (
            actual_hidden_dim != int(expected_hidden_dim)
            or actual_projection_dim != int(expected_projection_dim)
            or pre_hidden_state_mismatch
        ):
            mismatches.append(
                {
                    "layer": int(layer),
                    "expected_hidden_dim": int(expected_hidden_dim),
                    "actual_hidden_dim": actual_hidden_dim,
                    "expected_projection_dim": int(expected_projection_dim),
                    "actual_projection_dim": actual_projection_dim,
                    "expected_use_pre_hidden_state_feature": bool(expected_use_pre_hidden_state_feature),
                    "expected_pre_hidden_state_dim": (
                        int(expected_pre_hidden_state_dim) if expected_use_pre_hidden_state_feature else None
                    ),
                    "actual_pre_hidden_state_dim": actual_pre_hidden_state_dim,
                }
            )
    if mismatches:
        raise ValueError(
            "Router dimension mismatch after model construction. "
            f"Requested hidden/projection=({expected_hidden_dim}, {expected_projection_dim}), "
            f"but actual layers were {mismatches}."
        )
    return actual_dims


def _build_router_metadata(
    model_args: RouterModelArguments,
    data_args: RouterDataArguments,
    training_args: RouterTrainingArguments,
    resolved_target_layers: List[int],
    task_id_to_name: Dict[int, str],
    specialist_labels: List[str],
    sampling_probs,
    task_names: List[str],
    task_train_quotas,
    actual_router_dims: Optional[Dict[int, Dict[str, Optional[int]]]] = None,
) -> Dict:
    actual_hidden_dim = model_args.policy_hidden_dim
    actual_projection_dim = model_args.policy_projection_dim
    actual_pre_hidden_state_dim = model_args.pre_hidden_state_dim if model_args.use_pre_hidden_state_feature else None
    if actual_router_dims:
        first_layer = sorted(actual_router_dims)[0]
        first_dims = actual_router_dims[first_layer]
        if first_dims.get("policy_hidden_dim") is not None:
            actual_hidden_dim = int(first_dims["policy_hidden_dim"])
        if first_dims.get("policy_projection_dim") is not None:
            actual_projection_dim = int(first_dims["policy_projection_dim"])
        if first_dims.get("pre_hidden_state_dim") is not None:
            actual_pre_hidden_state_dim = int(first_dims["pre_hidden_state_dim"])

    return {
        "target_layers": list(resolved_target_layers),
        "router_layers": model_args.router_layers,
        "router_layer_policy": model_args.router_layer_policy,
        "router_feature_names": model_args.router_feature_names,
        "router_temperature": model_args.router_temperature,
        "router_hidden_dim": actual_hidden_dim,
        "router_projection_dim": actual_projection_dim,
        "use_pre_hidden_state_feature": model_args.use_pre_hidden_state_feature,
        "pre_hidden_state_dim": actual_pre_hidden_state_dim,
        "requested_router_hidden_dim": model_args.policy_hidden_dim,
        "requested_router_projection_dim": model_args.policy_projection_dim,
        "requested_pre_hidden_state_dim": (
            model_args.pre_hidden_state_dim if model_args.use_pre_hidden_state_feature else None
        ),
        "router_feature_stats_path": model_args.router_feature_stats_path,
        "router_feature_clip": model_args.router_feature_clip,
        "router_feature_eps": model_args.router_feature_eps,
        "base_composition_method": "compat_filtered_topk",
        "base_composition_temperature": model_args.base_composition_temperature,
        "base_composition_topk": model_args.base_composition_topk,
        "base_compat_threshold": model_args.base_compat_threshold,
        "base_score_source": model_args.base_score_source,
        "base_score_normalizer": model_args.base_score_normalizer,
        "base_score_stats_path": model_args.base_score_stats_path,
        "compat_impl": model_args.compat_impl,
        "dropout": training_args.dropout,
        "non_router_policy": "E6",
        "compose_domain": model_args.compose_domain,
        "task_id_to_name": task_id_to_name,
        "specialist_labels": specialist_labels,
        "sampling_probs": None if sampling_probs is None else dict(zip(task_names, sampling_probs)),
        "task_train_quotas": task_train_quotas,
        "router_kl_weight": training_args.router_kl_weight,
        "router_kl_target_prob": training_args.router_kl_target_prob,
        "bias_router_kl_stereotype_prob": training_args.bias_router_kl_stereotype_prob,
        "bias_router_kl_truth_prob": training_args.bias_router_kl_truth_prob,
        "bbq_qa_bridge_count": data_args.bbq_qa_bridge_count,
        "bbq_qa_bridge_sampling_prob": data_args.bbq_qa_bridge_sampling_prob,
        "bbq_qa_bridge_kl_truth_prob": training_args.bbq_qa_bridge_kl_truth_prob,
        "bbq_qa_bridge_kl_stereotype_prob": training_args.bbq_qa_bridge_kl_stereotype_prob,
    }


def _write_router_metadata(metadata: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


class RouterMetadataCallback(TrainerCallback):
    def __init__(self, metadata: Dict):
        self.metadata = dict(metadata)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        _write_router_metadata(self.metadata, checkpoint_dir / "router_training_metadata.json")
        return control


def _attach_task_ids(reft_dataset: ReftSupervisedDataset, task_name_to_id: Dict[str, int]) -> None:
    if not hasattr(reft_dataset, "task_dataset"):
        raise ValueError("Reft dataset is missing task_dataset; cannot attach task_ids.")
    if len(reft_dataset.result) != len(reft_dataset.task_dataset):
        raise ValueError("Reft dataset result/task_dataset length mismatch when attaching task_ids.")

    for idx, data_item in enumerate(reft_dataset.task_dataset):
        task_name = _normalize_task_name(data_item["task_name"])
        if task_name not in task_name_to_id:
            raise ValueError(f"Unknown task_name in dataset: {task_name}")
        reft_dataset.result[idx]["task_ids"] = int(task_name_to_id[task_name])


class RouterMonitorCollator:
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, instances):
        if not instances:
            raise ValueError("RouterMonitorCollator received an empty batch.")
        has_task_ids = all("task_ids" in instance for instance in instances)
        task_ids = [int(instance["task_ids"]) for instance in instances] if has_task_ids else None
        stripped_instances = []
        for instance in instances:
            item = dict(instance)
            item.pop("task_ids", None)
            stripped_instances.append(item)
        if any("input_ids" not in instance for instance in stripped_instances):
            raise ValueError(
                "RouterMonitorCollator received features without input_ids. "
                "This usually means Trainer removed required columns before collation."
            )
        batch_inputs = self.data_collator(stripped_instances)
        if task_ids is not None:
            batch_inputs["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
        return batch_inputs


class RouterMonitorTrainer(ReftTrainerForCausalLMDistributed):
    def __init__(
        self,
        *args,
        task_id_to_name: Dict[int, str],
        specialist_labels: List[str],
        router_layers: List[int],
        router_kl_weight: float = 0.0,
        router_kl_target_prob: float = 0.8,
        bias_router_kl_stereotype_prob: float = 0.6,
        bias_router_kl_truth_prob: float = 0.25,
        bbq_qa_bridge_kl_truth_prob: float = 0.45,
        bbq_qa_bridge_kl_stereotype_prob: float = 0.35,
        monitor_during_eval: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_id_to_name = {int(k): str(v) for k, v in task_id_to_name.items()}
        self.specialist_labels = list(specialist_labels)
        self.router_layers = [int(layer) for layer in router_layers]
        self.monitor_during_eval = bool(monitor_during_eval)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self.monitor_accumulator = self._empty_monitor_accumulator()
        self.last_monitor_logs: Dict[str, float] = {}
        self.last_router_kl_loss: Optional[float] = None
        self.router_kl_weight = float(router_kl_weight)
        self.router_kl_target_prob = float(router_kl_target_prob)
        self.bias_router_kl_stereotype_prob = float(bias_router_kl_stereotype_prob)
        self.bias_router_kl_truth_prob = float(bias_router_kl_truth_prob)
        self.bbq_qa_bridge_kl_truth_prob = float(bbq_qa_bridge_kl_truth_prob)
        self.bbq_qa_bridge_kl_stereotype_prob = float(bbq_qa_bridge_kl_stereotype_prob)
        self.truth_idx = self._find_specialist_index(["truthful"])
        self.tox_idx = self._find_specialist_index(["toxicity", "toxic"])
        self.moral_idx = self._find_specialist_index(["moral", "ethics"])
        self.stereo_idx = self._find_specialist_index(["stereotype", "bias"])
        self.task_target_specialist_idx = self._build_task_target_specialist_idx()
        if self.router_kl_weight > 0.0:
            unresolved_tasks = [
                task_name
                for task_id, task_name in sorted(self.task_id_to_name.items())
                if int(task_id) not in self.task_target_specialist_idx
            ]
            if unresolved_tasks:
                raise ValueError(
                    "Task-level KL could not map tasks to target specialists: "
                    f"{unresolved_tasks}. Check task names and specialist labels."
                )

    @staticmethod
    def _extract_layer_idx(layer_key) -> Optional[int]:
        if isinstance(layer_key, int):
            return layer_key
        if isinstance(layer_key, str):
            if layer_key.isdigit():
                return int(layer_key)
            match = re.search(r"layer_(\d+)", layer_key)
            if match is not None:
                return int(match.group(1))
        return None

    def _find_specialist_index(self, candidates: List[str]) -> Optional[int]:
        lowered = [label.lower() for label in self.specialist_labels]
        for idx, label in enumerate(lowered):
            if any(candidate in label for candidate in candidates):
                return idx
        return None

    def _build_task_target_specialist_idx(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for task_id, task_name in self.task_id_to_name.items():
            normalized_task_name = _normalize_task_name(task_name)
            target_idx = None
            if normalized_task_name in {"truthful", "truthfulqa"}:
                target_idx = self.truth_idx
            elif normalized_task_name in {"ethics", "moral"}:
                target_idx = self.moral_idx
            elif normalized_task_name in {"bbq", "stereotype", "bias", "bbq_qa_bridge"}:
                target_idx = self.stereo_idx
            elif normalized_task_name in {"toxicity", "toxic"}:
                target_idx = self.tox_idx
            if target_idx is not None:
                mapping[int(task_id)] = int(target_idx)
        return mapping

    def _empty_monitor_accumulator(self) -> Dict[str, Dict]:
        return {
            "loss_by_task": {},
            "alpha_by_task": {},
            "dominant_by_task": {},
            "entropy_by_task": {},
            "late_diffs": {},
        }

    def _accumulate_scalar(self, bucket: Dict, key: str, value: torch.Tensor, count: torch.Tensor) -> None:
        if key not in bucket:
            bucket[key] = {
                "sum": value.detach().to(torch.float32),
                "count": count.detach().to(torch.float32),
            }
        else:
            bucket[key]["sum"] += value.detach().to(torch.float32)
            bucket[key]["count"] += count.detach().to(torch.float32)

    def _update_loss_monitors(self, logits: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor) -> None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        valid_mask = shift_labels.ne(-100)
        token_counts = valid_mask.sum(dim=-1).clamp_min(1)
        per_sample_loss = (flat_loss * valid_mask).sum(dim=-1) / token_counts

        unique_task_ids = task_ids.unique()
        for task_id in unique_task_ids.tolist():
            task_mask = task_ids == task_id
            task_name = self.task_id_to_name[int(task_id)]
            self._accumulate_scalar(
                self.monitor_accumulator["loss_by_task"],
                task_name,
                per_sample_loss[task_mask].sum(),
                task_mask.sum(),
            )

    def _update_alpha_monitors(self, intervenable, task_ids: torch.Tensor) -> None:
        model = _unwrap_model(intervenable)
        unique_task_ids = task_ids.unique()

        for layer_key, intervention_value in model.interventions.items():
            layer_idx = self._extract_layer_idx(layer_key)
            if layer_idx is None:
                continue
            if layer_idx not in self.router_layers:
                continue

            intervention = _unwrap_intervention(intervention_value)
            alpha = getattr(intervention, "latest_monitor_alpha", None)
            if alpha is None:
                continue

            alpha = alpha.float()
            if alpha.dim() != 3:
                continue
            alpha_mean_by_sample = alpha.mean(dim=1)
            dominant = alpha.argmax(dim=-1)
            entropy = -(alpha.clamp_min(1e-8) * alpha.clamp_min(1e-8).log()).sum(dim=-1)

            for task_id in unique_task_ids.tolist():
                task_mask = task_ids == task_id
                if not torch.any(task_mask):
                    continue
                task_name = self.task_id_to_name[int(task_id)]
                task_alpha = alpha_mean_by_sample[task_mask]
                task_entropy = entropy[task_mask]
                task_dominant = dominant[task_mask]

                for spec_idx, specialist_label in enumerate(self.specialist_labels):
                    alpha_key = f"{task_name}|L{layer_idx}|{specialist_label}"
                    dominant_key = f"{task_name}|L{layer_idx}|{specialist_label}"
                    self._accumulate_scalar(
                        self.monitor_accumulator["alpha_by_task"],
                        alpha_key,
                        task_alpha[:, spec_idx].sum(),
                        torch.tensor(float(task_alpha.shape[0]), device=task_alpha.device),
                    )
                    self._accumulate_scalar(
                        self.monitor_accumulator["dominant_by_task"],
                        dominant_key,
                        (task_dominant == spec_idx).float().sum(),
                        torch.tensor(float(task_dominant.numel()), device=task_dominant.device),
                    )

                entropy_key = f"{task_name}|L{layer_idx}"
                self._accumulate_scalar(
                    self.monitor_accumulator["entropy_by_task"],
                    entropy_key,
                    task_entropy.sum(),
                    torch.tensor(float(task_entropy.numel()), device=task_entropy.device),
                )

                if task_name == "toxicity" and self.truth_idx is not None and self.tox_idx is not None:
                    diff_key = "truth_minus_tox_alpha_late"
                    diff_values = task_alpha[:, self.truth_idx] - task_alpha[:, self.tox_idx]
                    self._accumulate_scalar(
                        self.monitor_accumulator["late_diffs"],
                        diff_key,
                        diff_values.sum(),
                        torch.tensor(float(diff_values.numel()), device=diff_values.device),
                    )
                if task_name == "ethics" and self.truth_idx is not None and self.moral_idx is not None:
                    diff_key = "truth_minus_moral_alpha_late"
                    diff_values = task_alpha[:, self.truth_idx] - task_alpha[:, self.moral_idx]
                    self._accumulate_scalar(
                        self.monitor_accumulator["late_diffs"],
                        diff_key,
                        diff_values.sum(),
                        torch.tensor(float(diff_values.numel()), device=diff_values.device),
                    )

    def _reduce_accumulator_bucket(self, bucket: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        if dist.is_available() and dist.is_initialized():
            local_keys = sorted(bucket.keys())
            gathered_keys = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_keys, local_keys)
            keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})
            if not keys:
                return {}
            if bucket:
                device = next(iter(bucket.values()))["sum"].device
            elif torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
            else:
                device = torch.device("cpu")
        else:
            if not bucket:
                return {}
            keys = sorted(bucket.keys())
            device = next(iter(bucket.values()))["sum"].device

        zero = torch.zeros((), device=device, dtype=torch.float32)
        values = []
        for key in keys:
            if key in bucket:
                values.append(bucket[key]["sum"].detach().to(device=device, dtype=torch.float32))
                values.append(bucket[key]["count"].detach().to(device=device, dtype=torch.float32))
            else:
                values.append(zero.clone())
                values.append(zero.clone())
        stacked = torch.stack(values)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)

        reduced = {}
        for idx, key in enumerate(keys):
            reduced[key] = {
                "sum": float(stacked[2 * idx].item()),
                "count": float(stacked[2 * idx + 1].item()),
            }
        return reduced

    def _build_monitor_logs(self) -> Dict[str, float]:
        logs = {}
        reduced_loss = self._reduce_accumulator_bucket(self.monitor_accumulator["loss_by_task"])
        reduced_alpha = self._reduce_accumulator_bucket(self.monitor_accumulator["alpha_by_task"])
        reduced_dominant = self._reduce_accumulator_bucket(self.monitor_accumulator["dominant_by_task"])
        reduced_entropy = self._reduce_accumulator_bucket(self.monitor_accumulator["entropy_by_task"])
        reduced_diffs = self._reduce_accumulator_bucket(self.monitor_accumulator["late_diffs"])

        for task_name, stats in reduced_loss.items():
            if stats["count"] > 0:
                logs[f"loss_{task_name}"] = stats["sum"] / stats["count"]

        for packed_key, stats in reduced_alpha.items():
            if stats["count"] <= 0:
                continue
            task_name, layer_name, specialist_label = packed_key.split("|", 2)
            logs[f"alpha_mean/{task_name}/{layer_name}/{specialist_label}"] = stats["sum"] / stats["count"]

        for packed_key, stats in reduced_dominant.items():
            if stats["count"] <= 0:
                continue
            task_name, layer_name, specialist_label = packed_key.split("|", 2)
            logs[f"dominant_rate/{task_name}/{layer_name}/{specialist_label}"] = stats["sum"] / stats["count"]

        for packed_key, stats in reduced_entropy.items():
            if stats["count"] <= 0:
                continue
            task_name, layer_name = packed_key.split("|", 1)
            logs[f"alpha_entropy/{task_name}/{layer_name}"] = stats["sum"] / stats["count"]

        for diff_name, stats in reduced_diffs.items():
            if stats["count"] > 0:
                logs[diff_name] = stats["sum"] / stats["count"]

        return logs

    def _reset_monitor_accumulator(self) -> None:
        self.monitor_accumulator = self._empty_monitor_accumulator()

    def flush_router_monitor_logs(self) -> None:
        monitor_logs = self._build_monitor_logs()
        self._reset_monitor_accumulator()
        self.last_monitor_logs = dict(monitor_logs)
        if self.last_router_kl_loss is not None:
            self.last_monitor_logs["router_kl_loss"] = float(self.last_router_kl_loss)
        if self.last_monitor_logs:
            super().log(self.last_monitor_logs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        monitor_logs = self._build_monitor_logs()
        self._reset_monitor_accumulator()
        self.last_monitor_logs = dict(monitor_logs)
        if self.last_router_kl_loss is not None:
            self.last_monitor_logs["router_kl_loss"] = float(self.last_router_kl_loss)
        merged_logs = dict(logs)
        merged_logs.update(self.last_monitor_logs)
        return super().log(merged_logs, start_time=start_time)

    def evaluate_with_monitor_logs(self, eval_dataset=None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        self.last_monitor_logs = {}
        metrics = self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=metric_key_prefix)
        merged_metrics = dict(metrics)
        merged_metrics.update(self.last_monitor_logs)
        return merged_metrics

    def _collect_router_alphas_for_loss(self, intervenable) -> List[torch.Tensor]:
        model = _unwrap_model(intervenable)
        alphas: List[torch.Tensor] = []
        for layer_key, intervention_value in model.interventions.items():
            layer_idx = self._extract_layer_idx(layer_key)
            if layer_idx is None or layer_idx not in self.router_layers:
                continue
            intervention = _unwrap_intervention(intervention_value)
            alpha = getattr(intervention, "latest_policy_alpha", None)
            if alpha is not None:
                alphas.append(alpha)
        return alphas

    def _build_task_soft_targets(self, task_ids: torch.Tensor, num_specialists: int, dtype: torch.dtype) -> torch.Tensor:
        if num_specialists <= 1:
            raise ValueError("Task-level KL requires at least two specialists.")
        off_value = (1.0 - self.router_kl_target_prob) / float(num_specialists - 1)
        target = torch.full(
            (task_ids.shape[0], num_specialists),
            off_value,
            dtype=dtype,
            device=task_ids.device,
        )
        for task_id in task_ids.unique().tolist():
            task_id_int = int(task_id)
            if task_id_int not in self.task_target_specialist_idx:
                raise ValueError(
                    "Missing task-level KL target specialist for task "
                    f"{self.task_id_to_name.get(task_id_int, task_id)}."
                )
            task_mask = task_ids == task_id_int
            task_name = _normalize_task_name(self.task_id_to_name.get(task_id_int, str(task_id_int)))
            if task_name in {"bbq", "stereotype", "bias"}:
                if self.stereo_idx is None or self.truth_idx is None:
                    raise ValueError("Bias soft-target KL requires stereotype and truthful specialists.")
                target[task_mask] = self._build_bias_soft_target(num_specialists, dtype, task_ids.device)
                continue
            if task_name == "bbq_qa_bridge":
                if self.stereo_idx is None or self.truth_idx is None:
                    raise ValueError("BBQ QA bridge soft-target KL requires stereotype and truthful specialists.")
                target[task_mask] = self._build_bbq_qa_bridge_soft_target(num_specialists, dtype, task_ids.device)
                continue
            target_idx = self.task_target_specialist_idx[task_id_int]
            target[task_mask, target_idx] = self.router_kl_target_prob
        return target

    def _build_bias_soft_target(self, num_specialists: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        remaining = 1.0 - self.bias_router_kl_stereotype_prob - self.bias_router_kl_truth_prob
        non_bias_truth = num_specialists - 2
        if non_bias_truth < 0:
            raise ValueError("Bias soft-target KL requires at least two specialists.")
        if non_bias_truth == 0:
            if abs(remaining) > 1e-8:
                raise ValueError(
                    "Bias soft-target KL probabilities must sum to 1.0 when only stereotype and truthful specialists exist."
                )
            base = torch.zeros(num_specialists, dtype=dtype, device=device)
        else:
            base = torch.full(
                (num_specialists,),
                remaining / float(non_bias_truth),
                dtype=dtype,
                device=device,
            )
        base[self.stereo_idx] = self.bias_router_kl_stereotype_prob
        base[self.truth_idx] = self.bias_router_kl_truth_prob
        return base

    def _build_bbq_qa_bridge_soft_target(
        self,
        num_specialists: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        remaining = 1.0 - self.bbq_qa_bridge_kl_truth_prob - self.bbq_qa_bridge_kl_stereotype_prob
        non_bridge_focus = num_specialists - 2
        if non_bridge_focus < 0:
            raise ValueError("BBQ QA bridge soft-target KL requires at least two specialists.")
        if non_bridge_focus == 0:
            if abs(remaining) > 1e-8:
                raise ValueError(
                    "BBQ QA bridge soft-target KL probabilities must sum to 1.0 when only truthful and stereotype specialists exist."
                )
            base = torch.zeros(num_specialists, dtype=dtype, device=device)
        else:
            base = torch.full(
                (num_specialists,),
                remaining / float(non_bridge_focus),
                dtype=dtype,
                device=device,
            )
        base[self.truth_idx] = self.bbq_qa_bridge_kl_truth_prob
        base[self.stereo_idx] = self.bbq_qa_bridge_kl_stereotype_prob
        return base

    def _compute_task_level_kl_loss(self, intervenable, task_ids: torch.Tensor) -> torch.Tensor:
        router_alphas = self._collect_router_alphas_for_loss(intervenable)
        if not router_alphas:
            raise ValueError("Task-level KL requested but no trainable router alphas were captured.")

        first_alpha = router_alphas[0]
        target = self._build_task_soft_targets(
            task_ids=task_ids,
            num_specialists=first_alpha.shape[-1],
            dtype=first_alpha.dtype,
        )
        target = target.unsqueeze(1)

        per_layer_losses = []
        for alpha in router_alphas:
            if alpha.dim() != 3:
                raise ValueError(f"Expected router alpha shape [B, T, S], got {tuple(alpha.shape)}.")
            expanded_target = target.expand(-1, alpha.shape[1], -1)
            log_alpha = torch.log(alpha.clamp_min(1e-8))
            per_layer_losses.append(-(expanded_target * log_alpha).sum(dim=-1).mean())
        return torch.stack(per_layer_losses).mean()

    def compute_loss(
        self,
        intervenable,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        task_ids = inputs.pop("task_ids", None)
        unit_locations = None
        if "intervention_locations" in inputs:
            if inputs["intervention_locations"].dim() == 3:
                unit_locations = {
                    "sources->base": (
                        None,
                        inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                    )
                }
            else:
                unit_locations = {"sources->base": (None, 0)}
        base_outputs, cf_outputs = intervenable(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=inputs["subspaces"].permute(1, 0, 2).tolist() if "subspaces" in inputs else None,
        )
        output = cf_outputs if cf_outputs is not None else base_outputs
        loss = output.loss
        self.last_router_kl_loss = None

        if self.router_kl_weight > 0.0:
            if task_ids is None:
                raise ValueError("Task-level KL requires task_ids in the batch.")
            task_ids = task_ids.to(inputs["labels"].device)
            kl_loss = self._compute_task_level_kl_loss(intervenable, task_ids)
            loss = loss + (self.router_kl_weight * kl_loss)
            self.last_router_kl_loss = float(kl_loss.detach().item())

        if task_ids is not None and (intervenable.training or self.monitor_during_eval):
            task_ids = task_ids.to(inputs["labels"].device)
            with torch.no_grad():
                self._update_loss_monitors(output.logits.detach(), inputs["labels"], task_ids)
                self._update_alpha_monitors(intervenable, task_ids)

        return (loss, output) if return_outputs else loss


def main():
    parser = HfArgumentParser((RouterModelArguments, RouterDataArguments, RouterTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    training_args.ddp_find_unused_parameters = False
    if training_args.dataloader_num_workers == 0:
        training_args.dataloader_num_workers = 8
    set_seed(training_args.seed)

    if not model_args.reft_specialists:
        raise ValueError("--reft_specialists must be provided.")
    if not model_args.base_score_stats_path:
        raise ValueError("--base_score_stats_path is required for non-router E6 layers.")
    if training_args.sanity_eval_only and model_args.router_layer_policy != "no_training":
        raise ValueError("sanity_eval_only requires --router_layer_policy no_training.")
    if training_args.router_kl_weight < 0.0:
        raise ValueError("--router_kl_weight must be non-negative.")
    if not (0.0 < training_args.router_kl_target_prob < 1.0):
        raise ValueError("--router_kl_target_prob must be in (0, 1).")
    if not (0.0 <= training_args.bias_router_kl_stereotype_prob <= 1.0):
        raise ValueError("--bias_router_kl_stereotype_prob must be in [0, 1].")
    if not (0.0 <= training_args.bias_router_kl_truth_prob <= 1.0):
        raise ValueError("--bias_router_kl_truth_prob must be in [0, 1].")
    if training_args.bias_router_kl_stereotype_prob + training_args.bias_router_kl_truth_prob > 1.0:
        raise ValueError(
            "--bias_router_kl_stereotype_prob + --bias_router_kl_truth_prob must be <= 1.0."
        )
    if data_args.bbq_qa_bridge_count < 0:
        raise ValueError("--bbq_qa_bridge_count must be non-negative.")
    if data_args.bbq_qa_bridge_sampling_prob is not None and data_args.bbq_qa_bridge_sampling_prob < 0.0:
        raise ValueError("--bbq_qa_bridge_sampling_prob must be non-negative when provided.")
    if not (0.0 <= training_args.bbq_qa_bridge_kl_truth_prob <= 1.0):
        raise ValueError("--bbq_qa_bridge_kl_truth_prob must be in [0, 1].")
    if not (0.0 <= training_args.bbq_qa_bridge_kl_stereotype_prob <= 1.0):
        raise ValueError("--bbq_qa_bridge_kl_stereotype_prob must be in [0, 1].")
    if training_args.bbq_qa_bridge_kl_truth_prob + training_args.bbq_qa_bridge_kl_stereotype_prob > 1.0:
        raise ValueError(
            "--bbq_qa_bridge_kl_truth_prob + --bbq_qa_bridge_kl_stereotype_prob must be <= 1.0."
        )
    if training_args.router_kl_weight > 0.0 and model_args.router_layer_policy != "trainable":
        raise ValueError("Task-level KL is only supported with --router_layer_policy trainable.")

    task_dataset_map = _load_json_value(data_args.task_dataset_map_json)
    normalized_task_map = {_normalize_task_name(k): v for k, v in task_dataset_map.items()}
    base_task_names = list(normalized_task_map.keys())
    task_train_quotas = _build_task_train_quotas(base_task_names, data_args.task_train_quota_json)
    sampling_probs = (
        None
        if task_train_quotas is not None
        else _build_sampling_probabilities(base_task_names, data_args.sampling_probs_json)
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    train_datasets = []
    eval_datasets = []
    task_size_rows = []
    resolved_task_names: List[str] = list(base_task_names)
    truthful_train_dataset_for_bridge: Optional[Dataset] = None
    truthful_eval_dataset_for_bridge: Optional[Dataset] = None
    for task_name in base_task_names:
        dataset = _load_task_dataset(
            task_name=task_name,
            spec=normalized_task_map[task_name],
            max_samples=data_args.max_train_samples_per_task,
            seed=training_args.seed,
        )
        train_dataset, eval_dataset = _split_train_eval(
            dataset=dataset,
            eval_holdout_ratio=data_args.eval_holdout_ratio,
            max_eval_samples=data_args.max_eval_samples_per_task,
            seed=training_args.seed,
        )
        requested_quota = None
        final_train_dataset = train_dataset
        if task_train_quotas is not None:
            requested_quota = task_train_quotas[task_name]
            final_train_dataset = _truncate_dataset(train_dataset, requested_quota)
        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)
        if task_train_quotas is not None:
            train_datasets[-1] = final_train_dataset
        if task_name == "truthfulqa":
            truthful_train_dataset_for_bridge = train_dataset
            truthful_eval_dataset_for_bridge = eval_dataset
        task_size_rows.append(
            {
                "task": task_name,
                "raw": len(dataset),
                "train_after_holdout": len(train_dataset),
                "train_selected": len(final_train_dataset),
                "eval": len(eval_dataset),
                "requested_quota": requested_quota,
            }
        )

    if data_args.bbq_qa_bridge_count > 0:
        if "bbq" not in normalized_task_map:
            raise ValueError("--bbq_qa_bridge_count requires a `bbq` task in task_dataset_map_json.")
        if truthful_train_dataset_for_bridge is None or truthful_eval_dataset_for_bridge is None:
            raise ValueError("--bbq_qa_bridge_count requires a `truthfulqa` task in task_dataset_map_json.")
        bridge_train_dataset, bridge_eval_dataset = _build_bbq_qa_bridge_datasets(
            truthful_train_dataset=truthful_train_dataset_for_bridge,
            truthful_eval_dataset=truthful_eval_dataset_for_bridge,
            bridge_count=int(data_args.bbq_qa_bridge_count),
            max_eval_samples=int(data_args.max_eval_samples_per_task),
        )
        resolved_task_names.append("bbq_qa_bridge")
        train_datasets.append(bridge_train_dataset)
        eval_datasets.append(bridge_eval_dataset)
        task_size_rows.append(
            {
                "task": "bbq_qa_bridge",
                "raw": len(truthful_train_dataset_for_bridge),
                "train_after_holdout": len(truthful_train_dataset_for_bridge),
                "train_selected": len(bridge_train_dataset),
                "eval": len(bridge_eval_dataset),
                "requested_quota": int(data_args.bbq_qa_bridge_count),
            }
        )
        if sampling_probs is not None:
            sampling_prob_by_task = dict(zip(base_task_names, sampling_probs))
            bridge_prob = (
                float(data_args.bbq_qa_bridge_sampling_prob)
                if data_args.bbq_qa_bridge_sampling_prob is not None
                else float(sampling_prob_by_task.get("bbq", 0.0))
            )
            sampling_prob_by_task["bbq_qa_bridge"] = bridge_prob
            ordered_probs = [float(sampling_prob_by_task[task_name]) for task_name in resolved_task_names]
            total_prob = sum(ordered_probs)
            if total_prob <= 0:
                raise ValueError("Sampling probabilities must sum to a positive value after adding bbq_qa_bridge.")
            sampling_probs = [value / total_prob for value in ordered_probs]

    task_names = list(resolved_task_names)
    task_name_to_id = {task_name: idx for idx, task_name in enumerate(task_names)}
    task_id_to_name = {idx: task_name for task_name, idx in task_name_to_id.items()}

    if task_train_quotas is not None:
        mixed_train_dataset = datasets.concatenate_datasets(train_datasets)
        mixed_train_dataset = _shuffle_dataset(mixed_train_dataset, seed=training_args.seed)
    else:
        mixed_train_dataset = interleave_datasets(
            train_datasets,
            probabilities=sampling_probs,
            seed=training_args.seed,
            stopping_strategy="all_exhausted",
        )
    mixed_eval_dataset = datasets.concatenate_datasets(eval_datasets)
    mixed_eval_dataset = _truncate_eval_for_distributed(
        mixed_eval_dataset,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    )

    if local_rank == 0:
        print("Router stage-2 dataset summary:")
        for row in task_size_rows:
            quota_text = row["requested_quota"] if row["requested_quota"] is not None else "NA"
            print(
                f"  task={row['task']} raw={row['raw']} "
                f"train_after_holdout={row['train_after_holdout']} "
                f"train_selected={row['train_selected']} "
                f"eval={row['eval']} quota={quota_text}"
            )
        print(f"  mixed_train_total={len(mixed_train_dataset)}")
        print(f"  mixed_eval_total={len(mixed_eval_dataset)}")

    model, tokenizer = _build_model_and_tokenizer(model_args, training_args)
    default_layer_spec, layer_overrides = _build_layer_specs(model_args, training_args.dropout)
    resolved_target_layers = (
        list(range(len(model.model.layers)))
        if model_args.target_layers == [-1]
        else [int(layer) for layer in model_args.target_layers]
    )
    reft_model = load_mixed_composed_reft_model(
        model=model,
        specialist_dirs=model_args.reft_specialists,
        target_layers=resolved_target_layers,
        default_layer_spec=default_layer_spec,
        layer_policy_specs=layer_overrides,
    )
    actual_router_dims = {}
    if model_args.router_layer_policy == "trainable":
        actual_router_dims = _validate_router_policy_dims(
            reft_model=reft_model,
            router_layers=[int(layer) for layer in model_args.router_layers],
            expected_hidden_dim=int(model_args.policy_hidden_dim),
            expected_projection_dim=int(model_args.policy_projection_dim),
            expected_use_pre_hidden_state_feature=bool(model_args.use_pre_hidden_state_feature),
            expected_pre_hidden_state_dim=int(model_args.pre_hidden_state_dim),
        )
        if local_rank == 0:
            print("Validated router policy dims:")
            for layer in sorted(actual_router_dims):
                layer_dims = actual_router_dims[layer]
                print(
                    f"  layer={layer} "
                    f"hidden_dim={layer_dims['policy_hidden_dim']} "
                    f"projection_dim={layer_dims['policy_projection_dim']} "
                    f"pre_hidden_state_dim={layer_dims.get('pre_hidden_state_dim')}"
                )
    _freeze_for_router_training(reft_model)
    reft_model.print_trainable_parameters()
    specialist_labels = [normalize_specialist_label(path) for path in model_args.reft_specialists]
    router_metadata = _build_router_metadata(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        resolved_target_layers=resolved_target_layers,
        task_id_to_name=task_id_to_name,
        specialist_labels=specialist_labels,
        sampling_probs=sampling_probs,
        task_names=task_names,
        task_train_quotas=task_train_quotas,
        actual_router_dims=actual_router_dims,
    )

    train_dataset = ReftSupervisedDataset(
        "ComposableRouter",
        None,
        tokenizer,
        dataset=mixed_train_dataset,
        seed=training_args.seed,
        max_n_example=None,
        no_stop=False,
        **{
            "num_interventions": len(reft_model.interventions),
            "position": model_args.positions,
            "share_weights": True,
        },
        input_field="input",
        instruction_field="instruction",
        output_field="output",
    )
    _attach_task_ids(train_dataset, task_name_to_id)

    eval_dataset = ReftSupervisedDataset(
        "ComposableRouterEval",
        None,
        tokenizer,
        dataset=mixed_eval_dataset,
        seed=training_args.seed,
        max_n_example=None,
        no_stop=False,
        **{
            "num_interventions": len(reft_model.interventions),
            "position": model_args.positions,
            "share_weights": True,
        },
        input_field="input",
        instruction_field="instruction",
        output_field="output",
    )
    _attach_task_ids(eval_dataset, task_name_to_id)

    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )
    data_collator = RouterMonitorCollator(ReftDataCollator(data_collator=data_collator_fn))

    trainer = RouterMonitorTrainer(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        task_id_to_name=task_id_to_name,
        specialist_labels=specialist_labels,
        router_layers=model_args.router_layers,
        router_kl_weight=training_args.router_kl_weight,
        router_kl_target_prob=training_args.router_kl_target_prob,
        bias_router_kl_stereotype_prob=training_args.bias_router_kl_stereotype_prob,
        bias_router_kl_truth_prob=training_args.bias_router_kl_truth_prob,
        bbq_qa_bridge_kl_truth_prob=training_args.bbq_qa_bridge_kl_truth_prob,
        bbq_qa_bridge_kl_stereotype_prob=training_args.bbq_qa_bridge_kl_stereotype_prob,
        monitor_during_eval=training_args.sanity_eval_only,
    )
    trainer.add_callback(RouterMetadataCallback(router_metadata))

    if training_args.sanity_eval_only:
        split_names = _parse_sanity_eval_splits(training_args.sanity_eval_splits)
        sanity_metrics = {}
        for split_name in split_names:
            split_dataset = train_dataset if split_name == "train" else eval_dataset
            metric_key_prefix = f"{split_name}_sanity"
            sanity_metrics[split_name] = trainer.evaluate_with_monitor_logs(
                eval_dataset=split_dataset,
                metric_key_prefix=metric_key_prefix,
            )
        metrics_path = Path(training_args.output_dir) / "no_training_sanity_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(sanity_metrics, f, indent=2, sort_keys=True)
        print(f"No-training sanity metrics written to: {metrics_path}")
        return

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.flush_router_monitor_logs()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    if local_rank == 0:
        metadata_path = Path(training_args.output_dir) / "router_training_metadata.json"
        _write_router_metadata(router_metadata, metadata_path)
        print(f"Router metadata written to: {metadata_path}")


if __name__ == "__main__":
    main()
