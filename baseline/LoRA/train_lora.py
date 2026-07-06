import os
import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import transformers
import wandb
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from args import DataArguments, LoRAArguments, TrainingArguments

SUBSPACE_NAMES = ["truthful", "toxicity", "stereotype", "safety", "moral"]
SUBTASK_ALIASES = {"toxic": "toxicity"}
IGNORE_INDEX = -100

PROMPT_INPUT = """Below is an instruction that \
describes a task, paired with an input that provides \
further context. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

PROMPT_NO_INPUT = """Below is an instruction that \
describes a task. Write a response that appropriately \
completes the request.

### Instruction:
%s

### Response:
"""


truthful_trigger = "the correct answer is "
ethics_trigger =  "the action is "

def _format_truthful_output(example):
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(truthful_trigger):
        output = f"{truthful_trigger}{answer}"
    return output

def _format_ethics_out(example):
    answer = example["answer"] if "answer" in example and example["answer"] is not None else example["output"]
    answer = str(answer).strip()
    output = str(example["output"]).strip()
    if not output.lower().startswith(ethics_trigger):
        output = f"{ethics_trigger}{answer}"
    return output

def canonicalize_subtask(subtask: str) -> str:
    normalized = (subtask or "").strip().lower()
    return SUBTASK_ALIASES.get(normalized, normalized)


def load_subdataset(subtask: str, dataset_root: Path, seed: int, augment_stereotype_truthful: bool = True):
    subtask = canonicalize_subtask(subtask)
    dataset_paths = {
        "truthful": dataset_root / "alignment_truthful_format",
        "moral": dataset_root / "alignment_moral_cls",
        "safety": dataset_root / "alignment_pku_safety_format",
        "stereotype": dataset_root / "alignment_stereotype_format",
        "toxicity": dataset_root / "alignment_toxic_format",
    }

    if subtask not in dataset_paths:
        raise ValueError(f"Unsupported dataset name: {subtask}")

    if subtask == "helpful":
        return load_dataset("json", data_files=str(dataset_paths[subtask]))["train"]
    if subtask == "moral":
        return load_from_disk(str(dataset_paths[subtask]))

    if subtask == "stereotype":
        stereotype_data = load_from_disk('./dataset/alignment_stereotype_format')['train']
        if not augment_stereotype_truthful:
            return stereotype_data

        truthful_data = load_from_disk('./dataset/alignment_truthful_format')['train']
        truthful_data = truthful_data.shuffle(seed=seed)
        truthful_data = truthful_data.select(range(min(5000, len(truthful_data))))
        truthful_data = truthful_data.map(lambda x: {"output": _format_truthful_output(x)})

        # Keep only shared columns so concatenate_datasets can merge safely.
        common_columns = [col for col in stereotype_data.column_names if col in truthful_data.column_names]
        stereotype_data = stereotype_data.remove_columns(
            [col for col in stereotype_data.column_names if col not in common_columns]
        )
        truthful_data = truthful_data.remove_columns(
            [col for col in truthful_data.column_names if col not in common_columns]
        )

        data = concatenate_datasets([stereotype_data, truthful_data])
        return data
    return load_from_disk(str(dataset_paths[subtask]))["train"]


def prepare_subtask_dataset(
    subtask: str,
    dataset_root: Path,
    seed: int,
    augment_stereotype_truthful: bool = True,
):
    subtask = canonicalize_subtask(subtask)
    if subtask not in SUBSPACE_NAMES:
        raise ValueError(f"Unsupported subtask: {subtask}. Allowed: {SUBSPACE_NAMES}")

    data = load_subdataset(subtask, dataset_root, seed, augment_stereotype_truthful=augment_stereotype_truthful)
    subspace_label_id = SUBSPACE_NAMES.index(subtask)

    if subtask == "truthful":
        def _prepare_truthful(example):
            output = _format_truthful_output(example)
            return {
                "output": output,
                "subspace_labels": subspace_label_id,
            }

        data = data.map(_prepare_truthful)
    
    elif subtask == 'moral':
        def _prepare_ethics(example):
            output = _format_ethics_out(example)
            return {
                "output": output,
                "subspace_labels": subspace_label_id,
            }

        data = data.map(_prepare_ethics)
    else:
        data = data.map(lambda _: {"subspace_labels": subspace_label_id})

    return data


def build_dataset(data_args: DataArguments, dataset_root: Path, seed: int):
    dataset_name = canonicalize_subtask(data_args.dataset_name)

    if dataset_name == "combined":
        combined_subtasks = [name for name in SUBSPACE_NAMES if name != "safety"]
        datasets = [
            prepare_subtask_dataset(
                name,
                dataset_root,
                seed,
                augment_stereotype_truthful=False,
            )
            for name in combined_subtasks
        ]
        dataset = concatenate_datasets(datasets)
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = prepare_subtask_dataset(dataset_name, dataset_root, seed)

    if data_args.max_samples:
        dataset = dataset.select(range(min(data_args.max_samples, len(dataset))))

    return dataset, dataset_name


def preprocess_function(examples: Dict[str, List], tokenizer, model_max_length: int):
    if "instruction" not in examples or "output" not in examples:
        raise ValueError("Dataset must contain 'instruction' and 'output' to match multi_train/train.py logic.")

    instructions = examples["instruction"]
    outputs = examples["output"]
    inputs = examples["input"] if "input" in examples else [""] * len(instructions)

    full_prompts = []
    base_prompts = []
    for instruction, model_input, output in zip(instructions, inputs, outputs):
        instruction = "" if instruction is None else str(instruction)
        model_input = "" if model_input is None else str(model_input)
        output = "" if output is None else str(output)

        if model_input == "":
            base_prompt = PROMPT_NO_INPUT % instruction
        else:
            base_prompt = PROMPT_INPUT % (instruction, model_input)

        full_prompts.append(base_prompt + output + tokenizer.eos_token)
        base_prompts.append(base_prompt)

    tokenized = tokenizer(
        full_prompts,
        truncation=True,
        max_length=model_max_length,
        padding=False,
    )
    prompt_tokenized = tokenizer(
        base_prompts,
        truncation=True,
        max_length=model_max_length,
        padding=False,
    )

    labels = []
    for input_ids, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
        prompt_length = min(len(prompt_ids), len(input_ids))

        label_ids = list(input_ids)
        for idx in range(prompt_length):
            label_ids[idx] = -100
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _get_local_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() % max(1, torch.cuda.device_count())
    return int(os.environ.get("LOCAL_RANK", "0"))


class DiagnosticTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        diagnostic_interval: int = 100,
        diagnostic_early_steps: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diagnostic_interval = max(1, diagnostic_interval)
        self.diagnostic_early_steps = max(0, diagnostic_early_steps)
        self._forward_calls = 0
        self._zero_loss_streak = 0
        self._last_forward_diagnostics: Optional[Dict[str, Any]] = None

    def _periodic_should_log(self) -> bool:
        if self._forward_calls <= self.diagnostic_early_steps:
            return True
        return self._forward_calls % self.diagnostic_interval == 0

    def _summarize_forward_diagnostics(
        self,
        inputs: Dict[str, torch.Tensor],
        outputs: Any,
        loss: torch.Tensor,
    ) -> Dict[str, Any]:
        self._forward_calls += 1
        diag: Dict[str, Any] = {
            "rank": _get_rank(),
            "local_rank": _get_local_rank(),
            "global_step": int(self.state.global_step),
            "forward_call": self._forward_calls,
        }

        labels = inputs.get("labels")
        if labels is not None:
            valid_mask = labels.ne(IGNORE_INDEX)
            per_example_label_tokens = valid_mask.sum(dim=-1)
            diag.update(
                {
                    "batch_size": int(labels.shape[0]),
                    "seq_len": int(labels.shape[-1]),
                    "valid_label_tokens": int(valid_mask.sum().item()),
                    "zero_label_examples": int(per_example_label_tokens.eq(0).sum().item()),
                    "min_label_tokens": int(per_example_label_tokens.min().item()),
                    "max_label_tokens": int(per_example_label_tokens.max().item()),
                    "mean_label_tokens": float(per_example_label_tokens.float().mean().item()),
                }
            )

        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            input_lengths = attention_mask.sum(dim=-1)
            diag.update(
                {
                    "min_input_tokens": int(input_lengths.min().item()),
                    "max_input_tokens": int(input_lengths.max().item()),
                    "mean_input_tokens": float(input_lengths.float().mean().item()),
                }
            )

        detached_loss = loss.detach().float()
        loss_is_finite = bool(torch.isfinite(detached_loss).all().item())
        loss_value = float(detached_loss.item()) if loss_is_finite else float("nan")
        zero_loss = loss_is_finite and loss_value == 0.0
        self._zero_loss_streak = self._zero_loss_streak + 1 if zero_loss else 0
        diag.update(
            {
                "loss": loss_value,
                "loss_is_finite": loss_is_finite,
                "zero_loss_streak": self._zero_loss_streak,
            }
        )

        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
        else:
            logits = getattr(outputs, "logits", None)
        if logits is not None:
            logits_detached = logits.detach().float()
            logits_is_finite = bool(torch.isfinite(logits_detached).all().item())
            diag.update(
                {
                    "logits_is_finite": logits_is_finite,
                    "logits_abs_max": float(logits_detached.abs().max().item()) if logits_is_finite else float("nan"),
                }
            )
        else:
            diag["logits_is_finite"] = True
            diag["logits_abs_max"] = float("nan")

        diag["anomaly"] = any(
            (
                not diag["loss_is_finite"],
                not diag["logits_is_finite"],
                diag.get("valid_label_tokens", 1) == 0,
                diag.get("zero_label_examples", 0) > 0,
                diag["zero_loss_streak"] >= 3,
            )
        )
        diag["emit"] = diag["anomaly"] or self._periodic_should_log()
        return diag

    def _format_forward_diagnostics(self, diag: Dict[str, Any]) -> str:
        prefix = "[diag][anomaly]" if diag["anomaly"] else "[diag]"
        return (
            f"{prefix} rank={diag['rank']} local_rank={diag['local_rank']} "
            f"global_step={diag['global_step']} forward_call={diag['forward_call']} "
            f"loss={diag['loss']:.6f} finite_loss={diag['loss_is_finite']} "
            f"finite_logits={diag['logits_is_finite']} logits_abs_max={diag['logits_abs_max']:.4f} "
            f"valid_label_tokens={diag.get('valid_label_tokens', -1)} "
            f"zero_label_examples={diag.get('zero_label_examples', -1)} "
            f"label_tokens[min/mean/max]={diag.get('min_label_tokens', -1)}/"
            f"{diag.get('mean_label_tokens', float('nan')):.2f}/{diag.get('max_label_tokens', -1)} "
            f"input_tokens[min/mean/max]={diag.get('min_input_tokens', -1)}/"
            f"{diag.get('mean_input_tokens', float('nan')):.2f}/{diag.get('max_input_tokens', -1)} "
            f"zero_loss_streak={diag['zero_loss_streak']}"
        )

    def _summarize_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        grad_params = 0
        nonfinite_grad_params = 0
        max_abs_grad = 0.0
        grad_norm_sq = 0.0

        for _, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue

            grad = param.grad.detach().float()
            grad_params += 1
            if not torch.isfinite(grad).all():
                nonfinite_grad_params += 1
                continue

            max_abs_grad = max(max_abs_grad, float(grad.abs().max().item()))
            grad_norm_sq += float(torch.sum(grad * grad).item())

        grad_norm = math.sqrt(grad_norm_sq) if grad_norm_sq > 0 else 0.0
        return {
            "grad_params": grad_params,
            "nonfinite_grad_params": nonfinite_grad_params,
            "grad_abs_max": max_abs_grad,
            "grad_norm": grad_norm,
        }

    def _should_emit_gradients(self, diag: Dict[str, Any]) -> bool:
        return diag["anomaly"] or self._periodic_should_log()

    def _format_grad_diagnostics(self, grad_diag: Dict[str, Any], diag: Dict[str, Any]) -> str:
        prefix = "[grad][anomaly]" if grad_diag["nonfinite_grad_params"] > 0 else "[grad]"
        return (
            f"{prefix} rank={diag['rank']} local_rank={diag['local_rank']} "
            f"global_step={diag['global_step']} forward_call={diag['forward_call']} "
            f"grad_params={grad_diag['grad_params']} "
            f"nonfinite_grad_params={grad_diag['nonfinite_grad_params']} "
            f"grad_norm={grad_diag['grad_norm']:.6f} grad_abs_max={grad_diag['grad_abs_max']:.6f}"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        diag = self._summarize_forward_diagnostics(inputs, outputs, loss)
        self._last_forward_diagnostics = diag

        if diag["emit"]:
            if diag["anomaly"] or diag["rank"] == 0:
                print(self._format_forward_diagnostics(diag), flush=True)

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        diag = self._last_forward_diagnostics
        if diag is None or not self._should_emit_gradients(diag):
            return loss

        grad_diag = self._summarize_gradients(model)
        if diag["anomaly"] or grad_diag["nonfinite_grad_params"] > 0 or diag["rank"] == 0:
            print(self._format_grad_diagnostics(grad_diag, diag), flush=True)
        return loss


def print_full_finetune_trainable_params(model):
    trainable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {total_params:,} "
        f"|| trainable%: {100 * trainable_params / total_params:.6f}"
    )


def _checkpoint_step(checkpoint_path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", checkpoint_path.name)
    return int(match.group(1)) if match else -1


def _normalize_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        if checkpoint_path.name in {"trainer_state.json", "latest"}:
            return checkpoint_path.parent
    return checkpoint_path


def _is_resumable_checkpoint(checkpoint_path: Path) -> bool:
    checkpoint_path = _normalize_checkpoint_path(checkpoint_path)
    return checkpoint_path.is_dir() and (
        (checkpoint_path / "trainer_state.json").exists() or (checkpoint_path / "latest").exists()
    )


def _find_latest_resumable_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint is not None:
        checkpoint_path = Path(last_checkpoint)
        if _is_resumable_checkpoint(checkpoint_path):
            return checkpoint_path

    checkpoint_dirs = [
        path for path in output_dir.glob("checkpoint-*")
        if _is_resumable_checkpoint(path)
    ]
    if not checkpoint_dirs:
        return None
    return max(checkpoint_dirs, key=_checkpoint_step)


def _resolve_resume_checkpoint(output_dir: Path, resume_request: Optional[str]) -> Optional[Path]:
    if resume_request is None:
        return None

    request = str(resume_request).strip()
    if request == "" or request.lower() in {"false", "off", "none", "no"}:
        return None
    if request.lower() in {"true", "auto", "latest"}:
        return _find_latest_resumable_checkpoint(output_dir)

    checkpoint_path = _normalize_checkpoint_path(Path(request).expanduser())
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if not _is_resumable_checkpoint(checkpoint_path):
        raise ValueError(f"resume checkpoint not found or not resumable: {checkpoint_path}")
    return checkpoint_path


def _load_checkpoint_state(checkpoint_path: Path) -> Optional[dict]:
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if not trainer_state_path.exists():
        return None
    return json.loads(trainer_state_path.read_text())


def _is_checkpoint_complete(checkpoint_path: Path, num_train_epochs: float) -> bool:
    trainer_state = _load_checkpoint_state(checkpoint_path)
    if trainer_state is None:
        return False

    epoch = trainer_state.get("epoch")
    if epoch is None:
        return False
    return float(epoch) >= float(num_train_epochs)


def _final_artifacts_exist(output_dir: Path, use_lora: bool) -> bool:
    if use_lora:
        return (output_dir / "adapter_config.json").exists() and (output_dir / "adapter_model.safetensors").exists()

    if not (output_dir / "config.json").exists():
        return False
    return any(output_dir.glob("model*.safetensors")) or (output_dir / "pytorch_model.bin").exists()


def main():
    accelerator = Accelerator()
    parser = HfArgumentParser((LoRAArguments, TrainingArguments, DataArguments))
    lora_args, train_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "dataset"

    model = AutoModelForCausalLM.from_pretrained(
        train_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(
        train_args.model_name_or_path,
        model_max_length=train_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            added_pad_token = True
    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # `datasets.map()` writes cache files under the dataset directory. In DDP,
    # letting every rank build those caches concurrently can race on the same
    # `cache-*.arrow` path and leave non-main ranks reading a file that doesn't
    # exist yet. Build/cache on the main process first, then let other ranks
    # reuse the completed cache.
    with accelerator.main_process_first():
        dataset, dataset_name = build_dataset(data_args, dataset_root, train_args.seed)
    if accelerator.is_main_process:
        print(f"dataset={dataset_name}, samples={len(dataset)}")
        print(dataset[0])

    preprocessing_num_workers = max(1, min(8, os.cpu_count() or 1))
    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            fn_kwargs={
                "tokenizer": tokenizer,
                "model_max_length": train_args.model_max_length,
            },
            num_proc=preprocessing_num_workers,
        )

    use_lora = lora_args.lora_rank is not None
    train_mode = "lora" if use_lora else "sft"
    output_dir = Path(train_args.output_dir)
    resume_checkpoint = _resolve_resume_checkpoint(output_dir, train_args.resume_from_checkpoint)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME") or f"{train_mode}_{dataset_name}"
    diagnostic_interval = int(os.environ.get("TRAIN_DIAGNOSTIC_INTERVAL", "100"))
    diagnostic_early_steps = int(os.environ.get("TRAIN_DIAGNOSTIC_EARLY_STEPS", "3"))

    if use_lora:
        peft_config = LoraConfig(
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha if lora_args.lora_alpha is not None else 16,
            target_modules=lora_args.target_modules or ["q_proj", "v_proj"],
            lora_dropout=lora_args.lora_dropout if lora_args.lora_dropout is not None else 0.1,
            task_type=lora_args.task_type,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print_full_finetune_trainable_params(model)

    hf_train_args = transformers.TrainingArguments(
        num_train_epochs=train_args.num_train_epochs,
        output_dir=train_args.output_dir,
        run_name=wandb_run_name,
        learning_rate=train_args.learning_rate,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        warmup_ratio=train_args.warmup_ratio,
        weight_decay=train_args.weight_decay,
        save_strategy=train_args.save_strategy,
        report_to="wandb" if accelerator.is_main_process else None,
        label_names=["labels"],
        warmup_steps=train_args.warmup_steps,
        logging_steps=1,
        bf16=train_args.bf16,
        fp16=train_args.fp16,
        lr_scheduler_type=train_args.lr_scheduler_type,
        save_total_limit=train_args.save_total_limit,
        deepspeed=train_args.deepspeed,
        ddp_find_unused_parameters=False,
    )

    if accelerator.is_main_process:
        if resume_checkpoint is None:
            print("resume_from_checkpoint = None")
        else:
            trainer_state = _load_checkpoint_state(resume_checkpoint)
            progress = ""
            if trainer_state is not None:
                progress = (
                    f", global_step={trainer_state.get('global_step')}, "
                    f"epoch={trainer_state.get('epoch')}"
                )
            print(f"resume_from_checkpoint = {resume_checkpoint}{progress}")
        print(
            f"diagnostics: interval={diagnostic_interval}, "
            f"early_steps={diagnostic_early_steps}"
        )

        wandb.init(
            project=f"{train_mode.upper()}_{dataset_name}",
            name=wandb_run_name,
            config={
                "mode": train_mode,
                "dataset": dataset_name,
                "model_name_or_path": train_args.model_name_or_path,
                "model_max_length": train_args.model_max_length,
                "learning_rate": train_args.learning_rate,
                "num_train_epochs": train_args.num_train_epochs,
                "per_device_train_batch_size": train_args.per_device_train_batch_size,
                "gradient_accumulation_steps": train_args.gradient_accumulation_steps,
                "max_samples": data_args.max_samples,
                "lora_rank": lora_args.lora_rank,
                "lora_alpha": lora_args.lora_alpha,
                "lora_dropout": lora_args.lora_dropout,
                "target_modules": lora_args.target_modules,
            },
        )

    trainer = DiagnosticTrainer(
        model=model,
        args=hf_train_args,
        train_dataset=tokenized_dataset,
        diagnostic_interval=diagnostic_interval,
        diagnostic_early_steps=diagnostic_early_steps,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest",
        ),
    )

    if (
        resume_checkpoint is not None
        and _is_checkpoint_complete(resume_checkpoint, train_args.num_train_epochs)
        and _final_artifacts_exist(output_dir, use_lora)
    ):
        if accelerator.is_main_process:
            print(
                f"checkpoint {resume_checkpoint} already reached num_train_epochs={train_args.num_train_epochs}; "
                "final artifacts exist in output_dir, skipping train()."
            )
        accelerator.wait_for_everyone()
        return

    if resume_checkpoint is not None:
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()

    trainer.save_model(hf_train_args.output_dir)
    tokenizer.save_pretrained(hf_train_args.output_dir)


if __name__ == "__main__":
    main()
