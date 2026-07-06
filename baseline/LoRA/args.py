from dataclasses import dataclass
from typing import Optional, List
from transformers import TrainingArguments as HFTrainingArguments

@dataclass
class TrainingArguments(HFTrainingArguments):
    model_name_or_path: str = "yahma/llama-7b-hf"
    model_max_length: int = 512
    output_dir: str = "./lora_output"
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    dropout: float = 0.1
    save_strategy: str = "epoch"

@dataclass 
class DataArguments:
    dataset_name: str = "combined"  # 可选: truthful, helpful, moral, safety, stereotype, toxicity, combined
    max_samples: Optional[int] = None
    percentage: Optional[float] = None

@dataclass
class LoRAArguments:
    # 为空时表示不启用LoRA，走全参数SFT
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    target_modules: Optional[List[str]] = None  # 需要根据模型结构调整
    task_type: str = "CAUSAL_LM"
