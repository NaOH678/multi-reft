from dataclasses import dataclass
from typing import Optional, List
from transformers import TrainingArguments

@dataclass
class TrainingArguments(TrainingArguments):
    model_name_or_path: str = "yahma/llama-7b-hf"
    model_max_length: int = 768
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
    dataset_name: str = "combined"  # 可选: truthful, helpful, moral, safety, stereotype, toxic, combined
    max_samples: Optional[int] = None
    percentage: Optional[float] = None

@dataclass
class LoRAArguments:
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = None  # 需要根据模型结构调整
    task_type: str = "CAUSAL_LM"
