from typing import Optional, Dict, Sequence, List
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class ReftArguments:

    position: str = field(
        default="f5+l5",
        metadata={
            "help": "positions of the tokens to be intervened"
        }
    )
    target_layers: List[int] = field(
        default_factory=lambda: [-1],
        metadata={
            "help": "Layers to be intervened. -1 means all layers. eg: --target_layers 10 12 14 16 18 20"
        }
    )
    subspace_rank: int = field(
        default=4,
        metadata={
            "help": "rank of the subspace to be intervened"
        }
    )

@dataclass
class DataArguments:

    percentage: float = field(
        default=1.0,
        metadata={
            "help": "percentage of the dataset to be used"
        }
    )


    max_samples: int = field(
        default=None,
        metadata={
            "help": "Maximum number of samples to be used."
        }
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):

    model_name_or_path: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={
            "help": "model name or path"
        }
    )

    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # 为 wandb设置不同名字的任务
    # run_name: str = field(
    #     default="",
    #     metadata={
    #         "help": "run name"
    #     }
    # )

    dropout: float = field(
        default=0.0,
        metadata={
            "help": "drop out rate"
        }
    )

    
