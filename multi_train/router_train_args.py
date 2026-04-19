from typing import Optional, Dict, Sequence, List
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class RouterTrainArguments(transformers.TrainingArguments):
    num_total_subspaces: int = field(
        default=1, metadata={"help": "Total number of available subspaces"}
    )
    subspace_rank: int = field(
        default=1, metadata={"help": "Rank of the subspaces"}
    )
    model_name_or_path: str = field(
        default="yahma/llama-7b-hf", metadata={"help": "Model name or path"}
    )
    max_length: int = field(
        default=768, metadata={"help": "Maximum length of the input sequence"}
    )
    position: str = field(
        default="f5+l5",
        metadata={
            "help": "positions of the tokens to be intervened"
        }
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "drop out rate"
        }
    )

@dataclass
class RouterDataArguments:
    max_examples: int = field(
        default=None, metadata={"help": "Maximum number of examples to use"}
    )
    dataset_name: str = field(
        default="helpful", metadata={"help": "Name of the dataset"}
    )
    ratio: int = field(
        default=None, metadata={"help": "Ratio of the training data to use"}
    )

@dataclass
class RouterModelArguments:
    reft_weight_path: str = field(
        default=None, metadata={"help": "Path to the reft weight"}
    )

    



