from pathlib import Path
from typing import Optional

from multi_train.eval_common.output_naming import normalize_base_model_name


PROJECT_ROOT = Path(__file__).resolve().parents[2]


TRAIN_DATASETS = {
    "truth": PROJECT_ROOT / "dataset/caa/truth/train.json",
    "bias": PROJECT_ROOT / "dataset/caa/bias/train.json",
    "ethics": PROJECT_ROOT / "dataset/caa/ethics/train.json",
    "toxicity": PROJECT_ROOT / "dataset/caa/toxicity_10k/train.json",
}


TASK_INPUT_MODES = {
    "truth": "qa_response",
    "ethics": "qa_response",
    "bias": "raw_contrastive",
    "toxicity": "raw_contrastive",
}


TASK_SCHEMA_TYPES = {
    "truth": "prompt_plus_contrastive_responses",
    "ethics": "prompt_plus_contrastive_responses",
    "bias": "mixed_contrastive_sentences_and_prompt_responses",
    "toxicity": "contrastive_sentences_only",
}


TASK_ALIASES = {
    "truthful": "truth",
    "stereotype": "bias",
    "moral": "ethics",
    "toxic": "toxicity",
}


ITI_ROOT = PROJECT_ROOT / "baseline/iti_0"
ARTIFACT_ROOT = ITI_ROOT / "artifacts"
RUN_ROOT = ITI_ROOT / "runs"
INTERVENTION_ROOT = ARTIFACT_ROOT / "interventions"


DEFAULT_TOP_HEADS = 48
DEFAULT_VAL_RATIO = 0.2
DEFAULT_LOGISTIC_MAX_ITER = 1000
DEFAULT_SINGLE_LAYER_TASKS = {"truth", "bias", "ethics", "toxicity"}
MODEL_DEFAULT_SINGLE_LAYER_PATTERNS = [
    ("falcon-7b", 15),
    ("mistral-7b", 16),
    ("llama3.1-8b", 14),
    ("llama3-8b", 14),
    ("qwen2.5-7b", 14),
    ("qwen25-7b", 14),
]
DEFAULT_SINGLE_LAYER_FALLBACK = 14


def normalize_task_name(task: str) -> str:
    value = str(task or "").strip().lower()
    value = TASK_ALIASES.get(value, value)
    if value not in TRAIN_DATASETS:
        raise ValueError(f"Unsupported task={task}. Expected one of: {sorted(TRAIN_DATASETS)}")
    return value


def get_train_dataset_path(task: str) -> Path:
    return TRAIN_DATASETS[normalize_task_name(task)]


def get_task_input_mode(task: str) -> str:
    return TASK_INPUT_MODES[normalize_task_name(task)]


def get_task_schema_type(task: str) -> str:
    return TASK_SCHEMA_TYPES[normalize_task_name(task)]


def should_use_default_single_layer(task: str) -> bool:
    return normalize_task_name(task) in DEFAULT_SINGLE_LAYER_TASKS


def resolve_default_single_layer(base_model_path: str, task: Optional[str] = None) -> Optional[int]:
    if task is not None and not should_use_default_single_layer(task):
        return None

    model_name = normalize_base_model_name(base_model_path)
    for pattern, layer in MODEL_DEFAULT_SINGLE_LAYER_PATTERNS:
        if pattern in model_name:
            return int(layer)
    return int(DEFAULT_SINGLE_LAYER_FALLBACK)


def normalize_model_name(base_model_path: str) -> str:
    return normalize_base_model_name(base_model_path)
