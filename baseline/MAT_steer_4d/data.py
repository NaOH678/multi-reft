from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_MAP = {
    "truthfulqa": PROJECT_ROOT / "dataset" / "alignment_truthful_format",
    "bbq": PROJECT_ROOT / "dataset" / "alignment_stereotype_format",
    "ethics": PROJECT_ROOT / "dataset" / "alignment_moral_cls",
    "toxicity": PROJECT_ROOT / "dataset" / "alignment_toxic_format",
}
TASK_ORDER = ("truthfulqa", "bbq", "ethics", "toxicity")
TRUTHFUL_TRIGGER = "the correct answer is "
ANSWER_FORMAT_RE = re.compile(r"answer format\s*:\s*([^\n\r]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ContrastiveSample:
    text: str
    label: int
    task: str


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _dataset_to_iterable(path: Path):
    loaded = load_from_disk(str(path))
    if hasattr(loaded, "keys") and "train" in loaded:
        return loaded["train"]
    return loaded


def _extract_candidate_labels(instruction: str) -> list[str]:
    match = ANSWER_FORMAT_RE.search(str(instruction or ""))
    if not match:
        return []
    raw = match.group(1).strip()
    parts = [normalize_text(part).lower() for part in raw.split("/") if normalize_text(part)]
    return [part for part in parts if part]


def _truthful_samples(dataset) -> list[ContrastiveSample]:
    rows: list[ContrastiveSample] = []
    for example in dataset:
        instruction = normalize_text(example.get("instruction", ""))
        output_value = normalize_text(example.get("output", "")).lower()
        if not instruction or not output_value:
            continue
        candidate_labels = _extract_candidate_labels(instruction)
        if not candidate_labels or output_value not in candidate_labels:
            continue
        for candidate in candidate_labels:
            label = int(candidate == output_value)
            rows.append(
                ContrastiveSample(
                    text=f"Q: {instruction} A: {TRUTHFUL_TRIGGER}{candidate}",
                    label=label,
                    task="truthfulqa",
                )
            )
    return rows


def _bbq_samples(dataset) -> list[ContrastiveSample]:
    rows: list[ContrastiveSample] = []
    for example in dataset:
        instruction = normalize_text(example.get("instruction", ""))
        output = normalize_text(example.get("output", ""))
        input_text = normalize_text(example.get("input", ""))
        if not instruction or not output or input_text:
            continue
        rows.append(ContrastiveSample(text=instruction, label=0, task="bbq"))
        rows.append(ContrastiveSample(text=output, label=1, task="bbq"))
    return rows


def _ethics_samples(dataset) -> list[ContrastiveSample]:
    rows: list[ContrastiveSample] = []
    for example in dataset:
        instruction = normalize_text(example.get("instruction", ""))
        input_text = normalize_text(example.get("input", ""))
        output = normalize_text(example.get("output", "")).lower()
        if not instruction or not input_text or output not in {"wrong", "not wrong"}:
            continue
        question = f"{instruction}\n\n{input_text}"
        rows.append(ContrastiveSample(text=f"{question}\n\nAnswer: wrong", label=int(output == "wrong"), task="ethics"))
        rows.append(
            ContrastiveSample(
                text=f"{question}\n\nAnswer: not wrong",
                label=int(output == "not wrong"),
                task="ethics",
            )
        )
    return rows


def _toxicity_samples(dataset) -> list[ContrastiveSample]:
    rows: list[ContrastiveSample] = []
    for example in dataset:
        instruction = normalize_text(example.get("instruction", ""))
        output = normalize_text(example.get("output", ""))
        input_text = normalize_text(example.get("input", ""))
        if not instruction or not output or input_text:
            continue
        rows.append(ContrastiveSample(text=instruction, label=0, task="toxicity"))
        rows.append(ContrastiveSample(text=output, label=1, task="toxicity"))
    return rows


TASK_BUILDERS = {
    "truthfulqa": _truthful_samples,
    "bbq": _bbq_samples,
    "ethics": _ethics_samples,
    "toxicity": _toxicity_samples,
}


def build_task_samples(
    task_name: str,
    dataset_path: Path | None = None,
) -> list[ContrastiveSample]:
    if task_name not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task_name={task_name}. Expected one of {TASK_ORDER}.")
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_MAP[task_name]
    dataset = _dataset_to_iterable(path)
    return TASK_BUILDERS[task_name](dataset)


def build_all_task_samples(task_names: Iterable[str] | None = None) -> dict[str, list[ContrastiveSample]]:
    resolved_task_names = list(task_names) if task_names is not None else list(TASK_ORDER)
    return {task_name: build_task_samples(task_name) for task_name in resolved_task_names}


def save_task_json(task_name: str, output_path: Path, samples: list[ContrastiveSample]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"text": sample.text, "label": sample.label, "task": sample.task} for sample in samples]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
