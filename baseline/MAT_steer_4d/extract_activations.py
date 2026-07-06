from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PYVENE_ROOT = PROJECT_ROOT / "pyvene"
if str(PYVENE_ROOT) not in sys.path:
    sys.path.insert(0, str(PYVENE_ROOT))

import pyvene as pv

from .data import TASK_ORDER


class Collector:
    collect_state = True
    collect_action = False

    def __init__(self) -> None:
        self.states = []

    def reset(self) -> None:
        self.states = []

    def __call__(self, b, s):
        self.states.append(b.detach().clone())
        return b


def wrapper(intervener):
    def wrapped(*args, **kwargs):
        return intervener(*args, **kwargs)

    return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract MAT-Steer activations for four tasks")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--task_name", choices=list(TASK_ORDER), required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "features",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def normalize_model_name(base_model: str) -> str:
    name = str(base_model).rstrip("/").split("/")[-1]
    return name.replace(".", "_").replace("-", "_")


def load_json_rows(path: Path, max_samples: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if max_samples is not None:
        rows = rows[: max_samples]
    return rows


def main() -> None:
    args = parse_args()
    task_path = args.data_dir / f"{args.task_name}.json"
    rows = load_json_rows(task_path, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    collector = Collector()
    collected_model = pv.IntervenableModel(
        [
            {
                "component": f"model.layers[{args.layer}].self_attn.o_proj.input",
                "intervention": wrapper(collector),
            }
        ],
        model,
    )

    layer_activations = []
    labels = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"extract:{args.task_name}"):
        batch_rows = rows[start : start + args.batch_size]
        texts = [row["text"] for row in batch_rows]
        batch_labels = [int(row["label"]) for row in batch_rows]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(args.device)
        with torch.no_grad():
            collected_model({"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")})
        state_tensor = collector.states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            last_positions = torch.full(
                (state_tensor.shape[0],),
                state_tensor.shape[1] - 1,
                device=state_tensor.device,
                dtype=torch.long,
            )
        else:
            last_positions = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(state_tensor.shape[0], device=state_tensor.device)
        last_token_states = state_tensor[batch_indices, last_positions].float().cpu().numpy()
        layer_activations.extend(last_token_states)
        labels.extend(batch_labels)
        collector.reset()

    features_dir = args.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)
    model_name = normalize_model_name(args.base_model)
    np.save(features_dir / f"{model_name}_{args.task_name}_labels.npy", np.asarray(labels, dtype=np.int32))
    np.save(
        features_dir / f"{model_name}_{args.task_name}_layer_wise.npy",
        np.asarray(layer_activations, dtype=np.float32),
    )
    print(f"saved {len(labels)} activations for {args.task_name} to {features_dir}")


if __name__ == "__main__":
    main()
