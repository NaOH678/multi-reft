from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from .data import TASK_ORDER
from .steering import (
    SteeringModule,
    compute_mmd,
    normalize_activations,
    orthogonality_loss,
    preservation_loss,
    save_checkpoint,
    sparsity_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train four-task MAT-Steer baseline")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "features",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER), choices=list(TASK_ORDER))
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--lambda_sparse", type=float, default=0.9)
    parser.add_argument("--lambda_ortho", type=float, default=0.1)
    parser.add_argument("--lambda_pos", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_model_name(base_model: str) -> str:
    name = str(base_model).rstrip("/").split("/")[-1]
    return name.replace(".", "_").replace("-", "_")


def load_tasks(features_dir: Path, base_model: str, task_names: list[str]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    model_name = normalize_model_name(base_model)
    tasks = {}
    for task_name in task_names:
        labels = np.load(features_dir / f"{model_name}_{task_name}_labels.npy")
        activations = np.load(features_dir / f"{model_name}_{task_name}_layer_wise.npy")
        pos_acts = torch.tensor(activations[labels == 1], dtype=torch.float32)
        neg_acts = torch.tensor(activations[labels == 0], dtype=torch.float32)
        tasks[task_name] = (pos_acts, neg_acts)
        print(f"{task_name}: positive={pos_acts.shape[0]} negative={neg_acts.shape[0]}")
    return tasks


def train_model(args: argparse.Namespace, tasks: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> SteeringModule:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    task_names = list(tasks.keys())
    input_dim = list(tasks.values())[0][0].shape[1]
    model = SteeringModule(input_dim=input_dim, num_attributes=len(task_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        min_samples = min(min(pos.shape[0], neg.shape[0]) for pos, neg in tasks.values())
        effective_batch_size = min(args.batch_size // (2 * len(task_names)), min_samples)
        if effective_batch_size < 1:
            effective_batch_size = 1

        for _ in range(0, min_samples, effective_batch_size):
            batch_activations = []
            batch_labels = []
            batch_task_indices = []
            for task_idx, task_name in enumerate(task_names):
                pos_acts, neg_acts = tasks[task_name]
                pos_indices = torch.randperm(pos_acts.shape[0])[:effective_batch_size]
                neg_indices = torch.randperm(neg_acts.shape[0])[:effective_batch_size]
                batch_activations.extend([
                    pos_acts[pos_indices].to(device),
                    neg_acts[neg_indices].to(device),
                ])
                batch_labels.extend([1] * effective_batch_size)
                batch_labels.extend([0] * effective_batch_size)
                batch_task_indices.extend([task_idx] * (effective_batch_size * 2))

            activations = torch.cat(batch_activations, dim=0)
            labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
            task_indices = torch.tensor(batch_task_indices, dtype=torch.long, device=device)

            adjusted, gates = model(activations)
            adjusted = normalize_activations(activations, adjusted)

            loss_mmd = torch.tensor(0.0, device=device)
            for task_idx, task_name in enumerate(task_names):
                task_mask = task_indices == task_idx
                if task_mask.sum() == 0:
                    continue
                task_acts = adjusted[task_mask]
                task_labels = labels[task_mask]
                pos_mask = task_labels == 1
                neg_mask = task_labels == 0
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue
                neg_adjusted = task_acts[neg_mask]
                original_pos = tasks[task_name][0]
                sample_indices = torch.randperm(original_pos.shape[0])[: min(neg_adjusted.shape[0], original_pos.shape[0])]
                original_pos_sample = original_pos[sample_indices].to(device)
                loss_mmd = loss_mmd + compute_mmd(neg_adjusted, original_pos_sample, args.sigma)
            loss_mmd = loss_mmd / len(task_names)

            neg_mask = labels == 0
            pos_mask = labels == 1
            loss_sparse = sparsity_loss(gates[neg_mask]) if neg_mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss_pos = preservation_loss(gates[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=device)
            loss_ortho = orthogonality_loss(model.steering_vectors)
            batch_loss = (
                args.lambda_mmd * loss_mmd
                + args.lambda_sparse * loss_sparse
                + args.lambda_ortho * loss_ortho
                + args.lambda_pos * loss_pos
            )

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())
            num_batches += 1

        if epoch % 10 == 0:
            print(f"epoch={epoch} avg_loss={epoch_loss / max(num_batches, 1):.6f}")

    return model


def main() -> None:
    args = parse_args()
    task_names = list(args.tasks)
    tasks = load_tasks(args.features_dir, args.base_model, task_names)
    model = train_model(args, tasks)
    model_name = normalize_model_name(args.base_model)
    save_path = args.save_path
    if save_path is None:
        save_path = Path(__file__).resolve().parent / "checkpoints" / f"{model_name}_L{args.layer}_mat4d.pt"
    save_checkpoint(
        model,
        save_path,
        input_dim=list(tasks.values())[0][0].shape[1],
        task_names=task_names,
        layer=args.layer,
        model_name=args.base_model,
        hyperparams={
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "device": args.device,
            "lr": args.lr,
            "sigma": args.sigma,
            "lambda_mmd": args.lambda_mmd,
            "lambda_sparse": args.lambda_sparse,
            "lambda_ortho": args.lambda_ortho,
            "lambda_pos": args.lambda_pos,
        },
    )
    print(f"saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
