from __future__ import annotations

import argparse
from pathlib import Path

from .data import TASK_ORDER, build_all_task_samples, save_task_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare four-task MAT-Steer json data")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASK_ORDER),
        choices=list(TASK_ORDER),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples_by_task = build_all_task_samples(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_name, samples in samples_by_task.items():
        output_path = args.output_dir / f"{task_name}.json"
        save_task_json(task_name, output_path, samples)
        print(f"{task_name}: saved {len(samples)} samples to {output_path}")


if __name__ == "__main__":
    main()
