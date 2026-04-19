import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_instruction(question: str, choices: list[str]) -> str:
    answer_lines = " ".join(
        [f"Answer{i + 1}: {choice.strip()}" for i, choice in enumerate(choices)]
    )
    answer_format = "/".join([f"answer{i + 1}" for i in range(len(choices))])
    return (
        f"Please choose the truthful answer to the question: {question.strip()}\n\n"
        f"{answer_lines}\n\n"
        f"Answer format: {answer_format}"
    )


def convert_truthfulqa_mc(split: str, target: str) -> tuple[list[dict], int]:
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    target_key = f"{target}_targets"
    rows = []
    skipped = 0

    for item in dataset:
        question = (item.get("question") or "").strip()
        targets = item.get(target_key)
        if not question or not isinstance(targets, dict):
            skipped += 1
            continue

        choices = targets.get("choices") or []
        labels = targets.get("labels") or []
        if not choices or len(choices) != len(labels):
            skipped += 1
            continue

        correct_ids = [idx for idx, label in enumerate(labels) if int(label) == 1]
        if len(correct_ids) != 1:
            skipped += 1
            continue

        correct = correct_ids[0]
        rows.append(
            {
                "instruction": build_instruction(question, choices),
                "input": "",
                "output": "",
                "answer": f"answer{correct + 1}",
            }
        )

    return rows, skipped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target", choices=["mc1", "mc2"], default="mc1")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = repo_root / "dataset" / "truthfulqa_mc" / "test.json"

    rows, skipped = convert_truthfulqa_mc(split=args.split, target=args.target)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(rows)} samples to: {output_path}")
    print(f"Skipped {skipped} samples due to malformed labels/choices.")


if __name__ == "__main__":
    main()
