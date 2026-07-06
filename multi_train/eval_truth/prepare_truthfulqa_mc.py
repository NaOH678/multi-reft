import argparse
import hashlib
import json
import random
from pathlib import Path


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


def reorder_choices(
    question: str,
    choices: list[str],
    correct_idx: int,
    sample_idx: int,
    seed: int,
) -> tuple[list[str], int]:
    correct_choice = choices[correct_idx]
    incorrect_choices = [
        choice for idx, choice in enumerate(choices) if idx != correct_idx
    ]

    seed_material = f"{seed}\t{sample_idx}\t{question}".encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(rng_seed)
    rng.shuffle(incorrect_choices)

    new_correct_idx = (sample_idx + seed) % len(choices)
    reordered_choices = incorrect_choices[:]
    reordered_choices.insert(new_correct_idx, correct_choice)
    return reordered_choices, new_correct_idx


def convert_truthfulqa_mc(split: str, target: str, seed: int) -> tuple[list[dict], int]:
    from datasets import load_dataset

    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split=split)
    target_key = f"{target}_targets"
    rows = []
    skipped = 0

    for sample_idx, item in enumerate(dataset):
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

        reordered_choices, reordered_correct_idx = reorder_choices(
            question=question,
            choices=choices,
            correct_idx=correct_ids[0],
            sample_idx=sample_idx,
            seed=seed,
        )
        rows.append(
            {
                "instruction": build_instruction(question, reordered_choices),
                "input": "",
                "output": "",
                "answer": f"answer{reordered_correct_idx + 1}",
            }
        )

    return rows, skipped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation")
    parser.add_argument("--target", choices=["mc1", "mc2"], default="mc1")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = repo_root / "dataset" / "truthfulqa_mc" / "test.json"

    rows, skipped = convert_truthfulqa_mc(
        split=args.split,
        target=args.target,
        seed=args.seed,
    )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(rows)} samples to: {output_path}")
    print(f"Skipped {skipped} samples due to malformed labels/choices.")


if __name__ == "__main__":
    main()
