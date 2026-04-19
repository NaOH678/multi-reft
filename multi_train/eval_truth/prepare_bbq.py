import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]


def parse_categories(categories: str) -> list[str]:
    if not categories or categories.strip().lower() == "all":
        return DEFAULT_CATEGORIES

    items = [item.strip() for item in categories.split(",") if item.strip()]
    unknown = sorted(set(items) - set(DEFAULT_CATEGORIES))
    if unknown:
        raise ValueError(
            f"Unknown categories: {unknown}. Valid categories: {DEFAULT_CATEGORIES}"
        )
    return items


def build_instruction(context: str, question: str, choices: list[str]) -> str:
    answer_lines = " ".join(
        [f"Answer{i + 1}: {choice.strip()}" for i, choice in enumerate(choices)]
    )
    answer_format = "/".join([f"answer{i + 1}" for i in range(len(choices))])
    return (
        "Please choose the correct answer to the question based on the context:\n"
        f"{context.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        f"{answer_lines}\n\n"
        f"Answer format: {answer_format}"
    )


def convert_bbq(
    categories: list[str],
    context_condition: str,
    question_polarity: str,
) -> tuple[list[dict], int]:
    rows = []
    skipped = 0

    for category in categories:
        file_path = hf_hub_download(
            repo_id="heegyu/bbq",
            filename=f"data/{category}.jsonl",
            repo_type="dataset",
        )
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                if (
                    context_condition != "all"
                    and item.get("context_condition") != context_condition
                ):
                    continue
                if (
                    question_polarity != "all"
                    and item.get("question_polarity") != question_polarity
                ):
                    continue

                context = (item.get("context") or "").strip()
                question = (item.get("question") or "").strip()
                choices = [item.get("ans0"), item.get("ans1"), item.get("ans2")]
                if not context or not question:
                    skipped += 1
                    continue
                if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
                    skipped += 1
                    continue

                label = item.get("label")
                try:
                    label_idx = int(label)
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                if label_idx < 0 or label_idx >= len(choices):
                    skipped += 1
                    continue

                rows.append(
                    {
                        "instruction": build_instruction(context, question, choices),
                        "input": "",
                        "output": "",
                        "answer": f"answer{label_idx + 1}",
                    }
                )

    return rows, skipped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, default="all")
    parser.add_argument(
        "--context_condition",
        choices=["all", "ambig", "disambig"],
        default="all",
    )
    parser.add_argument(
        "--question_polarity",
        choices=["all", "neg", "nonneg"],
        default="all",
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    categories = parse_categories(args.categories)
    repo_root = Path(__file__).resolve().parents[2]
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = repo_root / "dataset" / "bbq" / "test.json"

    rows, skipped = convert_bbq(
        categories=categories,
        context_condition=args.context_condition,
        question_polarity=args.question_polarity,
    )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(rows)} samples to: {output_path}")
    print(f"Skipped {skipped} malformed samples.")
    print(f"Categories: {', '.join(categories)}")
    print(f"context_condition={args.context_condition}, question_polarity={args.question_polarity}")


if __name__ == "__main__":
    main()
