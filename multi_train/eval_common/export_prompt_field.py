import argparse
import json
from pathlib import Path

from datasets import load_from_disk

PROMPT_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
%s

### Input:
%s

### Response:
"""

PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
%s

### Response:
"""


def parse_args():
    parser = argparse.ArgumentParser("Export one text field from a HuggingFace disk dataset.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="input")
    parser.add_argument("--instruction_column", type=str, default="instruction")
    parser.add_argument(
        "--export_mode",
        type=str,
        choices=["field", "reft_supervised_prompt"],
        default="reft_supervised_prompt",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strip", action="store_true")
    return parser.parse_args()


def build_reft_supervised_prompt(example, instruction_column: str, input_column: str) -> str:
    instruction = example.get(instruction_column)
    if instruction is None:
        raise ValueError(
            f"instruction_column `{instruction_column}` missing from example keys: {list(example.keys())}"
        )
    instruction = str(instruction)

    input_value = example.get(input_column, "")
    input_value = "" if input_value is None else str(input_value)

    if input_value == "":
        return PROMPT_NO_INPUT % instruction
    return PROMPT_INPUT % (instruction, input_value)


def main():
    args = parse_args()
    dataset_obj = load_from_disk(args.dataset_path)
    if hasattr(dataset_obj, "keys") and args.split in dataset_obj:
        dataset = dataset_obj[args.split]
    else:
        dataset = dataset_obj

    if args.export_mode == "field":
        if args.text_column not in dataset.column_names:
            raise ValueError(
                f"text_column `{args.text_column}` not found in {args.dataset_path}. "
                f"Available columns: {dataset.column_names}"
            )
    else:
        if args.instruction_column not in dataset.column_names:
            raise ValueError(
                f"instruction_column `{args.instruction_column}` not found in {args.dataset_path}. "
                f"Available columns: {dataset.column_names}"
            )
        if args.text_column not in dataset.column_names:
            raise ValueError(
                f"text_column `{args.text_column}` not found in {args.dataset_path}. "
                f"Available columns: {dataset.column_names}"
            )

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            if args.export_mode == "field":
                text = example.get(args.text_column)
                if text is None:
                    continue
                text = str(text)
            else:
                text = build_reft_supervised_prompt(
                    example,
                    instruction_column=args.instruction_column,
                    input_column=args.text_column,
                )
            if args.strip:
                text = text.strip()
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if args.max_samples is not None and count >= args.max_samples:
                break

    print(f"Exported {count} prompts to: {output_path}")


if __name__ == "__main__":
    main()
