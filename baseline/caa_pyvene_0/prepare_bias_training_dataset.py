#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.truth_label_utils import expand_truth_response_with_choice_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand truth-style rows inside the bias training set so that "
            "`the correct answer is answerX` becomes "
            "`the correct answer is answerX. <choice text>`."
        )
    )
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_json = Path(args.source_json)
    output_json = Path(args.output_json)

    with source_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    transformed_rows = []
    truth_style_rows = 0
    expanded_positive = 0
    expanded_negative = 0

    for row in rows:
        updated = dict(row)
        question = str(updated.get("question", "") or "").strip()
        if question:
            truth_style_rows += 1
            old_positive = str(updated.get("answer_matching_behavior", "") or "")
            old_negative = str(updated.get("answer_not_matching_behavior", "") or "")
            new_positive = expand_truth_response_with_choice_text(question, old_positive)
            new_negative = expand_truth_response_with_choice_text(question, old_negative)
            if new_positive != old_positive:
                expanded_positive += 1
            if new_negative != old_negative:
                expanded_negative += 1
            updated["answer_matching_behavior"] = new_positive
            updated["answer_not_matching_behavior"] = new_negative
        transformed_rows.append(updated)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(transformed_rows, handle, ensure_ascii=False, indent=2)

    print(f"source_json={source_json}")
    print(f"output_json={output_json}")
    print(f"num_rows={len(transformed_rows)}")
    print(f"truth_style_rows={truth_style_rows}")
    print(f"expanded_positive={expanded_positive}")
    print(f"expanded_negative={expanded_negative}")


if __name__ == "__main__":
    main()
