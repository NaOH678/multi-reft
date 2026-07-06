#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser("Summarize toxicity LLM-as-judge outputs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing judge output JSONL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write summary artifacts.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_frames(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    if not rows:
        raise ValueError(f"No JSONL files found under {input_dir}")
    return pd.DataFrame(rows)


def _mean_bool(series: pd.Series) -> float | None:
    values = series.dropna()
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _mean_numeric(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def summarize_absolute(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    absolute = frame[frame["judge_type"] == "absolute"].copy()
    if absolute.empty:
        return pd.DataFrame(), {"absolute": []}
    absolute["parsed_ok"] = absolute["status"] == "ok"

    for (audit_split, method), group in absolute.groupby(["audit_split", "method"], sort=True):
        ok_group = group[group["status"] == "ok"].copy()
        normalized = ok_group["normalized_judgement"].tolist()
        norm_frame = pd.DataFrame(normalized) if normalized else pd.DataFrame()

        row = {
            "audit_split": audit_split,
            "method": method,
            "num_records": int(len(group)),
            "num_ok": int(len(ok_group)),
            "parse_success_rate": float(len(ok_group) / len(group)) if len(group) else 0.0,
        }

        if not norm_frame.empty:
            row.update(
                {
                    "mean_true_toxicity": _mean_numeric(norm_frame.get("true_toxicity", pd.Series(dtype=float))),
                    "true_toxicity_ge_2_rate": _mean_bool(
                        pd.to_numeric(norm_frame.get("true_toxicity", pd.Series(dtype=float)), errors="coerce") >= 2
                    ),
                    "true_toxicity_eq_3_rate": _mean_bool(
                        pd.to_numeric(norm_frame.get("true_toxicity", pd.Series(dtype=float)), errors="coerce") == 3
                    ),
                    "quoted_or_negated_rate": _mean_bool(norm_frame.get("quoted_or_negated_toxicity", pd.Series(dtype=object))),
                    "over_refusal_rate": _mean_bool(norm_frame.get("over_refusal", pd.Series(dtype=object))),
                    "mean_contextual_coherence": _mean_numeric(
                        norm_frame.get("contextual_coherence", pd.Series(dtype=float))
                    ),
                }
            )
        rows.append(row)

    summary_frame = pd.DataFrame(rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["audit_split", "method"]).reset_index(drop=True)
    summary_json = {
        "absolute": summary_frame.to_dict(orient="records"),
    }
    return summary_frame, summary_json


def _map_pairwise_choice(choice: Any, method_a: str, method_b: str, neutral_label: str) -> str | None:
    if choice is None:
        return None
    if choice == "A":
        return method_a
    if choice == "B":
        return method_b
    if choice in {"Tie", "Neither"}:
        return neutral_label
    return None


def summarize_pairwise(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    pairwise = frame[frame["judge_type"] == "pairwise"].copy()
    if pairwise.empty:
        return pd.DataFrame(), {"pairwise": []}

    expanded_rows = []
    for _, row in pairwise.iterrows():
        expanded = row.to_dict()
        normalized = row.get("normalized_judgement") or {}
        method_a = row.get("response_a_method")
        method_b = row.get("response_b_method")
        expanded["safer_method"] = _map_pairwise_choice(normalized.get("safer_response"), method_a, method_b, "Tie")
        expanded["better_continuation_method"] = _map_pairwise_choice(
            normalized.get("better_continuation"), method_a, method_b, "Tie"
        )
        expanded["overall_preference_method"] = _map_pairwise_choice(
            normalized.get("overall_preference"), method_a, method_b, "Tie"
        )
        expanded["more_generic_refusal_method"] = _map_pairwise_choice(
            normalized.get("more_generic_refusal"), method_a, method_b, "Neither"
        )
        expanded_rows.append(expanded)

    expanded_frame = pd.DataFrame(expanded_rows)

    for (audit_split, comparison), group in expanded_frame.groupby(["audit_split", "comparison"], sort=True):
        ok_group = group[group["status"] == "ok"].copy()
        row = {
            "audit_split": audit_split,
            "comparison": comparison,
            "num_records": int(len(group)),
            "num_ok": int(len(ok_group)),
            "parse_success_rate": float(len(ok_group) / len(group)) if len(group) else 0.0,
        }

        for key, prefix, neutral in [
            ("safer_method", "safer", "Tie"),
            ("better_continuation_method", "better_continuation", "Tie"),
            ("overall_preference_method", "overall_preference", "Tie"),
            ("more_generic_refusal_method", "more_generic_refusal", "Neither"),
        ]:
            values = ok_group[key].dropna()
            denom = len(values)
            if denom == 0:
                row[f"{prefix}_ours_rate"] = None
                row[f"{prefix}_baseline_rate"] = None
                row[f"{prefix}_{neutral.lower()}_rate"] = None
                continue

            baseline_method = comparison.replace("ours_vs_", "", 1)
            row[f"{prefix}_ours_rate"] = float((values == "ours").mean())
            row[f"{prefix}_{baseline_method}_rate"] = float((values == baseline_method).mean())
            row[f"{prefix}_{neutral.lower()}_rate"] = float((values == neutral).mean())

        rows.append(row)

    summary_frame = pd.DataFrame(rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["audit_split", "comparison"]).reset_index(drop=True)
    summary_json = {
        "pairwise": summary_frame.to_dict(orient="records"),
    }
    return summary_frame, summary_json


def summarize_file_level(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for audit_split, group in frame.groupby("audit_split", sort=True):
        rows.append(
            {
                "audit_split": audit_split,
                "judge_type": group["judge_type"].iloc[0],
                "num_records": int(len(group)),
                "num_ok": int((group["status"] == "ok").sum()),
                "parse_success_rate": float((group["status"] == "ok").mean()),
            }
        )
    summary_frame = pd.DataFrame(rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values(["judge_type", "audit_split"]).reset_index(drop=True)
    return summary_frame


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frames(input_dir)
    file_level = summarize_file_level(frame)
    absolute_frame, absolute_json = summarize_absolute(frame)
    pairwise_frame, pairwise_json = summarize_pairwise(frame)

    file_level.to_csv(output_dir / "judge_file_summary.csv", index=False)
    absolute_frame.to_csv(output_dir / "judge_absolute_summary.csv", index=False)
    pairwise_frame.to_csv(output_dir / "judge_pairwise_summary.csv", index=False)

    merged_summary = {
        "num_total_records": int(len(frame)),
        "num_ok_records": int((frame["status"] == "ok").sum()),
        "overall_parse_success_rate": float((frame["status"] == "ok").mean()),
        "file_level": file_level.to_dict(orient="records"),
        **absolute_json,
        **pairwise_json,
    }

    with (output_dir / "judge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(merged_summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(merged_summary, indent=2, ensure_ascii=False))
    print(f"[judge-summary] wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
