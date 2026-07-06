#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge toxicity eval summary with auxiliary artifacts.")
    parser.add_argument("--summary_json", type=str, required=True, help="Base toxicity summary JSON.")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output path. Defaults to overwriting summary_json.",
    )
    parser.add_argument(
        "--perplexity_json",
        type=str,
        default=None,
        help="Optional perplexity/statistics JSON produced by compute_perplexity.py.",
    )
    parser.add_argument(
        "--judge_summary_json",
        type=str,
        default=None,
        help="Optional judge_summary.json produced by summarize_llm_judge.py.",
    )
    parser.add_argument(
        "--audit_summary_json",
        type=str,
        default=None,
        help="Optional audit_summary.json produced by build_llm_judge_audit.py.",
    )
    parser.add_argument(
        "--judge_method",
        type=str,
        default=None,
        choices=["ours", "sft", "lora"],
        help="Method key used in judge_summary absolute results.",
    )
    parser.add_argument(
        "--allow_input_mismatch",
        action="store_true",
        help="Allow perplexity input_path to differ from summary scored_csv/results_csv.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}")
    return payload


def sanitize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: sanitize_json_like(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json_like(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    return value


def infer_judge_method(summary: dict[str, Any]) -> str:
    model_tag = str(summary.get("model_tag") or "").lower()
    if "lora" in model_tag:
        return "lora"
    if "sft" in model_tag:
        return "sft"
    return "ours"


def maybe_verify_perplexity_input(
    summary: dict[str, Any],
    perplexity_payload: dict[str, Any],
    allow_input_mismatch: bool,
) -> None:
    input_path = perplexity_payload.get("input_path")
    if not input_path:
        return

    expected_paths = {
        str(summary.get("scored_csv") or "").strip(),
        str(summary.get("results_csv") or "").strip(),
    }
    expected_paths.discard("")
    if not expected_paths:
        return

    if str(input_path) not in expected_paths and not allow_input_mismatch:
        raise ValueError(
            "Perplexity input_path does not match summary results/scored csv: "
            f"input_path={input_path}, expected_one_of={sorted(expected_paths)}"
        )


def build_auxiliary_metrics(perplexity_payload: dict[str, Any]) -> dict[str, Any]:
    excluded = {"input_path"}
    return {k: v for k, v in perplexity_payload.items() if k not in excluded}


def build_audit_metadata(audit_payload: dict[str, Any]) -> dict[str, Any]:
    keys = ["config", "sources", "input_stats", "subset_sizes", "judge_record_counts"]
    return {key: audit_payload.get(key) for key in keys if key in audit_payload}


def index_absolute_by_split(judge_payload: dict[str, Any], judge_method: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in judge_payload.get("absolute", []) or []:
        if row.get("method") != judge_method:
            continue
        split = row.get("audit_split")
        if split:
            indexed[str(split)] = row
    return indexed


def index_pairwise_by_comparison(judge_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in judge_payload.get("pairwise", []) or []:
        comparison = row.get("comparison")
        if comparison:
            indexed[str(comparison)] = row
    return indexed


def build_llm_judge_block(
    judge_payload: dict[str, Any],
    judge_method: str,
    judge_summary_path: str,
    audit_summary_path: str | None,
    audit_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "available": True,
        "method": judge_method,
        "judge_summary_path": judge_summary_path,
        "audit_summary_path": audit_summary_path,
        "num_total_records": judge_payload.get("num_total_records"),
        "num_ok_records": judge_payload.get("num_ok_records"),
        "overall_parse_success_rate": judge_payload.get("overall_parse_success_rate"),
        "file_level": judge_payload.get("file_level", []),
        "absolute_by_split": index_absolute_by_split(judge_payload, judge_method),
    }

    if judge_method == "ours":
        pairwise = index_pairwise_by_comparison(judge_payload)
        block["pairwise_by_comparison"] = {
            key: value
            for key, value in pairwise.items()
            if key in {"ours_vs_sft", "ours_vs_lora"}
        }
    else:
        block["pairwise_by_comparison"] = {}

    if audit_payload is not None:
        block["audit_metadata"] = build_audit_metadata(audit_payload)

    return block


def merge_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = Path(args.summary_json)
    output_path = Path(args.output_json) if args.output_json else summary_path

    summary = load_json(summary_path)
    summary["artifact_paths"] = dict(summary.get("artifact_paths") or {})
    summary["summary_enriched_at"] = datetime.now(timezone.utc).isoformat()

    if args.perplexity_json:
        perplexity_path = Path(args.perplexity_json)
        perplexity_payload = load_json(perplexity_path)
        maybe_verify_perplexity_input(summary, perplexity_payload, args.allow_input_mismatch)
        summary["artifact_paths"]["perplexity_json"] = str(perplexity_path)
        summary["auxiliary_metrics"] = build_auxiliary_metrics(perplexity_payload)

    audit_payload = None
    if args.audit_summary_json:
        audit_path = Path(args.audit_summary_json)
        audit_payload = load_json(audit_path)
        summary["artifact_paths"]["audit_summary_json"] = str(audit_path)

    if args.judge_summary_json:
        judge_summary_path = Path(args.judge_summary_json)
        judge_payload = load_json(judge_summary_path)
        judge_method = args.judge_method or infer_judge_method(summary)
        summary["artifact_paths"]["judge_summary_json"] = str(judge_summary_path)
        summary["llm_judge"] = build_llm_judge_block(
            judge_payload=judge_payload,
            judge_method=judge_method,
            judge_summary_path=str(judge_summary_path),
            audit_summary_path=args.audit_summary_json,
            audit_payload=audit_payload,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = sanitize_json_like(summary)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "summary_json": str(summary_path),
        "output_json": str(output_path),
        "artifact_paths": summary.get("artifact_paths", {}),
        "has_auxiliary_metrics": "auxiliary_metrics" in summary,
        "has_llm_judge": "llm_judge" in summary,
    }


def main() -> None:
    args = parse_args()
    merged = merge_artifacts(args)
    print(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
