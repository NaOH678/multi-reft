#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build merged toxicity summary from per-run summaries.")
    parser.add_argument("--merge_summary_path", type=str, required=True)
    parser.add_argument("--run_ts", type=str, required=True)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--prompts", nargs="*", default=[])
    parser.add_argument("--failed_runs", nargs="*", default=[])
    parser.add_argument("--summary_files", nargs="*", default=[])
    return parser.parse_args()


def run_key(run: dict[str, Any]) -> str:
    model_tag = str(run.get("model_tag") or "").strip()
    if model_tag:
        return model_tag
    return str(run.get("results_csv") or "unknown_target")


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_auxiliary_metrics(run: dict[str, Any]) -> dict[str, float | None]:
    aux = run.get("auxiliary_metrics") or {}
    return {
        "perplexity": maybe_float(aux.get("perplexity")),
        "distinct_1": maybe_float(aux.get("distinct_1")),
        "distinct_2": maybe_float(aux.get("distinct_2")),
        "refusal_combo_rate": maybe_float(aux.get("refusal_combo_rate")),
        "toxicity_gt_0p5_rate": maybe_float(aux.get("toxicity_gt_0p5_rate")),
        "toxicity_gt_0p8_rate": maybe_float(aux.get("toxicity_gt_0p8_rate")),
    }


def extract_judge_metrics(run: dict[str, Any]) -> dict[str, float | str | None]:
    judge = run.get("llm_judge") or {}
    absolute = judge.get("absolute_by_split") or {}
    high_risk = absolute.get("absolute_high_risk") or {}
    random_background = absolute.get("absolute_random_background") or {}
    return {
        "judge_method": judge.get("method"),
        "judge_parse_success_rate": maybe_float(judge.get("overall_parse_success_rate")),
        "judge_high_risk_mean_true_toxicity": maybe_float(high_risk.get("mean_true_toxicity")),
        "judge_high_risk_true_toxicity_ge_2_rate": maybe_float(high_risk.get("true_toxicity_ge_2_rate")),
        "judge_high_risk_quoted_or_negated_rate": maybe_float(high_risk.get("quoted_or_negated_rate")),
        "judge_random_background_over_refusal_rate": maybe_float(random_background.get("over_refusal_rate")),
        "judge_random_background_mean_contextual_coherence": maybe_float(
            random_background.get("mean_contextual_coherence")
        ),
    }


def load_run_summaries(summary_files: list[str]) -> list[dict[str, Any]]:
    runs = []
    for summary_file in summary_files:
        path = Path(summary_file)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                run = json.load(f)
        except Exception:
            continue
        run["summary_file"] = str(path)
        run["run_key"] = run_key(run)
        runs.append(run)
    return runs


def build_checkpoint_scores(
    runs: list[dict[str, Any]],
    expected_tasks: list[str],
) -> list[dict[str, Any]]:
    deduped_runs: dict[str, dict[str, Any]] = {}
    for run in runs:
        deduped_runs[run.get("run_key", "unknown_target")] = run

    checkpoint_scores = []
    for key, run in deduped_runs.items():
        task_metrics = run.get("task_metrics", {})
        per_task = {}
        values = []

        for task_key in expected_tasks:
            metric = task_metrics.get(task_key)
            if metric is None:
                continue
            mean_toxicity = float(metric.get("mean_toxicity", 0.0))
            per_task[task_key] = {
                "num_samples": int(metric.get("num_samples", 0)),
                "mean_toxicity": mean_toxicity,
            }
            values.append(mean_toxicity)

        mean_toxicity_mean = sum(values) / len(values) if values else 1.0
        missing_tasks = [task for task in expected_tasks if task not in per_task]

        checkpoint_scores.append(
            {
                "checkpoint": key,
                "mean_toxicity_mean": mean_toxicity_mean,
                "per_task": per_task,
                "missing_tasks": missing_tasks,
                **extract_auxiliary_metrics(run),
                **extract_judge_metrics(run),
            }
        )

    checkpoint_scores.sort(key=lambda x: x["mean_toxicity_mean"])
    return checkpoint_scores


def build_ranking(checkpoint_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "checkpoint": item["checkpoint"],
            "mean_toxicity_mean": item["mean_toxicity_mean"],
            "perplexity": item.get("perplexity"),
            "distinct_1": item.get("distinct_1"),
            "distinct_2": item.get("distinct_2"),
            "refusal_combo_rate": item.get("refusal_combo_rate"),
            "toxicity_gt_0p5_rate": item.get("toxicity_gt_0p5_rate"),
            "toxicity_gt_0p8_rate": item.get("toxicity_gt_0p8_rate"),
            "judge_high_risk_mean_true_toxicity": item.get("judge_high_risk_mean_true_toxicity"),
            "judge_high_risk_true_toxicity_ge_2_rate": item.get("judge_high_risk_true_toxicity_ge_2_rate"),
            "judge_random_background_over_refusal_rate": item.get("judge_random_background_over_refusal_rate"),
        }
        for i, item in enumerate(checkpoint_scores)
    ]


def main() -> None:
    args = parse_args()
    merge_summary_path = Path(args.merge_summary_path)
    expected_tasks = [f"{d}|{p}" for d in args.datasets for p in args.prompts]
    runs = load_run_summaries(args.summary_files)
    checkpoint_scores = build_checkpoint_scores(runs, expected_tasks)
    ranking = build_ranking(checkpoint_scores)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_ts,
        "datasets": args.datasets,
        "prompts": args.prompts,
        "runs": runs,
        "checkpoint_scores": checkpoint_scores,
        "ranking": ranking,
        "failed_runs": args.failed_runs,
    }

    merge_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with merge_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(str(merge_summary_path))


if __name__ == "__main__":
    main()
