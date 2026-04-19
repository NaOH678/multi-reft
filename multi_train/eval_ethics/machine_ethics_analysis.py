import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _to_float(value):
    return float(round(float(value), 6))


def _resolve_model_name(df, fallback):
    if "model_tag" in df.columns and not df["model_tag"].dropna().empty:
        return str(df["model_tag"].dropna().iloc[0])
    return fallback


def summarize_ethics_csv(
    results_csv,
    prompt_types=None,
    summary_json_path=None,
    model_name=None,
):
    csv_path = Path(results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"results csv does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    if "prompt_type" not in df.columns or "pred" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "results csv must contain columns: prompt_type, pred, label"
        )

    if prompt_types is None:
        requested_prompts = sorted(df["prompt_type"].dropna().astype(int).unique().tolist())
    else:
        requested_prompts = [int(x) for x in prompt_types]

    prompt_metrics = {}
    accuracies = []

    for prompt_type in requested_prompts:
        sub_df = df[df["prompt_type"].astype(int) == int(prompt_type)]
        num_samples = int(len(sub_df))
        if num_samples == 0:
            continue

        correct = int((sub_df["pred"].astype(int) == sub_df["label"].astype(int)).sum())
        accuracy = correct / num_samples
        fpr = 1.0 - accuracy

        prompt_metrics[str(prompt_type)] = {
            "num_samples": num_samples,
            "correct": correct,
            "accuracy": _to_float(accuracy),
            "fpr": _to_float(fpr),
        }
        accuracies.append(accuracy)

    overall_score = _to_float(sum(accuracies) / len(accuracies)) if accuracies else 0.0

    resolved_model = model_name or _resolve_model_name(df, csv_path.stem)
    summary = {
        "model_name": resolved_model,
        "results_csv": str(csv_path),
        "prompt_types": requested_prompts,
        "prompt_metrics": prompt_metrics,
        "overall_score": overall_score,
        "num_rows": int(len(df)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    saved_path = None
    if summary_json_path is not None:
        saved_path = Path(summary_json_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(saved_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary, str(saved_path) if saved_path else None


def parse_args():
    parser = argparse.ArgumentParser("Machine ethics summary tool")
    parser.add_argument("--results_csv", type=str, default=None)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="multi_train/eval_ethics/data/generations",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="multi_train/eval_ethics/data/outputs",
    )
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument("--prompt_types", type=int, nargs="+", default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--pattern", type=str, default="*-ethics.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_csv:
        summary, summary_path = summarize_ethics_csv(
            results_csv=args.results_csv,
            prompt_types=args.prompt_types,
            summary_json_path=args.summary_json,
            model_name=args.model_name,
        )
        if summary_path:
            print(f"Saved ethics summary to: {summary_path}")
        else:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    results_dir = Path(args.results_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(results_dir.glob(args.pattern))
    if not csv_files:
        print(f"No ethics csv found in {results_dir} matching {args.pattern}")
        return

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "runs": [],
    }

    for csv_path in csv_files:
        per_summary_path = outputs_dir / f"{csv_path.stem}_summary.json"
        summary, saved = summarize_ethics_csv(
            results_csv=csv_path,
            prompt_types=args.prompt_types,
            summary_json_path=per_summary_path,
            model_name=args.model_name,
        )
        aggregate["runs"].append(summary)
        print(f"Saved ethics summary to: {saved}")

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Saved aggregated summary to: {out_path}")


if __name__ == "__main__":
    main()
