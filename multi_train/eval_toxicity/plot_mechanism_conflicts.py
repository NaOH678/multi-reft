#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


MODULE_ORDER = {"truthful": 0, "moral": 1, "stereotype": 2, "toxicity": 3}
MODULES = ["truthful", "moral", "stereotype", "toxicity"]
PAIR_ORDER = [
    "truthful--moral",
    "truthful--stereotype",
    "truthful--toxicity",
    "moral--stereotype",
    "moral--toxicity",
    "stereotype--toxicity",
]
PAIR_COLORS = {
    "truthful--moral": "#4c78a8",
    "truthful--stereotype": "#72b7b2",
    "truthful--toxicity": "#54a24b",
    "moral--stereotype": "#f58518",
    "moral--toxicity": "#e45756",
    "stereotype--toxicity": "#b279a2",
}
MODULE_DISPLAY = {
    "truthful": "Truth",
    "moral": "Ethics",
    "stereotype": "Bias",
    "toxicity": "Toxic",
}
MODULE_DISPLAY_SHORT = {
    "truthful": "T",
    "moral": "E",
    "stereotype": "B",
    "toxicity": "X",
}
PAIR_DISPLAY = {
    "truthful--moral": "T-E",
    "truthful--stereotype": "T-B",
    "truthful--toxicity": "T-X",
    "moral--stereotype": "E-B",
    "moral--toxicity": "E-X",
    "stereotype--toxicity": "B-X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mechanism-conflict CSVs and render a high-density figure suite."
    )
    parser.add_argument("--trace_path", required=True, help="Path to *_mechanism_trace.jsonl.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write CSVs and figures. Defaults to a sibling directory next to the trace.",
    )
    parser.add_argument("--stem", default="structured_edit_conflicts", help="Stem for output figure files.")
    return parser.parse_args()


def infer_output_dir(trace_path: Path) -> Path:
    return trace_path.parent / f"{trace_path.stem}_figures"


def short_module(label: str) -> str:
    lower = label.lower()
    if "truthful" in lower or "truth" in lower:
        return "truthful"
    if "moral" in lower:
        return "moral"
    if "stereotype" in lower or "bias" in lower:
        return "stereotype"
    if "toxicity" in lower or "toxic" in lower:
        return "toxicity"
    return label


def canonical_pair_name(pair_name: str | None) -> str | None:
    if not pair_name:
        return None
    left, right = pair_name.split("__", 1)
    left_mod = short_module(left)
    right_mod = short_module(right)
    ordered = sorted((left_mod, right_mod), key=lambda name: (MODULE_ORDER.get(name, 999), name))
    return f"{ordered[0]}--{ordered[1]}"


def display_module_name(label: str) -> str:
    return MODULE_DISPLAY.get(short_module(label), label)


def display_pair_name(label: str) -> str:
    return PAIR_DISPLAY.get(label, label)


def display_module_short_name(label: str) -> str:
    return MODULE_DISPLAY_SHORT.get(short_module(label), label)


def parse_layer_index(layer_name: str) -> int:
    parts = str(layer_name).split("_")
    if len(parts) < 2:
        raise ValueError(f"Unable to parse layer index from {layer_name}")
    return int(parts[1])


def pct(value: float) -> float:
    return value * 100.0


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem_path: Path) -> None:
    fig.savefig(stem_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem_path.with_suffix(".pdf"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def annotate_heatmap(ax, matrix, fmt: str = ".3f", percent: bool = False) -> None:
    flat = [cell for row in matrix for cell in row if cell is not None and not math.isnan(cell)]
    vmax = max(flat) if flat else 1.0
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value is None or math.isnan(value):
                label = "NA"
                color = "#202020"
            else:
                label = f"{value:.1f}" if percent else format(value, fmt)
                color = "white" if value > 0.55 * vmax else "#202020"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.2, color=color)


def style_heatmap_axes(ax, x_label: str, y_label: str, x_ticks: list[int], y_ticks: list[int], x_ticklabels: list[str], y_ticklabels: list[str]) -> None:
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.tick_params(axis="both", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_soft_cmap(colors: list[str], name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors)


def build_layer_pair_matrix(layer_pair_rows: list[dict], key: str) -> list[list[float]]:
    row_map = {(row["pair"], row["layer"]): row for row in layer_pair_rows}
    return [[row_map[(pair, layer)][key] for layer in range(32)] for pair in PAIR_ORDER]


def build_pair_matrix(rows: list[dict], value_key: str) -> list[list[float | None]]:
    row_map = {row["pair"]: row for row in rows}
    matrix: list[list[float | None]] = [[None for _ in MODULES] for _ in MODULES]
    for pair_name, row in row_map.items():
        left, right = pair_name.split("--")
        i = MODULE_ORDER[left]
        j = MODULE_ORDER[right]
        matrix[i][j] = row[value_key]
        matrix[j][i] = row[value_key]
    return matrix


def draw_main_figure(output_stem: Path, layer_rows: list[dict], pair_rows: list[dict], top_demand_rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]
    mean_conflicts = [row["mean_total_conflict"] for row in layer_rows]

    ax = axes[0]
    ax.plot(layers, mean_conflicts, color="#1f4e79", linewidth=2.4, marker="o", markersize=3.5)
    ax.set_title("(a) Layer-wise unfiltered conflict", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean total conflict")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(min(layers), max(layers))
    ax.text(
        0.98,
        0.98,
        (
            f"Positive: {summary['overall_positive_conflict_rate_pct']:.1f}%\n"
            f"Mean: {summary['overall_mean_conflict']:.2e}\n"
            f"Active-negative: {summary['overall_active_negative_rate_pct']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0"},
    )

    ax = axes[1]
    pair_labels = [row["pair"] for row in pair_rows]
    pair_shares = [row["conflict_share_pct"] for row in pair_rows]
    ax.bar(range(len(pair_labels)), pair_shares, color=[PAIR_COLORS.get(label, "#4c78a8") for label in pair_labels], width=0.75)
    ax.set_title("(b) Conflict mass by module pair", loc="left", fontsize=12)
    ax.set_ylabel("Conflict share (%)")
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels([display_pair_name(label) for label in pair_labels], rotation=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(
        0.98,
        0.98,
        f"Top-4 share: {summary['top4_pair_conflict_share_pct']:.1f}%\nTruthful-related: {summary['truthful_conflict_share_pct']:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d0d0d0"},
    )

    ax = axes[2]
    settings = [row["setting"] for row in top_demand_rows]
    neg_rates = [row["negative_cos_rate_pct"] for row in top_demand_rows]
    ax.bar(range(len(settings)), neg_rates, color=["#e15759", "#f28e2b"], width=0.65)
    ax.set_title("(c) Top-demand pair incompatibility", loc="left", fontsize=12)
    ax.set_ylabel("Negative-cosine rate (%)")
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    fig.suptitle("Structured edit conflicts under direct demand-based mixing", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_layer_metrics_panel(output_stem: Path, layer_rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 4.8), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]
    specs = [
        ("mean_total_conflict", "Mean total conflict", "#1f4e79", summary["overall_mean_conflict"], ".3e"),
        ("positive_conflict_rate_pct", "Positive-conflict rate (%)", "#e15759", summary["overall_positive_conflict_rate_pct"], ".1f"),
        ("active_negative_rate_pct", "Active-negative rate (%)", "#59a14f", summary["overall_active_negative_rate_pct"], ".1f"),
    ]
    for ax, (key, title, color, overall, fmt) in zip(axes, specs):
        values = [row[key] for row in layer_rows]
        ax.plot(layers, values, color=color, linewidth=2.4, marker="o", markersize=3.2)
        ax.axhline(overall, color="#888888", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Layer")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.text(
            0.97,
            0.95,
            f"overall = {format(overall, fmt)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.3,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#d0d0d0"},
        )
    fig.suptitle("Layer-wise conflict profiles", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_layer_pair_heatmaps(output_stem: Path, layer_pair_rows: list[dict]) -> None:
    metric_specs = [
        ("mean_pair_conflict", "(a) Mean pair conflict", "magma", ".3e"),
        ("pair_conflict_share_in_layer_pct", "(b) Pair conflict share within layer (%)", "viridis", ".1f"),
        ("negative_cos_rate_pct", "(c) Negative-cosine rate (%)", "cividis", ".1f"),
        ("active_negative_rate_pct", "(d) Active-negative rate (%)", "plasma", ".1f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 8.2), constrained_layout=True)
    short_pairs = [display_pair_name(pair) for pair in PAIR_ORDER]
    for ax, (key, title, cmap, fmt) in zip(axes.flat, metric_specs):
        matrix = build_layer_pair_matrix(layer_pair_rows, key)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12)
        style_heatmap_axes(
            ax,
            "Layer",
            "Pair",
            list(range(0, 32, 2)),
            list(range(len(PAIR_ORDER))),
            [str(i) for i in range(0, 32, 2)],
            short_pairs,
        )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    fig.suptitle("Layer-by-pair conflict structure", fontsize=14, y=1.01)
    save_figure(fig, output_stem)


def draw_pair_matrices(output_stem: Path, pair_rows: list[dict], candidate_pair_rows: list[dict]) -> None:
    candidate_map = {row["pair"]: row for row in candidate_pair_rows}
    enriched_rows = []
    for row in pair_rows:
        merged = dict(row)
        merged["negative_candidate_rate_pct"] = candidate_map.get(row["pair"], {}).get("negative_candidate_rate_pct", math.nan)
        merged["rejected_candidate_rate_pct"] = candidate_map.get(row["pair"], {}).get("rejected_candidate_rate_pct", math.nan)
        enriched_rows.append(merged)
    metric_specs = [
        ("conflict_share_pct", "Pair conflict share (%)", "Blues"),
        ("negative_candidate_rate_pct", "Top-demand negative-cos rate (%)", "Reds"),
        ("rejected_candidate_rate_pct", "Top-demand rejected rate (%)", "Purples"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), constrained_layout=True)
    for ax, (key, title, cmap) in zip(axes, metric_specs):
        matrix = build_pair_matrix(enriched_rows, key)
        numeric = [[0.0 if cell is None else cell for cell in row] for row in matrix]
        im = ax.imshow(numeric, cmap=cmap, vmin=0.0)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks(range(len(MODULES)))
        ax.set_yticks(range(len(MODULES)))
        ax.set_xticklabels([display_module_name(name) for name in MODULES], rotation=20, ha="right")
        ax.set_yticklabels([display_module_name(name) for name in MODULES])
        vmax = max([v for row in numeric for v in row] or [1.0])
        for i in range(len(MODULES)):
            for j in range(len(MODULES)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#202020", fontsize=10.5)
                else:
                    value = matrix[i][j]
                    ax.text(
                        j,
                        i,
                        f"{value:.1f}" if value is not None else "NA",
                        ha="center",
                        va="center",
                        color="white" if value is not None and value > 0.5 * vmax else "#202020",
                        fontsize=9.3,
                    )
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Pair-level conflict and incompatibility matrices", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_candidate_pair_panel(output_stem: Path, top_demand_rows: list[dict], candidate_pair_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), constrained_layout=True)
    width = 0.36

    ax = axes[0]
    settings = [row["setting"] for row in top_demand_rows]
    neg_rates = [row["negative_cos_rate_pct"] for row in top_demand_rows]
    rej_rates = [row["rejected_rate_pct"] for row in top_demand_rows]
    x = list(range(len(settings)))
    ax.bar([i - width / 2 for i in x], neg_rates, width=width, color="#e15759", label="Negative cos")
    ax.bar([i + width / 2 for i in x], rej_rates, width=width, color="#4c78a8", label="Rejected")
    ax.set_title("(a) Overall top-demand incompatibility", loc="left", fontsize=12)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)

    ordered = [row for row in candidate_pair_rows if row["pair"] in PAIR_ORDER]
    ax = axes[1]
    x = list(range(len(ordered)))
    neg = [row["negative_candidate_rate_pct"] for row in ordered]
    rej = [row["rejected_candidate_rate_pct"] for row in ordered]
    ax.bar([i - width / 2 for i in x], neg, width=width, color="#f58518", label="Negative cos")
    ax.bar([i + width / 2 for i in x], rej, width=width, color="#6f4e7c", label="Rejected")
    ax.set_title("(b) By pair type", loc="left", fontsize=12)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([display_pair_name(row["pair"]) for row in ordered], rotation=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)

    ax = axes[2]
    kept = [row["kept_mean_cos"] for row in ordered]
    rejected = [row["rejected_mean_cos"] for row in ordered]
    ax.bar([i - width / 2 for i in x], rejected, width=width, color="#e15759", label="Rejected")
    ax.bar([i + width / 2 for i in x], kept, width=width, color="#4c78a8", label="Kept")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_title("(c) Mean cosine of kept vs. rejected top-demand pairs", loc="left", fontsize=12)
    ax.set_ylabel("Mean candidate pairwise cosine")
    ax.set_xticks(x)
    ax.set_xticklabels([display_pair_name(row["pair"]) for row in ordered], rotation=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False)

    fig.suptitle("Top-demand candidate-pair behavior", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_group_split_panel(output_stem: Path, group_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8), constrained_layout=True)

    def select(group_type: str) -> list[dict]:
        return [row for row in group_rows if row["group_type"] == group_type]

    specs = [
        ("dataset", "(a) By dataset"),
        ("prompt_type", "(b) By prompt type"),
        ("dataset__prompt_type", "(c) By dataset × prompt type"),
    ]
    for ax, (group_type, title) in zip(axes, specs):
        rows = select(group_type)
        labels = [row["group_label"] for row in rows]
        means = [row["mean_total_conflict"] for row in rows]
        pos_rates = [row["positive_conflict_rate_pct"] for row in rows]
        x = list(range(len(rows)))
        ax.bar(x, means, color="#4c78a8", width=0.65)
        ax2 = ax.twinx()
        ax2.plot(x, pos_rates, color="#e15759", marker="o", linewidth=2.0)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20 if len(rows) > 2 else 0, ha="right" if len(rows) > 2 else "center")
        ax.set_ylabel("Mean total conflict", color="#4c78a8")
        ax2.set_ylabel("Positive-conflict rate (%)", color="#e15759")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    fig.suptitle("Conflict splits by prompt group", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_filter_conflict_panel(output_stem: Path, layer_rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.9), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]

    ax = axes[0]
    ax.plot(layers, [row["mean_total_conflict"] for row in layer_rows], color="#1f4e79", linewidth=2.3, label="Before filter")
    ax.plot(layers, [row["mean_filtered_conflict"] for row in layer_rows], color="#e15759", linewidth=2.3, label="After filter")
    ax.set_title("(a) Layer-wise mean conflict", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean conflict")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(layers, [row["positive_conflict_rate_pct"] for row in layer_rows], color="#4c78a8", linewidth=2.3, label="Before filter")
    ax.plot(layers, [row["filtered_positive_conflict_rate_pct"] for row in layer_rows], color="#f58518", linewidth=2.3, label="After filter")
    ax.set_title("(b) Layer-wise positive-conflict rate", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    labels = ["Mean conflict", "Positive rate", "Conflict mass"]
    before_vals = [
        summary["overall_mean_conflict"],
        summary["overall_positive_conflict_rate_pct"],
        summary["total_conflict_mass"],
    ]
    after_vals = [
        summary["overall_mean_filtered_conflict"],
        summary["overall_filtered_positive_conflict_rate_pct"],
        summary["total_filtered_conflict_mass"],
    ]
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], before_vals, width=width, color="#4c78a8", label="Before")
    ax.bar([i + width / 2 for i in x], after_vals, width=width, color="#e15759", label="After")
    ax.set_title("(c) Overall conflict reduction", loc="left", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.text(
        0.98,
        0.95,
        f"Reduction: {summary['overall_conflict_reduction_pct']:.1f}%\nPositive-rate drop: {summary['filtered_positive_rate_drop_pct']:.1f} pts",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.1,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d0d0d0"},
    )

    fig.suptitle("Compatibility filter reduces realized conflict", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_selection_behavior_panel(
    output_stem: Path,
    summary: dict,
    selection_metric_rows: list[dict],
    selection_size_rows: list[dict],
    layer_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.9), constrained_layout=True)

    ax = axes[0]
    metric_labels = [row["metric_label"] for row in selection_metric_rows]
    metric_values = [row["value_pct"] if row["is_percent"] else row["value"] for row in selection_metric_rows]
    colors = ["#e15759", "#4c78a8", "#f58518", "#59a14f", "#6f4e7c"]
    ax.bar(range(len(metric_labels)), metric_values, color=colors[: len(metric_labels)], width=0.7)
    ax.set_title("(a) Overall routing effects", loc="left", fontsize=12)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=18, ha="right")
    ax.set_ylabel("Rate (%) / mean count")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    ax = axes[1]
    size_labels = [row["selected_size_label"] for row in selection_size_rows]
    size_values = [row["selected_size_rate_pct"] for row in selection_size_rows]
    ax.bar(range(len(size_labels)), size_values, color="#4c78a8", width=0.7)
    ax.set_title("(b) Selected-set size distribution", loc="left", fontsize=12)
    ax.set_xticks(range(len(size_labels)))
    ax.set_xticklabels(size_labels)
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    ax = axes[2]
    layers = [row["layer"] for row in layer_rows]
    ax.plot(layers, [row["selection_changed_rate_pct"] for row in layer_rows], color="#e15759", linewidth=2.2, label="Selection changed")
    ax.plot(layers, [row["top2_rejected_rate_pct"] for row in layer_rows], color="#f58518", linewidth=2.2, label="Top2 rejected")
    ax.plot(layers, [row["top1_kept_rate_pct"] for row in layer_rows], color="#4c78a8", linewidth=2.2, label="Top1 kept")
    ax.set_title("(c) Layer-wise filter action", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Compatibility filter changes routing selectively", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_before_after_conflict_panel(output_stem: Path, summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.6), constrained_layout=True)
    specs = [
        ("Mean conflict", summary["overall_mean_conflict"], summary["overall_mean_filtered_conflict"]),
        ("Positive-conflict rate (%)", summary["overall_positive_conflict_rate_pct"], summary["overall_filtered_positive_conflict_rate_pct"]),
        ("Total conflict mass", summary["total_conflict_mass"], summary["total_filtered_conflict_mass"]),
    ]
    width = 0.34
    for ax, (title, before, after) in zip(axes, specs):
        ax.bar([-width / 2], [before], width=width, color="#4c78a8", label="Before")
        ax.bar([width / 2], [after], width=width, color="#e15759", label="After")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks([-width / 2, width / 2])
        ax.set_xticklabels(["Before", "After"])
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=9)
    axes[2].text(
        0.98,
        0.95,
        f"Conflict reduction: {summary['overall_conflict_reduction_pct']:.1f}%",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d0d0d0"},
    )
    fig.suptitle("Before/After conflict under compatibility filtering", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_layerwise_before_after_conflict_panel(output_stem: Path, layer_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]

    ax = axes[0]
    ax.plot(layers, [row["mean_total_conflict"] for row in layer_rows], color="#4c78a8", linewidth=2.3, label="Before")
    ax.plot(layers, [row["mean_filtered_conflict"] for row in layer_rows], color="#e15759", linewidth=2.3, label="After")
    ax.set_title("(a) Mean conflict by layer", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean conflict")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(layers, [row["positive_conflict_rate_pct"] for row in layer_rows], color="#4c78a8", linewidth=2.3, label="Before")
    ax.plot(layers, [row["filtered_positive_conflict_rate_pct"] for row in layer_rows], color="#e15759", linewidth=2.3, label="After")
    ax.set_title("(b) Positive-conflict rate by layer", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Layer-wise before/after conflict", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_layerwise_filter_action_panel(output_stem: Path, layer_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.0), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]
    specs = [
        ("selection_changed_rate_pct", "(a) Selection changed rate by layer", "#e15759", "Rate (%)"),
        ("top2_rejected_rate_pct", "(b) Top-2 rejected rate by layer", "#f58518", "Rate (%)"),
        ("mean_selected_count", "(c) Mean selected count by layer", "#4c78a8", "Mean count"),
        ("single_selected_rate_pct", "(d) Single-selected rate by layer", "#59a14f", "Rate (%)"),
    ]
    for ax, (key, title, color, ylabel) in zip(axes.flat, specs):
        ax.plot(layers, [row[key] for row in layer_rows], color=color, linewidth=2.3, marker="o", markersize=3.0)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.suptitle("Layer-wise filter action", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_before_after_selection_panel(output_stem: Path, summary: dict, selection_size_rows: list[dict], layer_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    width = 0.36

    ax = axes[0]
    labels = ["Mean set size", "Single-expert rate (%)"]
    before_vals = [summary["mean_topk_count"], 0.0]
    after_vals = [summary["mean_selected_count"], summary["single_selected_rate_pct"]]
    x = list(range(len(labels)))
    ax.bar([i - width / 2 for i in x], before_vals, width=width, color="#4c78a8", label="Before")
    ax.bar([i + width / 2 for i in x], after_vals, width=width, color="#e15759", label="After")
    ax.set_title("(a) Aggregate selection size", loc="left", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    sizes = [row["selected_size_label"] for row in selection_size_rows]
    before_size = [row["topk_size_rate_pct"] for row in selection_size_rows]
    after_size = [row["selected_size_rate_pct"] for row in selection_size_rows]
    x = list(range(len(sizes)))
    ax.bar([i - width / 2 for i in x], before_size, width=width, color="#4c78a8", label="Before")
    ax.bar([i + width / 2 for i in x], after_size, width=width, color="#e15759", label="After")
    ax.set_title("(b) Set-size distribution", loc="left", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Set size")
    ax.set_ylabel("Rate (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    layers = [row["layer"] for row in layer_rows]
    ax.plot(layers, [row["mean_topk_count"] for row in layer_rows], color="#4c78a8", linewidth=2.3, label="Before |topk|")
    ax.plot(layers, [row["mean_selected_count"] for row in layer_rows], color="#e15759", linewidth=2.3, label="After |selected|")
    ax.set_title("(c) Selection size by layer", loc="left", fontsize=12)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean set size")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Before/after selection behavior", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_layer_pair_before_after_heatmap(output_stem: Path, layer_pair_rows: list[dict]) -> None:
    metric_specs = [
        ("mean_pair_conflict", "(a) Before: mean pair conflict", "magma", ".3e"),
        ("mean_filtered_pair_conflict", "(b) After: mean pair conflict", "magma", ".3e"),
        ("pair_conflict_reduction_pct", "(c) Pair conflict reduction (%)", "viridis", ".1f"),
        ("filtered_positive_pair_conflict_rate_pct", "(d) After: positive pair-conflict rate (%)", "plasma", ".1f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 8.2), constrained_layout=True)
    short_pairs = [display_pair_name(pair) for pair in PAIR_ORDER]
    for ax, (key, title, cmap, fmt) in zip(axes.flat, metric_specs):
        matrix = build_layer_pair_matrix(layer_pair_rows, key)
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12)
        style_heatmap_axes(
            ax,
            "Layer",
            "Pair",
            list(range(0, 32, 2)),
            list(range(len(PAIR_ORDER))),
            [str(i) for i in range(0, 32, 2)],
            short_pairs,
        )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    fig.suptitle("Layer-by-pair before/after conflict", fontsize=14, y=1.01)
    save_figure(fig, output_stem)


def draw_routing_heatmap_panel(
    output_stem: Path,
    specialist_labels: list[str],
    routing_rows: list[dict],
) -> None:
    short_labels = [display_module_name(label) for label in specialist_labels]
    selected_map = {(row["specialist_index"], row["layer"]): row["selected_rate_pct"] for row in routing_rows}
    alpha_map = {(row["specialist_index"], row["layer"]): row["alpha_mean"] for row in routing_rows}
    selected_matrix = [[selected_map[(idx, layer)] for layer in range(32)] for idx in range(len(short_labels))]
    alpha_matrix = [[alpha_map[(idx, layer)] for layer in range(32)] for idx in range(len(short_labels))]

    fig, axes = plt.subplots(1, 2, figsize=(15.6, 4.9), constrained_layout=True)
    specs = [
        (selected_matrix, "(a) Selected rate (%)", "viridis", True),
        (alpha_matrix, "(b) Mean filtered alpha", "magma", False),
    ]
    for ax, (matrix, title, cmap, percent) in zip(axes, specs):
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12)
        style_heatmap_axes(
            ax,
            "Layer",
            "Specialist",
            list(range(0, 32, 2)),
            list(range(len(short_labels))),
            [str(i) for i in range(0, 32, 2)],
            short_labels,
        )
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    fig.suptitle("Routing heatmap across layers and specialists", fontsize=14, y=1.02)
    save_figure(fig, output_stem)


def draw_publication_heatmap_strip(
    output_stem: Path,
    specialist_labels: list[str],
    routing_rows: list[dict],
    layer_pair_rows: list[dict],
) -> None:
    short_pairs = [display_pair_name(pair) for pair in PAIR_ORDER]
    short_labels = [display_module_short_name(label) for label in specialist_labels]
    selected_map = {(row["specialist_index"], row["layer"]): row["selected_rate_pct"] for row in routing_rows}
    alpha_map = {(row["specialist_index"], row["layer"]): row["alpha_mean"] for row in routing_rows}
    selected_matrix = [[selected_map[(idx, layer)] for layer in range(32)] for idx in range(len(short_labels))]
    alpha_matrix = [[alpha_map[(idx, layer)] for layer in range(32)] for idx in range(len(short_labels))]

    mean_pair_conflict_matrix = build_layer_pair_matrix(layer_pair_rows, "mean_pair_conflict")
    active_negative_matrix = build_layer_pair_matrix(layer_pair_rows, "active_negative_rate_pct")

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16.6, 2.95),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.0]},
    )
    conflict_cmap = make_soft_cmap(["#faf8fb", "#ead8ef", "#d49ec6", "#8e5ea2"], "soft_conflict")
    active_cmap = make_soft_cmap(["#fbfafc", "#efe2f4", "#cf9bc8", "#7c5aa6"], "soft_active")
    selected_cmap = make_soft_cmap(["#f8fafc", "#d8e3f2", "#93add6", "#5a73b5"], "soft_selected")
    alpha_cmap = make_soft_cmap(["#faf9fc", "#e6def0", "#b9abd9", "#7e6ab0"], "soft_alpha")
    xticks = list(range(0, 32, 4))
    xticklabels = [str(i) for i in xticks]

    axes[0].imshow(mean_pair_conflict_matrix, cmap=conflict_cmap, aspect="auto")
    axes[0].set_title("(a) Pair Conflict", loc="left", fontsize=14.0, pad=6)
    style_heatmap_axes(axes[0], "", "", xticks, list(range(len(short_pairs))), xticklabels, short_pairs)

    axes[1].imshow(active_negative_matrix, cmap=active_cmap, aspect="auto")
    axes[1].set_title("(b) Neg. Overlap", loc="left", fontsize=14.0, pad=6)
    style_heatmap_axes(axes[1], "", "", xticks, list(range(len(short_pairs))), xticklabels, [""] * len(short_pairs))

    axes[2].imshow(selected_matrix, cmap=selected_cmap, aspect="auto")
    axes[2].set_title("(c) Selection", loc="left", fontsize=14.0, pad=6)
    style_heatmap_axes(axes[2], "", "", xticks, list(range(len(short_labels))), xticklabels, short_labels)

    axes[3].imshow(alpha_matrix, cmap=alpha_cmap, aspect="auto")
    axes[3].set_title("(d) Alpha", loc="left", fontsize=14.0, pad=6)
    style_heatmap_axes(axes[3], "", "", xticks, list(range(len(short_labels))), xticklabels, [""] * len(short_labels))

    for ax in axes:
        ax.tick_params(axis="x", pad=2, length=3.0, width=0.8, colors="#444444", labelsize=12.5)
        ax.tick_params(axis="y", pad=2, length=3.0, width=0.8, colors="#444444", labelsize=12.5)
    fig.supxlabel("Layer", fontsize=14.0, y=0.02)
    fig.subplots_adjust(left=0.042, right=0.995, top=0.88, bottom=0.16, wspace=0.03)
    save_figure(fig, output_stem)


def draw_showcase_panel(output_stem: Path, layer_rows: list[dict], pair_rows: list[dict], candidate_pair_rows: list[dict], top_demand_rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    layers = [row["layer"] for row in layer_rows]
    axes[0, 0].plot(layers, [row["mean_total_conflict"] for row in layer_rows], color="#1f4e79", linewidth=2.3)
    axes[0, 0].plot(layers, [row["positive_conflict_rate"] for row in layer_rows], color="#e15759", linewidth=2.0)
    axes[0, 0].plot(layers, [row["active_negative_rate"] for row in layer_rows], color="#59a14f", linewidth=2.0)
    axes[0, 0].set_title("Layer-wise conflict profiles", loc="left", fontsize=12)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].legend(["Mean conflict", "Positive rate", "Active-negative rate"], frameon=False, fontsize=9)
    axes[0, 0].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    ordered_pairs = [row for row in pair_rows if row["pair"] in PAIR_ORDER]
    axes[0, 1].bar(range(len(ordered_pairs)), [row["conflict_share_pct"] for row in ordered_pairs], color=[PAIR_COLORS[row["pair"]] for row in ordered_pairs], width=0.75)
    axes[0, 1].set_title("Conflict mass by pair", loc="left", fontsize=12)
    axes[0, 1].set_ylabel("Share (%)")
    axes[0, 1].set_xticks(range(len(ordered_pairs)))
    axes[0, 1].set_xticklabels([display_pair_name(row["pair"]) for row in ordered_pairs], rotation=0)
    axes[0, 1].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    ordered_candidates = [row for row in candidate_pair_rows if row["pair"] in PAIR_ORDER]
    x = list(range(len(ordered_candidates)))
    width = 0.36
    axes[1, 0].bar([i - width / 2 for i in x], [row["negative_candidate_rate_pct"] for row in ordered_candidates], width=width, color="#f58518", label="Negative cos")
    axes[1, 0].bar([i + width / 2 for i in x], [row["rejected_candidate_rate_pct"] for row in ordered_candidates], width=width, color="#6f4e7c", label="Rejected")
    axes[1, 0].set_title("Top-demand incompatibility by pair", loc="left", fontsize=12)
    axes[1, 0].set_ylabel("Rate (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([display_pair_name(row["pair"]) for row in ordered_candidates], rotation=0)
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    settings = [row["setting"] for row in top_demand_rows]
    neg_rates = [row["negative_cos_rate_pct"] for row in top_demand_rows]
    rej_rates = [row["rejected_rate_pct"] for row in top_demand_rows]
    x2 = list(range(len(settings)))
    axes[1, 1].bar([i - width / 2 for i in x2], neg_rates, width=width, color="#e15759", label="Negative cos")
    axes[1, 1].bar([i + width / 2 for i in x2], rej_rates, width=width, color="#4c78a8", label="Rejected")
    axes[1, 1].set_title("Overall top-demand incompatibility", loc="left", fontsize=12)
    axes[1, 1].set_ylabel("Rate (%)")
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels(settings)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5)

    fig.suptitle("Conflict analysis overview", fontsize=14, y=1.01)
    save_figure(fig, output_stem)


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(trace_path)
    debug_summary_path = trace_path.with_name(trace_path.name.replace("_mechanism_trace.jsonl", "_debug_summary.json"))
    if not debug_summary_path.exists():
        raise FileNotFoundError(f"Expected debug summary next to trace: {debug_summary_path}")

    output_dir = Path(args.output_dir) if args.output_dir else infer_output_dir(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with debug_summary_path.open("r", encoding="utf-8") as f:
        debug_summary = json.load(f)
    specialist_labels = list(debug_summary.get("specialist_labels", []))
    routing_rows = []
    for layer_name, layer_payload in debug_summary.get("layers", {}).items():
        layer_idx = parse_layer_index(layer_name)
        for entry in layer_payload.get("per_specialist", []):
            routing_rows.append(
                {
                    "layer": layer_idx,
                    "specialist_index": int(entry["index"]),
                    "specialist_label": str(entry["label"]),
                    "alpha_mean": float(entry["alpha_mean"]),
                    "selected_rate": float(entry["selected_rate"]),
                    "selected_rate_pct": pct(float(entry["selected_rate"])),
                    "topk_rate": float(entry.get("topk_rate", 0.0)),
                    "topk_rate_pct": pct(float(entry.get("topk_rate", 0.0))),
                    "rejected_conflict_rate": float(entry.get("rejected_conflict_rate", 0.0)),
                    "rejected_conflict_rate_pct": pct(float(entry.get("rejected_conflict_rate", 0.0))),
                }
            )

    layer_stats: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_total_conflict": 0.0,
            "positive_conflict_count": 0.0,
            "active_negative_count": 0.0,
            "sum_filtered_conflict": 0.0,
            "filtered_positive_conflict_count": 0.0,
            "selection_changed_count": 0.0,
            "top1_kept_count": 0.0,
            "top2_rejected_count": 0.0,
            "top2_available_count": 0.0,
            "sum_selected_count": 0.0,
            "sum_topk_count": 0.0,
            "single_selected_count": 0.0,
        }
    )
    layer_pair_stats: dict[tuple[int, str], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_pair_conflict": 0.0,
            "sum_filtered_pair_conflict": 0.0,
            "sum_pairwise_cos": 0.0,
            "negative_cos_count": 0.0,
            "active_negative_count": 0.0,
            "positive_pair_conflict_count": 0.0,
            "filtered_positive_pair_conflict_count": 0.0,
        }
    )
    group_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "sum_total_conflict": 0.0,
            "positive_conflict_count": 0.0,
            "active_negative_count": 0.0,
            "sum_truthful_conflict_mass": 0.0,
        }
    )
    layer_total_conflict_sum: dict[int, float] = defaultdict(float)
    pair_conflict_sum: dict[str, float] = defaultdict(float)
    pair_negative_cos_count: Counter[str] = Counter()
    pair_active_negative_count: Counter[str] = Counter()
    candidate_pair_counts: Counter[str] = Counter()
    candidate_pair_neg_counts: Counter[str] = Counter()
    candidate_pair_rejected_counts: Counter[str] = Counter()
    candidate_pair_conflict_positive_counts: Counter[str] = Counter()
    candidate_pair_neg_conflict_positive_counts: Counter[str] = Counter()
    candidate_pair_rejected_conflict_positive_counts: Counter[str] = Counter()
    candidate_pair_kept_cos: dict[str, list[float]] = defaultdict(list)
    candidate_pair_rejected_cos: dict[str, list[float]] = defaultdict(list)
    selected_size_dist: Counter[int] = Counter()
    topk_size_dist: Counter[int] = Counter()

    total_records = 0
    positive_conflict_records = 0
    active_negative_records = 0
    total_conflict_mass = 0.0
    total_filtered_conflict_mass = 0.0
    truthful_conflict_mass = 0.0
    candidate_neg_all = 0
    candidate_neg_conflict_pos = 0
    candidate_rejected_all = 0
    candidate_rejected_conflict_pos = 0
    filtered_positive_conflict_records = 0
    selection_changed_records = 0
    selection_changed_conflict_positive_records = 0
    top1_kept_records = 0
    top2_rejected_records = 0
    top2_available_records = 0
    selected_count_sum = 0.0
    topk_count_sum = 0.0
    single_selected_records = 0

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_records += 1
            layer_idx = parse_layer_index(record["layer"])
            total_conflict = float(record["softmax_total_conflict"])
            filtered_conflict = float(record.get("filtered_conflict_score") or 0.0)
            truthful_conflict = float(record["truthful_conflict_mass"])
            filtered_alpha = [float(x) for x in record.get("filtered_alpha", [])]
            selected_indices = [int(x) for x in record.get("selected_indices", [])]
            topk_indices = [int(x) for x in record.get("topk_indices", [])]
            rejected_indices = [int(x) for x in record.get("rejected_indices", [])]
            top1_index = int(record.get("top1_index", -1))
            top2_index = int(record.get("top2_index", -1))
            selected_count = len(selected_indices)
            topk_count = len(topk_indices)
            selection_changed = selected_indices != topk_indices
            top1_kept = top1_index >= 0 and top1_index in selected_indices
            top2_available = top2_index >= 0
            top2_rejected = top2_available and top2_index in rejected_indices

            layer_entry = layer_stats[layer_idx]
            layer_entry["count"] += 1
            layer_entry["sum_total_conflict"] += total_conflict
            layer_entry["sum_filtered_conflict"] += filtered_conflict
            layer_total_conflict_sum[layer_idx] += total_conflict
            total_conflict_mass += total_conflict
            total_filtered_conflict_mass += filtered_conflict
            truthful_conflict_mass += truthful_conflict
            layer_entry["sum_selected_count"] += selected_count
            layer_entry["sum_topk_count"] += topk_count

            selected_size_dist[selected_count] += 1
            topk_size_dist[topk_count] += 1
            selected_count_sum += selected_count
            topk_count_sum += topk_count
            if selected_count == 1:
                single_selected_records += 1
                layer_entry["single_selected_count"] += 1

            if total_conflict > 0.0:
                positive_conflict_records += 1
                layer_entry["positive_conflict_count"] += 1
            if filtered_conflict > 0.0:
                filtered_positive_conflict_records += 1
                layer_entry["filtered_positive_conflict_count"] += 1
            if selection_changed:
                selection_changed_records += 1
                layer_entry["selection_changed_count"] += 1
                if total_conflict > 0.0:
                    selection_changed_conflict_positive_records += 1
            if top1_kept:
                top1_kept_records += 1
                layer_entry["top1_kept_count"] += 1
            if top2_available:
                top2_available_records += 1
                layer_entry["top2_available_count"] += 1
            if top2_rejected:
                top2_rejected_records += 1
                layer_entry["top2_rejected_count"] += 1

            any_active_negative = False
            for pair_entry in record["pair_conflicts"]:
                pair_name = canonical_pair_name(pair_entry["pair_name"])
                pair_conflict = float(pair_entry["softmax_pair_conflict"])
                pair_cos = float(pair_entry["pairwise_cos"])
                active_negative = bool(pair_entry["active_negative"])
                idx_i, idx_j = [int(x) for x in pair_entry["pair_indices"]]
                filtered_pair_conflict = 0.0
                if idx_i < len(filtered_alpha) and idx_j < len(filtered_alpha):
                    filtered_pair_conflict = filtered_alpha[idx_i] * filtered_alpha[idx_j] * max(-pair_cos, 0.0)

                pair_conflict_sum[pair_name] += pair_conflict
                if pair_cos < 0.0:
                    pair_negative_cos_count[pair_name] += 1
                if active_negative:
                    pair_active_negative_count[pair_name] += 1
                    any_active_negative = True

                lp_entry = layer_pair_stats[(layer_idx, pair_name)]
                lp_entry["count"] += 1
                lp_entry["sum_pair_conflict"] += pair_conflict
                lp_entry["sum_filtered_pair_conflict"] += filtered_pair_conflict
                lp_entry["sum_pairwise_cos"] += pair_cos
                if pair_cos < 0.0:
                    lp_entry["negative_cos_count"] += 1
                if active_negative:
                    lp_entry["active_negative_count"] += 1
                if pair_conflict > 0.0:
                    lp_entry["positive_pair_conflict_count"] += 1
                if filtered_pair_conflict > 0.0:
                    lp_entry["filtered_positive_pair_conflict_count"] += 1

            if any_active_negative:
                active_negative_records += 1
                layer_entry["active_negative_count"] += 1

            for group_type, group_label in [
                ("dataset", str(record["dataset"])),
                ("prompt_type", str(record["prompt_type"])),
                ("dataset__prompt_type", f"{record['dataset']}__{record['prompt_type']}"),
            ]:
                g = group_stats[(group_type, group_label)]
                g["count"] += 1
                g["sum_total_conflict"] += total_conflict
                g["sum_truthful_conflict_mass"] += truthful_conflict
                if total_conflict > 0.0:
                    g["positive_conflict_count"] += 1
                if any_active_negative:
                    g["active_negative_count"] += 1

            candidate_pair = canonical_pair_name(record.get("candidate_pair_name"))
            candidate_cos = record.get("candidate_pairwise_cos")
            candidate_rejected = bool(record.get("candidate_pair_rejected"))
            if candidate_pair is not None and candidate_cos is not None:
                candidate_cos = float(candidate_cos)
                candidate_pair_counts[candidate_pair] += 1
                if total_conflict > 0.0:
                    candidate_pair_conflict_positive_counts[candidate_pair] += 1
                if candidate_cos < 0.0:
                    candidate_pair_neg_counts[candidate_pair] += 1
                    candidate_neg_all += 1
                    if total_conflict > 0.0:
                        candidate_neg_conflict_pos += 1
                        candidate_pair_neg_conflict_positive_counts[candidate_pair] += 1
                if candidate_rejected:
                    candidate_rejected_all += 1
                    candidate_pair_rejected_counts[candidate_pair] += 1
                    candidate_pair_rejected_cos[candidate_pair].append(candidate_cos)
                    if total_conflict > 0.0:
                        candidate_rejected_conflict_pos += 1
                        candidate_pair_rejected_conflict_positive_counts[candidate_pair] += 1
                else:
                    candidate_pair_kept_cos[candidate_pair].append(candidate_cos)

    if total_records == 0:
        raise ValueError(f"No records found in {trace_path}")

    layer_rows = []
    for layer in range(32):
        stats = layer_stats[layer]
        count = stats["count"]
        sum_total_conflict = stats["sum_total_conflict"]
        sum_filtered_conflict = stats["sum_filtered_conflict"]
        layer_rows.append(
            {
                "layer": layer,
                "mean_total_conflict": safe_div(sum_total_conflict, count),
                "mean_filtered_conflict": safe_div(sum_filtered_conflict, count),
                "conflict_reduction_pct": pct(1.0 - safe_div(sum_filtered_conflict, sum_total_conflict))
                if sum_total_conflict > 0.0
                else 0.0,
                "positive_conflict_rate": safe_div(stats["positive_conflict_count"], count),
                "positive_conflict_rate_pct": pct(safe_div(stats["positive_conflict_count"], count)),
                "filtered_positive_conflict_rate": safe_div(stats["filtered_positive_conflict_count"], count),
                "filtered_positive_conflict_rate_pct": pct(safe_div(stats["filtered_positive_conflict_count"], count)),
                "active_negative_rate": safe_div(stats["active_negative_count"], count),
                "active_negative_rate_pct": pct(safe_div(stats["active_negative_count"], count)),
                "selection_changed_rate": safe_div(stats["selection_changed_count"], count),
                "selection_changed_rate_pct": pct(safe_div(stats["selection_changed_count"], count)),
                "top1_kept_rate": safe_div(stats["top1_kept_count"], count),
                "top1_kept_rate_pct": pct(safe_div(stats["top1_kept_count"], count)),
                "top2_rejected_rate": safe_div(stats["top2_rejected_count"], stats["top2_available_count"]),
                "top2_rejected_rate_pct": pct(safe_div(stats["top2_rejected_count"], stats["top2_available_count"])),
                "mean_selected_count": safe_div(stats["sum_selected_count"], count),
                "mean_topk_count": safe_div(stats["sum_topk_count"], count),
                "single_selected_rate": safe_div(stats["single_selected_count"], count),
                "single_selected_rate_pct": pct(safe_div(stats["single_selected_count"], count)),
            }
        )

    pair_rows = []
    for pair_name in PAIR_ORDER:
        pair_total = pair_conflict_sum[pair_name]
        pair_count = sum(layer_pair_stats[(layer, pair_name)]["count"] for layer in range(32))
        pair_rows.append(
            {
                "pair": pair_name,
                "total_conflict": pair_total,
                "conflict_share": safe_div(pair_total, total_conflict_mass),
                "conflict_share_pct": pct(safe_div(pair_total, total_conflict_mass)),
                "negative_cos_rate": safe_div(pair_negative_cos_count[pair_name], pair_count),
                "negative_cos_rate_pct": pct(safe_div(pair_negative_cos_count[pair_name], pair_count)),
                "active_negative_rate": safe_div(pair_active_negative_count[pair_name], pair_count),
                "active_negative_rate_pct": pct(safe_div(pair_active_negative_count[pair_name], pair_count)),
            }
        )
    pair_rows.sort(key=lambda row: row["conflict_share"], reverse=True)

    layer_pair_rows = []
    for pair_name in PAIR_ORDER:
        for layer in range(32):
            stats = layer_pair_stats[(layer, pair_name)]
            count = stats["count"]
            layer_pair_rows.append(
                {
                    "layer": layer,
                    "pair": pair_name,
                    "mean_pair_conflict": safe_div(stats["sum_pair_conflict"], count),
                    "mean_filtered_pair_conflict": safe_div(stats["sum_filtered_pair_conflict"], count),
                    "pair_conflict_reduction_pct": pct(
                        1.0 - safe_div(stats["sum_filtered_pair_conflict"], stats["sum_pair_conflict"])
                    )
                    if stats["sum_pair_conflict"] > 0.0
                    else 0.0,
                    "pair_conflict_share_in_layer": safe_div(stats["sum_pair_conflict"], layer_total_conflict_sum[layer]),
                    "pair_conflict_share_in_layer_pct": pct(safe_div(stats["sum_pair_conflict"], layer_total_conflict_sum[layer])),
                    "pair_conflict_share_global": safe_div(stats["sum_pair_conflict"], total_conflict_mass),
                    "pair_conflict_share_global_pct": pct(safe_div(stats["sum_pair_conflict"], total_conflict_mass)),
                    "mean_pairwise_cos": safe_div(stats["sum_pairwise_cos"], count),
                    "negative_cos_rate": safe_div(stats["negative_cos_count"], count),
                    "negative_cos_rate_pct": pct(safe_div(stats["negative_cos_count"], count)),
                    "active_negative_rate": safe_div(stats["active_negative_count"], count),
                    "active_negative_rate_pct": pct(safe_div(stats["active_negative_count"], count)),
                    "positive_pair_conflict_rate": safe_div(stats["positive_pair_conflict_count"], count),
                    "positive_pair_conflict_rate_pct": pct(safe_div(stats["positive_pair_conflict_count"], count)),
                    "filtered_positive_pair_conflict_rate": safe_div(stats["filtered_positive_pair_conflict_count"], count),
                    "filtered_positive_pair_conflict_rate_pct": pct(
                        safe_div(stats["filtered_positive_pair_conflict_count"], count)
                    ),
                }
            )

    top_demand_rows = [
        {
            "setting": "All records",
            "negative_cos_rate": safe_div(candidate_neg_all, total_records),
            "negative_cos_rate_pct": pct(safe_div(candidate_neg_all, total_records)),
            "rejected_rate": safe_div(candidate_rejected_all, total_records),
            "rejected_rate_pct": pct(safe_div(candidate_rejected_all, total_records)),
        },
        {
            "setting": "Conflict-positive records",
            "negative_cos_rate": safe_div(candidate_neg_conflict_pos, positive_conflict_records),
            "negative_cos_rate_pct": pct(safe_div(candidate_neg_conflict_pos, positive_conflict_records)),
            "rejected_rate": safe_div(candidate_rejected_conflict_pos, positive_conflict_records),
            "rejected_rate_pct": pct(safe_div(candidate_rejected_conflict_pos, positive_conflict_records)),
        },
    ]

    candidate_pair_rows = []
    for pair_name in PAIR_ORDER:
        total_pair_count = candidate_pair_counts[pair_name]
        conflict_positive_count = candidate_pair_conflict_positive_counts[pair_name]
        kept_list = candidate_pair_kept_cos[pair_name]
        rejected_list = candidate_pair_rejected_cos[pair_name]
        candidate_pair_rows.append(
            {
                "pair": pair_name,
                "candidate_count": total_pair_count,
                "negative_candidate_rate": safe_div(candidate_pair_neg_counts[pair_name], total_pair_count),
                "negative_candidate_rate_pct": pct(safe_div(candidate_pair_neg_counts[pair_name], total_pair_count)),
                "rejected_candidate_rate": safe_div(candidate_pair_rejected_counts[pair_name], total_pair_count),
                "rejected_candidate_rate_pct": pct(safe_div(candidate_pair_rejected_counts[pair_name], total_pair_count)),
                "negative_candidate_rate_conflict_positive": safe_div(candidate_pair_neg_conflict_positive_counts[pair_name], conflict_positive_count),
                "negative_candidate_rate_conflict_positive_pct": pct(safe_div(candidate_pair_neg_conflict_positive_counts[pair_name], conflict_positive_count)),
                "rejected_candidate_rate_conflict_positive": safe_div(candidate_pair_rejected_conflict_positive_counts[pair_name], conflict_positive_count),
                "rejected_candidate_rate_conflict_positive_pct": pct(safe_div(candidate_pair_rejected_conflict_positive_counts[pair_name], conflict_positive_count)),
                "kept_mean_cos": sum(kept_list) / len(kept_list) if kept_list else math.nan,
                "rejected_mean_cos": sum(rejected_list) / len(rejected_list) if rejected_list else math.nan,
                "kept_count": len(kept_list),
                "rejected_count": len(rejected_list),
            }
        )

    selection_metric_rows = [
        {
            "metric_key": "selection_changed_rate",
            "metric_label": "Selection changed",
            "value": safe_div(selection_changed_records, total_records),
            "value_pct": pct(safe_div(selection_changed_records, total_records)),
            "is_percent": True,
        },
        {
            "metric_key": "top1_kept_rate",
            "metric_label": "Top1 kept",
            "value": safe_div(top1_kept_records, total_records),
            "value_pct": pct(safe_div(top1_kept_records, total_records)),
            "is_percent": True,
        },
        {
            "metric_key": "top2_rejected_rate",
            "metric_label": "Top2 rejected",
            "value": safe_div(top2_rejected_records, top2_available_records),
            "value_pct": pct(safe_div(top2_rejected_records, top2_available_records)),
            "is_percent": True,
        },
        {
            "metric_key": "single_selected_rate",
            "metric_label": "|selected|=1",
            "value": safe_div(single_selected_records, total_records),
            "value_pct": pct(safe_div(single_selected_records, total_records)),
            "is_percent": True,
        },
        {
            "metric_key": "mean_selected_count",
            "metric_label": "Mean |selected|",
            "value": safe_div(selected_count_sum, total_records),
            "value_pct": safe_div(selected_count_sum, total_records),
            "is_percent": False,
        },
    ]

    selection_size_rows = []
    all_size_keys = sorted(set(selected_size_dist.keys()) | set(topk_size_dist.keys()))
    for size_key in all_size_keys:
        selection_size_rows.append(
            {
                "selected_size": size_key,
                "selected_size_label": str(size_key),
                "selected_size_count": selected_size_dist[size_key],
                "selected_size_rate": safe_div(selected_size_dist[size_key], total_records),
                "selected_size_rate_pct": pct(safe_div(selected_size_dist[size_key], total_records)),
                "topk_size_count": topk_size_dist[size_key],
                "topk_size_rate": safe_div(topk_size_dist[size_key], total_records),
                "topk_size_rate_pct": pct(safe_div(topk_size_dist[size_key], total_records)),
            }
        )

    group_rows = []
    for (group_type, group_label), stats in sorted(group_stats.items()):
        count = stats["count"]
        group_total_conflict = stats["sum_total_conflict"]
        group_rows.append(
            {
                "group_type": group_type,
                "group_label": group_label,
                "count": count,
                "mean_total_conflict": safe_div(group_total_conflict, count),
                "positive_conflict_rate": safe_div(stats["positive_conflict_count"], count),
                "positive_conflict_rate_pct": pct(safe_div(stats["positive_conflict_count"], count)),
                "active_negative_rate": safe_div(stats["active_negative_count"], count),
                "active_negative_rate_pct": pct(safe_div(stats["active_negative_count"], count)),
                "truthful_conflict_share": safe_div(stats["sum_truthful_conflict_mass"], group_total_conflict),
                "truthful_conflict_share_pct": pct(safe_div(stats["sum_truthful_conflict_mass"], group_total_conflict)),
            }
        )

    early_layers = [row for row in layer_rows if 0 <= row["layer"] <= 7]
    middle_layers = [row for row in layer_rows if 8 <= row["layer"] <= 23]
    late_layers = [row for row in layer_rows if 24 <= row["layer"] <= 31]
    max_layer_row = max(layer_rows, key=lambda row: row["mean_total_conflict"])
    top4_share = sum(row["conflict_share"] for row in pair_rows[:4])
    summary = {
        "trace_path": str(trace_path),
        "total_records": total_records,
        "overall_mean_conflict": safe_div(total_conflict_mass, total_records),
        "overall_mean_filtered_conflict": safe_div(total_filtered_conflict_mass, total_records),
        "overall_positive_conflict_rate": safe_div(positive_conflict_records, total_records),
        "overall_positive_conflict_rate_pct": pct(safe_div(positive_conflict_records, total_records)),
        "overall_filtered_positive_conflict_rate": safe_div(filtered_positive_conflict_records, total_records),
        "overall_filtered_positive_conflict_rate_pct": pct(safe_div(filtered_positive_conflict_records, total_records)),
        "overall_active_negative_rate": safe_div(active_negative_records, total_records),
        "overall_active_negative_rate_pct": pct(safe_div(active_negative_records, total_records)),
        "total_conflict_mass": total_conflict_mass,
        "total_filtered_conflict_mass": total_filtered_conflict_mass,
        "overall_conflict_reduction": 1.0 - safe_div(total_filtered_conflict_mass, total_conflict_mass)
        if total_conflict_mass > 0.0
        else 0.0,
        "overall_conflict_reduction_pct": pct(
            1.0 - safe_div(total_filtered_conflict_mass, total_conflict_mass) if total_conflict_mass > 0.0 else 0.0
        ),
        "filtered_positive_rate_drop_pct": pct(safe_div(positive_conflict_records, total_records))
        - pct(safe_div(filtered_positive_conflict_records, total_records)),
        "truthful_conflict_share": safe_div(truthful_conflict_mass, total_conflict_mass),
        "truthful_conflict_share_pct": pct(safe_div(truthful_conflict_mass, total_conflict_mass)),
        "top4_pair_conflict_share": top4_share,
        "top4_pair_conflict_share_pct": pct(top4_share),
        "selection_changed_rate": safe_div(selection_changed_records, total_records),
        "selection_changed_rate_pct": pct(safe_div(selection_changed_records, total_records)),
        "selection_changed_rate_conflict_positive": safe_div(selection_changed_conflict_positive_records, positive_conflict_records),
        "selection_changed_rate_conflict_positive_pct": pct(
            safe_div(selection_changed_conflict_positive_records, positive_conflict_records)
        ),
        "top1_kept_rate": safe_div(top1_kept_records, total_records),
        "top1_kept_rate_pct": pct(safe_div(top1_kept_records, total_records)),
        "top2_rejected_rate": safe_div(top2_rejected_records, top2_available_records),
        "top2_rejected_rate_pct": pct(safe_div(top2_rejected_records, top2_available_records)),
        "mean_selected_count": safe_div(selected_count_sum, total_records),
        "mean_topk_count": safe_div(topk_count_sum, total_records),
        "single_selected_rate": safe_div(single_selected_records, total_records),
        "single_selected_rate_pct": pct(safe_div(single_selected_records, total_records)),
        "candidate_pair_negative_rate_all": safe_div(candidate_neg_all, total_records),
        "candidate_pair_negative_rate_all_pct": pct(safe_div(candidate_neg_all, total_records)),
        "candidate_pair_negative_rate_conflict_positive": safe_div(candidate_neg_conflict_pos, positive_conflict_records),
        "candidate_pair_negative_rate_conflict_positive_pct": pct(safe_div(candidate_neg_conflict_pos, positive_conflict_records)),
        "candidate_pair_rejected_rate_all": safe_div(candidate_rejected_all, total_records),
        "candidate_pair_rejected_rate_all_pct": pct(safe_div(candidate_rejected_all, total_records)),
        "candidate_pair_rejected_rate_conflict_positive": safe_div(candidate_rejected_conflict_pos, positive_conflict_records),
        "candidate_pair_rejected_rate_conflict_positive_pct": pct(safe_div(candidate_rejected_conflict_pos, positive_conflict_records)),
        "early_mean_conflict": sum(row["mean_total_conflict"] for row in early_layers) / len(early_layers),
        "middle_mean_conflict": sum(row["mean_total_conflict"] for row in middle_layers) / len(middle_layers),
        "late_mean_conflict": sum(row["mean_total_conflict"] for row in late_layers) / len(late_layers),
        "max_layer": int(max_layer_row["layer"]),
        "max_layer_mean_conflict": max_layer_row["mean_total_conflict"],
    }

    write_csv(
        output_dir / "layer_conflict.csv",
        [
            "layer",
            "mean_total_conflict",
            "mean_filtered_conflict",
            "conflict_reduction_pct",
            "positive_conflict_rate",
            "positive_conflict_rate_pct",
            "filtered_positive_conflict_rate",
            "filtered_positive_conflict_rate_pct",
            "active_negative_rate",
            "active_negative_rate_pct",
            "selection_changed_rate",
            "selection_changed_rate_pct",
            "top1_kept_rate",
            "top1_kept_rate_pct",
            "top2_rejected_rate",
            "top2_rejected_rate_pct",
            "mean_selected_count",
            "mean_topk_count",
            "single_selected_rate",
            "single_selected_rate_pct",
        ],
        layer_rows,
    )
    write_csv(output_dir / "pair_conflict.csv", ["pair", "total_conflict", "conflict_share", "conflict_share_pct", "negative_cos_rate", "negative_cos_rate_pct", "active_negative_rate", "active_negative_rate_pct"], pair_rows)
    write_csv(
        output_dir / "layer_pair_conflict.csv",
        [
            "layer",
            "pair",
            "mean_pair_conflict",
            "mean_filtered_pair_conflict",
            "pair_conflict_reduction_pct",
            "pair_conflict_share_in_layer",
            "pair_conflict_share_in_layer_pct",
            "pair_conflict_share_global",
            "pair_conflict_share_global_pct",
            "mean_pairwise_cos",
            "negative_cos_rate",
            "negative_cos_rate_pct",
            "active_negative_rate",
            "active_negative_rate_pct",
            "positive_pair_conflict_rate",
            "positive_pair_conflict_rate_pct",
            "filtered_positive_pair_conflict_rate",
            "filtered_positive_pair_conflict_rate_pct",
        ],
        layer_pair_rows,
    )
    write_csv(output_dir / "top_demand_pair_incompatibility.csv", ["setting", "negative_cos_rate", "negative_cos_rate_pct", "rejected_rate", "rejected_rate_pct"], top_demand_rows)
    write_csv(output_dir / "candidate_pair_by_type.csv", ["pair", "candidate_count", "negative_candidate_rate", "negative_candidate_rate_pct", "rejected_candidate_rate", "rejected_candidate_rate_pct", "negative_candidate_rate_conflict_positive", "negative_candidate_rate_conflict_positive_pct", "rejected_candidate_rate_conflict_positive", "rejected_candidate_rate_conflict_positive_pct", "kept_mean_cos", "rejected_mean_cos", "kept_count", "rejected_count"], candidate_pair_rows)
    kept_vs_rejected_rows = [
        {
            "pair": row["pair"],
            "kept_mean_cos": row["kept_mean_cos"],
            "rejected_mean_cos": row["rejected_mean_cos"],
            "kept_count": row["kept_count"],
            "rejected_count": row["rejected_count"],
        }
        for row in candidate_pair_rows
    ]
    write_csv(output_dir / "kept_vs_rejected_cosine.csv", ["pair", "kept_mean_cos", "rejected_mean_cos", "kept_count", "rejected_count"], kept_vs_rejected_rows)
    write_csv(output_dir / "group_conflict.csv", ["group_type", "group_label", "count", "mean_total_conflict", "positive_conflict_rate", "positive_conflict_rate_pct", "active_negative_rate", "active_negative_rate_pct", "truthful_conflict_share", "truthful_conflict_share_pct"], group_rows)
    write_csv(output_dir / "selection_behavior.csv", ["metric_key", "metric_label", "value", "value_pct", "is_percent"], selection_metric_rows)
    write_csv(output_dir / "selection_size_distribution.csv", ["selected_size", "selected_size_label", "selected_size_count", "selected_size_rate", "selected_size_rate_pct", "topk_size_count", "topk_size_rate", "topk_size_rate_pct"], selection_size_rows)
    write_csv(
        output_dir / "routing_heatmap.csv",
        [
            "layer",
            "specialist_index",
            "specialist_label",
            "alpha_mean",
            "selected_rate",
            "selected_rate_pct",
            "topk_rate",
            "topk_rate_pct",
            "rejected_conflict_rate",
            "rejected_conflict_rate_pct",
        ],
        routing_rows,
    )
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    draw_main_figure(output_dir / args.stem, layer_rows, pair_rows, top_demand_rows, summary)
    draw_layer_metrics_panel(output_dir / f"{args.stem}_layer_metrics_panel", layer_rows, summary)
    draw_layer_pair_heatmaps(output_dir / f"{args.stem}_layer_pair_heatmaps", layer_pair_rows)
    draw_pair_matrices(output_dir / f"{args.stem}_pair_matrices", pair_rows, candidate_pair_rows)
    draw_candidate_pair_panel(output_dir / f"{args.stem}_candidate_pair_panel", top_demand_rows, candidate_pair_rows)
    draw_group_split_panel(output_dir / f"{args.stem}_group_split_panel", group_rows)
    draw_filter_conflict_panel(output_dir / f"{args.stem}_filter_conflict_panel", layer_rows, summary)
    draw_selection_behavior_panel(output_dir / f"{args.stem}_selection_behavior_panel", summary, selection_metric_rows, selection_size_rows, layer_rows)
    draw_before_after_conflict_panel(output_dir / f"{args.stem}_before_after_conflict_panel", summary)
    draw_layerwise_before_after_conflict_panel(output_dir / f"{args.stem}_layerwise_before_after_conflict_panel", layer_rows)
    draw_layerwise_filter_action_panel(output_dir / f"{args.stem}_layerwise_filter_action_panel", layer_rows)
    draw_before_after_selection_panel(output_dir / f"{args.stem}_before_after_selection_panel", summary, selection_size_rows, layer_rows)
    draw_layer_pair_before_after_heatmap(output_dir / f"{args.stem}_layer_pair_before_after_heatmap", layer_pair_rows)
    draw_routing_heatmap_panel(output_dir / f"{args.stem}_routing_heatmap_panel", specialist_labels, routing_rows)
    draw_publication_heatmap_strip(output_dir / f"{args.stem}_publication_heatmap_strip", specialist_labels, routing_rows, layer_pair_rows)
    draw_showcase_panel(output_dir / f"{args.stem}_showcase_panel", layer_rows, pair_rows, candidate_pair_rows, top_demand_rows)

    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
