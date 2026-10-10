#!/usr/bin/env python3
"""
plot_subset_results.py
======================
Generate visualizations for the filename-based subset experiments.

Figures:
1. F1 by subset (baseline / new_files / changed_label) across configs
2. MCC by subset — highlights negative correlation for changed_label
3. Per-pair F1 trends across version pairs for each subset
4. Dimension comparison (200 vs 400) per project

Usage:
    python src/plot_subset_results.py
    python src/plot_subset_results.py --format pdf --dpi 300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_DISPLAY = {"pairwise": "pairwise", "cumulative-fresh": "cumulative"}

# Primary dimension for both projects
PRIMARY_DIM = {"django": "200", "calcite": "200"}

SUBSET_COLORS = {
    "baseline": "#2196F3",
    "new_files": "#FF9800",
    "changed_label": "#F44336",
}

SUBSET_LABELS = {
    "baseline": "Baseline",
    "new_files": "New files",
    "changed_label": "Changed label",
}


def parse_config(filename: str) -> dict:
    """Parse project, strategy, and dimension from a results filename."""
    stem = filename.replace("_summary", "").replace("_predictions", "")

    if "cumulative-fresh" in stem:
        strategy = "cumulative-fresh"
        rest = stem.replace("cumulative-fresh_", "").split("_")
    else:
        strategy = "pairwise"
        rest = stem.replace("pairwise_", "").split("_")

    project = rest[0]
    dim_part = rest[-1] if len(rest) > 1 and rest[-1].startswith("dim") else None
    dim = dim_part.replace("dim", "") if dim_part else PRIMARY_DIM.get(project, "?")

    return {"project": project, "strategy": strategy, "dim": dim}


def load_summaries(pred_dir: Path) -> pd.DataFrame:
    """Load all new-format summary CSVs (no leakage_method column)."""
    rows = []
    for f in pred_dir.glob("*_summary.csv"):
        df = pd.read_csv(f)
        if "leakage_method" in df.columns:
            continue  # skip old-format files
        config = parse_config(f.stem)
        df["project"] = config["project"]
        df["strategy"] = config["strategy"]
        df["dim"] = config["dim"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """Load all new-format per-pair prediction CSVs."""
    rows = []
    for f in pred_dir.glob("*_predictions.csv"):
        df = pd.read_csv(f)
        if "leakage_method" in df.columns:
            continue
        config = parse_config(f.stem)
        df["project"] = config["project"]
        df["strategy"] = config["strategy"]
        df["dim"] = config["dim"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_f1_by_subset(summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Grouped bar chart: F1 for each subset across project x strategy (primary dims)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("F1 Macro by Evaluation Subset — RandomForest", fontsize=14, y=1.02)

    subsets = ["baseline", "new_files", "changed_label"]
    bar_width = 0.22

    for ax, project in zip(axes, ["django", "calcite"]):
        primary = PRIMARY_DIM[project]
        sub = summaries[
            (summaries["project"] == project) & (summaries["dim"] == primary)
        ]

        strategies = ["pairwise", "cumulative-fresh"]
        x = np.arange(len(strategies))

        for i, subset in enumerate(subsets):
            vals = []
            for strategy in strategies:
                row = sub[(sub["strategy"] == strategy) & (sub["subset"] == subset)]
                vals.append(row["f1_macro"].values[0] if len(row) > 0 else 0)
            ax.bar(
                x + i * bar_width,
                vals,
                bar_width,
                label=SUBSET_LABELS[subset],
                color=SUBSET_COLORS[subset],
                alpha=0.85,
            )

        ax.set_xlabel("Strategy")
        ax.set_ylabel("F1 Macro" if project == "django" else "")
        ax.set_title(f"{project.title()} (dim {primary})")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies])
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"subset_f1_comparison.{fmt}", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"  Wrote subset_f1_comparison.{fmt}")


def plot_mcc_by_subset(summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Grouped bar chart: MCC for each subset — shows negative MCC for changed_label."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Matthews Correlation Coefficient by Evaluation Subset — RandomForest",
        fontsize=14,
        y=1.02,
    )

    subsets = ["baseline", "new_files", "changed_label"]
    bar_width = 0.22

    for ax, project in zip(axes, ["django", "calcite"]):
        primary = PRIMARY_DIM[project]
        sub = summaries[
            (summaries["project"] == project) & (summaries["dim"] == primary)
        ]

        strategies = ["pairwise", "cumulative-fresh"]
        x = np.arange(len(strategies))

        for i, subset in enumerate(subsets):
            vals = []
            for strategy in strategies:
                row = sub[(sub["strategy"] == strategy) & (sub["subset"] == subset)]
                vals.append(row["mcc"].values[0] if len(row) > 0 else 0)
            ax.bar(
                x + i * bar_width,
                vals,
                bar_width,
                label=SUBSET_LABELS[subset],
                color=SUBSET_COLORS[subset],
                alpha=0.85,
            )

        ax.set_xlabel("Strategy")
        ax.set_ylabel("MCC" if project == "django" else "")
        ax.set_title(f"{project.title()} (dim {primary})")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies])
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"subset_mcc_comparison.{fmt}", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"  Wrote subset_mcc_comparison.{fmt}")


def plot_per_pair_f1(predictions: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Per-pair F1 trends for each subset, one panel per project x strategy (primary dims)."""
    configs = [
        ("django", "pairwise"),
        ("django", "cumulative-fresh"),
        ("calcite", "pairwise"),
        ("calcite", "cumulative-fresh"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    fig.suptitle(
        "F1 Macro per Version Pair by Subset — RandomForest", fontsize=14, y=1.02
    )

    subsets = ["baseline", "new_files", "changed_label"]

    for ax, (project, strategy) in zip(axes.flat, configs):
        primary = PRIMARY_DIM[project]
        sub = predictions[
            (predictions["project"] == project)
            & (predictions["strategy"] == strategy)
            & (predictions["dim"] == primary)
        ]

        for subset in subsets:
            s = sub[sub["subset"] == subset].sort_values("pair")
            if len(s) > 0:
                ax.plot(
                    s["pair"],
                    s["f1_macro"],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    label=SUBSET_LABELS[subset],
                    color=SUBSET_COLORS[subset],
                    alpha=0.8,
                )

        strategy_label = STRATEGY_DISPLAY[strategy]
        ax.set_title(f"{project.title()} — {strategy_label} (dim {primary})")
        ax.set_xlabel("Version pair")
        ax.set_ylabel("F1 Macro")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"subset_per_pair_f1.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Wrote subset_per_pair_f1.{fmt}")


def plot_auc_by_subset(summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Grouped bar chart: AUC for each subset across project x strategy (primary dims)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("AUC by Evaluation Subset — RandomForest", fontsize=14, y=1.02)

    subsets = ["baseline", "new_files", "changed_label"]
    bar_width = 0.22

    for ax, project in zip(axes, ["django", "calcite"]):
        primary = PRIMARY_DIM[project]
        sub = summaries[
            (summaries["project"] == project) & (summaries["dim"] == primary)
        ]

        strategies = ["pairwise", "cumulative-fresh"]
        x = np.arange(len(strategies))

        for i, subset in enumerate(subsets):
            vals = []
            for strategy in strategies:
                row = sub[(sub["strategy"] == strategy) & (sub["subset"] == subset)]
                vals.append(row["auc"].values[0] if len(row) > 0 else 0)
            ax.bar(
                x + i * bar_width,
                vals,
                bar_width,
                label=SUBSET_LABELS[subset],
                color=SUBSET_COLORS[subset],
                alpha=0.85,
            )

        ax.set_xlabel("Strategy")
        ax.set_ylabel("AUC" if project == "django" else "")
        ax.set_title(f"{project.title()} (dim {primary})")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies])
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"subset_auc_comparison.{fmt}", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"  Wrote subset_auc_comparison.{fmt}")


def plot_per_pair_auc(predictions: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Per-pair AUC trends for each subset, one panel per project x strategy (primary dims)."""
    configs = [
        ("django", "pairwise"),
        ("django", "cumulative-fresh"),
        ("calcite", "pairwise"),
        ("calcite", "cumulative-fresh"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    fig.suptitle("AUC per Version Pair by Subset — RandomForest", fontsize=14, y=1.02)

    subsets = ["baseline", "new_files", "changed_label"]

    for ax, (project, strategy) in zip(axes.flat, configs):
        primary = PRIMARY_DIM[project]
        sub = predictions[
            (predictions["project"] == project)
            & (predictions["strategy"] == strategy)
            & (predictions["dim"] == primary)
        ]

        for subset in subsets:
            s = sub[sub["subset"] == subset].sort_values("pair")
            if len(s) > 0 and "auc" in s.columns:
                ax.plot(
                    s["pair"],
                    s["auc"],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    label=SUBSET_LABELS[subset],
                    color=SUBSET_COLORS[subset],
                    alpha=0.8,
                )

        strategy_label = STRATEGY_DISPLAY[strategy]
        ax.set_title(f"{project.title()} — {strategy_label} (dim {primary})")
        ax.set_xlabel("Version pair")
        ax.set_ylabel("AUC")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f"subset_per_pair_auc.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Wrote subset_per_pair_auc.{fmt}")


def plot_dimension_comparison(
    summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int
):
    """Side-by-side dim 200 vs 400 for each subset, grouped by project."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("F1 Macro: Dimension 200 vs 400 — RandomForest", fontsize=14, y=1.02)

    subsets = ["baseline", "new_files", "changed_label"]
    bar_width = 0.12

    for ax, project in zip(axes, ["django", "calcite"]):
        sub = summaries[summaries["project"] == project]

        # x positions: one group per strategy, within each group: subset pairs (dim200, dim400)
        strategies = ["pairwise", "cumulative-fresh"]
        group_width = len(subsets) * 2 * bar_width + 0.15
        x_groups = np.arange(len(strategies)) * (group_width + 0.3)

        for si, subset in enumerate(subsets):
            for di, (dim, hatch) in enumerate([("200", ""), ("400", "//")]):
                vals = []
                for strategy in strategies:
                    row = sub[
                        (sub["strategy"] == strategy)
                        & (sub["subset"] == subset)
                        & (sub["dim"] == dim)
                    ]
                    vals.append(row["f1_macro"].values[0] if len(row) > 0 else 0)

                offset = si * 2 * bar_width + di * bar_width
                ax.bar(
                    x_groups + offset,
                    vals,
                    bar_width,
                    label=f"{SUBSET_LABELS[subset]} (dim {dim})",
                    color=SUBSET_COLORS[subset],
                    alpha=0.7 if di == 0 else 0.45,
                    hatch=hatch,
                    edgecolor="white" if not hatch else SUBSET_COLORS[subset],
                )

        ax.set_xlabel("Strategy")
        ax.set_ylabel("F1 Macro" if project == "django" else "")
        ax.set_title(f"{project.title()}")
        center_offset = (len(subsets) * 2 * bar_width - bar_width) / 2
        ax.set_xticks(x_groups + center_offset)
        ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies])
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen[lbl] = True
                unique_handles.append(h)
                unique_labels.append(lbl)
        ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"subset_dimension_comparison.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Wrote subset_dimension_comparison.{fmt}")


def plot_dimension_comparison_mcc(
    summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int
):
    """Side-by-side dim 200 vs 400 for MCC by subset, grouped by project."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("MCC: Dimension 200 vs 400 — RandomForest", fontsize=14, y=1.02)

    subsets = ["baseline", "new_files", "changed_label"]
    bar_width = 0.12

    for ax, project in zip(axes, ["django", "calcite"]):
        sub = summaries[summaries["project"] == project]

        strategies = ["pairwise", "cumulative-fresh"]
        group_width = len(subsets) * 2 * bar_width + 0.15
        x_groups = np.arange(len(strategies)) * (group_width + 0.3)

        for si, subset in enumerate(subsets):
            for di, (dim, hatch) in enumerate([("200", ""), ("400", "//")]):
                vals = []
                for strategy in strategies:
                    row = sub[
                        (sub["strategy"] == strategy)
                        & (sub["subset"] == subset)
                        & (sub["dim"] == dim)
                    ]
                    vals.append(row["mcc"].values[0] if len(row) > 0 else 0)

                offset = si * 2 * bar_width + di * bar_width
                ax.bar(
                    x_groups + offset,
                    vals,
                    bar_width,
                    label=f"{SUBSET_LABELS[subset]} (dim {dim})",
                    color=SUBSET_COLORS[subset],
                    alpha=0.7 if di == 0 else 0.45,
                    hatch=hatch,
                    edgecolor="white" if not hatch else SUBSET_COLORS[subset],
                )

        ax.set_xlabel("Strategy")
        ax.set_ylabel("MCC" if project == "django" else "")
        ax.set_title(f"{project.title()}")
        center_offset = (len(subsets) * 2 * bar_width - bar_width) / 2
        ax.set_xticks(x_groups + center_offset)
        ax.set_xticklabels([STRATEGY_DISPLAY[s] for s in strategies])
        ax.set_ylim(-1.0, 1.0)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)

        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        unique_handles, unique_labels = [], []
        for h, lbl in zip(handles, labels):
            if lbl not in seen:
                seen[lbl] = True
                unique_handles.append(h)
                unique_labels.append(lbl)
        ax.legend(unique_handles, unique_labels, loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"subset_dimension_comparison_mcc.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Wrote subset_dimension_comparison_mcc.{fmt}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot filename-based subset experiment results."
    )
    parser.add_argument(
        "--pred-dir",
        default="results/prediction",
        help="Directory with prediction CSVs",
    )
    parser.add_argument(
        "--output-dir", default="results/plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf"], help="Output format"
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_summaries(pred_dir)
    predictions = load_predictions(pred_dir)

    if summaries.empty:
        print("No new-format summary files found")
        return

    print("Generating plots...")
    plot_f1_by_subset(summaries, output_dir, args.format, args.dpi)
    plot_mcc_by_subset(summaries, output_dir, args.format, args.dpi)
    if "auc" in summaries.columns:
        plot_auc_by_subset(summaries, output_dir, args.format, args.dpi)
    if not predictions.empty:
        plot_per_pair_f1(predictions, output_dir, args.format, args.dpi)
        if "auc" in predictions.columns:
            plot_per_pair_auc(predictions, output_dir, args.format, args.dpi)
    plot_dimension_comparison(summaries, output_dir, args.format, args.dpi)
    plot_dimension_comparison_mcc(summaries, output_dir, args.format, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
