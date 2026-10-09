#!/usr/bin/env python3
"""
plot_prediction_results.py
==========================
Generate publication-quality visualizations of the leakage impact experiments.

Figures:
1. F1 delta (baseline - cleaned) by project, strategy, and dimension
2. Threshold sweep curves showing F1 delta vs similarity threshold
3. Embedding-based vs same-code leakage comparison
4. Per-pair F1 trends for baseline vs cleaned

Usage:
    python src/plot_prediction_results.py
    python src/plot_prediction_results.py --format pdf --dpi 300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STRATEGY_DISPLAY = {"pairwise": "pairwise", "cumulative-fresh": "cumulative"}


def load_all_summaries(pred_dir: Path) -> pd.DataFrame:
    """Load all summary CSVs into a single DataFrame with config metadata."""
    rows = []
    for f in pred_dir.glob("*_summary.csv"):
        parts = f.stem.replace("_summary", "").split("_")
        # Parse: {project}_{strategy}_{dim}
        project = parts[0]
        if "cumulative-fresh" in f.stem:
            strategy = "cumulative-fresh"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"
        else:
            strategy = "pairwise"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"

        df = pd.read_csv(f)
        df["project"] = project
        df["strategy"] = strategy
        df["dim"] = dim
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_all_predictions(pred_dir: Path) -> pd.DataFrame:
    """Load all per-pair prediction CSVs."""
    rows = []
    for f in pred_dir.glob("*_predictions.csv"):
        parts = f.stem.replace("_predictions", "").split("_")
        project = parts[0]
        if "cumulative-fresh" in f.stem:
            strategy = "cumulative-fresh"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"
        else:
            strategy = "pairwise"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"

        df = pd.read_csv(f)
        df["project"] = project
        df["strategy"] = strategy
        df["dim"] = dim
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_all_sweeps(pred_dir: Path) -> pd.DataFrame:
    """Load all threshold sweep CSVs."""
    rows = []
    for f in pred_dir.glob("*_threshold_sweep.csv"):
        parts = f.stem.replace("_threshold_sweep", "").split("_")
        project = parts[0]
        if "cumulative-fresh" in f.stem:
            strategy = "cumulative-fresh"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"
        else:
            strategy = "pairwise"
            dim = parts[-1] if parts[-1].startswith("dim") else "unknown"

        df = pd.read_csv(f)
        df["project"] = project
        df["strategy"] = strategy
        df["dim"] = dim
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_f1_delta_bar(summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Bar chart: F1 delta (baseline - cleaned) for each configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, clf in zip(axes, ["RandomForest", "LogisticRegression"]):
        sub = summaries[
            (summaries["classifier"] == clf)
            & (summaries["leakage_method"] == "embedding_0.99")
        ]

        configs = []
        deltas = []
        colors = []
        color_map = {
            ("django", "pairwise"): "#2196F3",
            ("django", "cumulative-fresh"): "#64B5F6",
            ("calcite", "pairwise"): "#FF9800",
            ("calcite", "cumulative-fresh"): "#FFB74D",
        }

        for _, row_group in sub.groupby(["project", "strategy", "dim"]):
            grp = row_group.set_index("subset")
            if "baseline" not in grp.index or "cleaned" not in grp.index:
                continue
            bl = grp.loc["baseline", "f1_macro"]
            cl = grp.loc["cleaned", "f1_macro"]
            proj = row_group["project"].iloc[0]
            strat = row_group["strategy"].iloc[0]
            dim = row_group["dim"].iloc[0]
            label = f"{proj.title()}\n{STRATEGY_DISPLAY.get(strat, strat)}\n{dim}"
            configs.append(label)
            deltas.append(bl - cl)
            colors.append(color_map.get((proj, strat), "#999999"))

        # Sort by delta descending
        order = np.argsort(deltas)[::-1]
        configs = [configs[i] for i in order]
        deltas = [deltas[i] for i in order]
        colors = [colors[i] for i in order]

        bars = ax.bar(
            range(len(configs)), deltas, color=colors, edgecolor="black", linewidth=0.5
        )
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=8)
        ax.set_ylabel("F1 Delta (baseline - cleaned)" if ax == axes[0] else "")
        ax.set_title(clf, fontsize=12, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"+{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "F1 Inflation from Train/Test Leakage (embedding @0.99)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"f1_delta_bar.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Wrote f1_delta_bar.{fmt}")


def plot_threshold_sweep(sweeps: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Line plot: F1 delta vs threshold for each configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    clf = "RandomForest"
    styles = {
        "pairwise": "-",
        "cumulative-fresh": "--",
    }  # keyed by internal name, not display name
    project_colors = {
        "django": {"dim200": "#1565C0", "dim400": "#42A5F5"},
        "calcite": {"dim200": "#E65100", "dim400": "#FFA726"},
    }

    # Left panel: delta
    ax = axes[0]
    for (proj, strat, dim), grp in sweeps.groupby(["project", "strategy", "dim"]):
        sub = grp[(grp["classifier"] == clf)]
        baseline = (
            sub[sub["subset"] == "baseline"].groupby("threshold")["f1_macro"].mean()
        )
        cleaned = (
            sub[sub["subset"] == "cleaned"].groupby("threshold")["f1_macro"].mean()
        )
        delta = baseline - cleaned
        delta = delta.sort_index()

        color = project_colors.get(proj, {}).get(dim, "#999")
        ls = styles.get(strat, "-")
        label = f"{proj.title()} {STRATEGY_DISPLAY.get(strat, strat)} ({dim})"
        ax.plot(
            delta.index,
            delta.values,
            color=color,
            linestyle=ls,
            marker="o",
            markersize=4,
            label=label,
            linewidth=1.5,
        )

    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("F1 Delta (baseline - cleaned)")
    ax.set_title("Metric Inflation vs Threshold", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0.895, 1.005)
    ax.grid(True, alpha=0.3)

    # Right panel: leaked-only F1
    ax = axes[1]
    for (proj, strat, dim), grp in sweeps.groupby(["project", "strategy", "dim"]):
        sub = grp[(grp["classifier"] == clf)]
        leaked = (
            sub[sub["subset"] == "leaked-only"].groupby("threshold")["f1_macro"].mean()
        )
        leaked = leaked.sort_index()

        color = project_colors.get(proj, {}).get(dim, "#999")
        ls = styles.get(strat, "-")
        label = f"{proj.title()} {STRATEGY_DISPLAY.get(strat, strat)} ({dim})"
        ax.plot(
            leaked.index,
            leaked.values,
            color=color,
            linestyle=ls,
            marker="s",
            markersize=4,
            label=label,
            linewidth=1.5,
        )

    ax.set_xlabel("Cosine Similarity Threshold")
    ax.set_ylabel("F1 (leaked-only subset)")
    ax.set_title(
        "Leaked Files: Classification Performance", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0.895, 1.005)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Threshold Sweep — RandomForest", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"threshold_sweep.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Wrote threshold_sweep.{fmt}")


def plot_leakage_method_comparison(
    summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int
):
    """Grouped bar chart: embedding-based vs same-code leakage definition."""
    # Only pairwise has same-code data
    sub = summaries[
        (summaries["strategy"] == "pairwise")
        & (summaries["classifier"] == "RandomForest")
        & (summaries["subset"].isin(["baseline", "cleaned", "leaked-only"]))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, proj in zip(axes, ["django", "calcite"]):
        proj_data = sub[sub["project"] == proj]
        dims = sorted(proj_data["dim"].unique())

        x_positions = []
        x_labels = []
        offset = 0

        for dim in dims:
            dim_data = proj_data[proj_data["dim"] == dim]
            methods = sorted(dim_data["leakage_method"].unique())

            for method in methods:
                method_data = dim_data[dim_data["leakage_method"] == method]
                subsets = method_data.set_index("subset")

                bl = (
                    subsets.loc["baseline", "f1_macro"]
                    if "baseline" in subsets.index
                    else 0
                )
                cl = (
                    subsets.loc["cleaned", "f1_macro"]
                    if "cleaned" in subsets.index
                    else 0
                )
                lo = (
                    subsets.loc["leaked-only", "f1_macro"]
                    if "leaked-only" in subsets.index
                    else 0
                )

                width = 0.25
                positions = [offset, offset + width, offset + 2 * width]
                colors = ["#2196F3", "#4CAF50", "#F44336"]
                ax.bar(
                    positions,
                    [bl, cl, lo],
                    width=width * 0.9,
                    color=colors,
                    edgecolor="black",
                    linewidth=0.5,
                )

                method_label = method.replace("embedding_0.99", "emb@0.99").replace(
                    "same_code", "same-code"
                )
                x_positions.append(offset + width)
                x_labels.append(f"{dim}\n{method_label}")
                offset += 1.2

            offset += 0.4

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel("F1 (macro)")
        ax.set_title(f"{proj.title()} — Pairwise", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", edgecolor="black", label="Baseline"),
        Patch(facecolor="#4CAF50", edgecolor="black", label="Cleaned"),
        Patch(facecolor="#F44336", edgecolor="black", label="Leaked-only"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.suptitle(
        "Leakage Definition Comparison (RandomForest)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"leakage_method_comparison.{fmt}", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"  Wrote leakage_method_comparison.{fmt}")


def plot_per_pair_f1(predictions: pd.DataFrame, output_dir: Path, fmt: str, dpi: int):
    """Per-pair F1 line plots showing baseline vs cleaned across version pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        ("django", "pairwise", "dim400"),
        ("django", "pairwise", "dim200"),
        ("calcite", "pairwise", "dim200"),
        ("calcite", "pairwise", "dim400"),
    ]

    for ax, (proj, strat, dim) in zip(axes.flat, configs):
        sub = predictions[
            (predictions["project"] == proj)
            & (predictions["strategy"] == strat)
            & (predictions["dim"] == dim)
            & (predictions["classifier"] == "RandomForest")
            & (predictions["leakage_method"] == "embedding_0.99")
        ]

        for subset, color, marker in [
            ("baseline", "#2196F3", "o"),
            ("cleaned", "#4CAF50", "s"),
            ("leaked-only", "#F44336", "^"),
        ]:
            s = sub[sub["subset"] == subset].sort_values("pair")
            ax.plot(
                s["pair"],
                s["f1_macro"],
                color=color,
                marker=marker,
                markersize=4,
                label=subset,
                linewidth=1,
                alpha=0.8,
            )

        ax.set_xlabel("Version Pair")
        ax.set_ylabel("F1 (macro)")
        ax.set_title(
            f"{proj.title()} — {STRATEGY_DISPLAY.get(strat, strat)} ({dim})",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Per-Pair Classification Performance (RandomForest, emb@0.99)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"per_pair_f1.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  Wrote per_pair_f1.{fmt}")


def plot_dimension_comparison(
    summaries: pd.DataFrame, output_dir: Path, fmt: str, dpi: int
):
    """Side-by-side comparison of 200-dim vs 400-dim results."""
    sub = summaries[
        (summaries["classifier"] == "RandomForest")
        & (summaries["leakage_method"] == "embedding_0.99")
        & (summaries["subset"].isin(["baseline", "cleaned"]))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, proj in zip(axes, ["django", "calcite"]):
        proj_data = sub[sub["project"] == proj]

        strategies = sorted(proj_data["strategy"].unique())
        x = np.arange(len(strategies))
        width = 0.18

        for i, (dim, color_bl, color_cl) in enumerate(
            [
                ("dim200", "#1565C0", "#4CAF50"),
                ("dim400", "#42A5F5", "#81C784"),
            ]
        ):
            for j, subset in enumerate(["baseline", "cleaned"]):
                vals = []
                for strat in strategies:
                    v = proj_data[
                        (proj_data["strategy"] == strat)
                        & (proj_data["dim"] == dim)
                        & (proj_data["subset"] == subset)
                    ]["f1_macro"]
                    vals.append(v.values[0] if len(v) > 0 else 0)

                color = color_bl if subset == "baseline" else color_cl
                offset = (i * 2 + j - 1.5) * width
                hatch = "" if subset == "baseline" else "//"
                bars = ax.bar(
                    x + offset,
                    vals,
                    width * 0.9,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    hatch=hatch,
                )

                for bar, val in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        rotation=45,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_DISPLAY.get(s, s) for s in strategies], fontsize=9)
        ax.set_ylabel("F1 (macro)")
        ax.set_title(f"{proj.title()}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, axis="y")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#1565C0", edgecolor="black", label="dim200 baseline"),
        Patch(facecolor="#4CAF50", edgecolor="black", label="dim200 cleaned"),
        Patch(facecolor="#42A5F5", edgecolor="black", label="dim400 baseline"),
        Patch(facecolor="#81C784", edgecolor="black", label="dim400 cleaned"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle(
        "Embedding Dimensionality Comparison (RandomForest, emb@0.99)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"dimension_comparison.{fmt}", dpi=dpi, bbox_inches="tight"
    )
    plt.close()
    print(f"  Wrote dimension_comparison.{fmt}")


def main():
    parser = argparse.ArgumentParser(description="Plot prediction experiment results.")
    parser.add_argument(
        "--pred-dir", default="results/prediction", help="Prediction results directory"
    )
    parser.add_argument(
        "--output", default="results/plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf"], help="Output format"
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    summaries = load_all_summaries(pred_dir)
    predictions = load_all_predictions(pred_dir)
    sweeps = load_all_sweeps(pred_dir)

    print(f"  Summaries: {len(summaries)} rows")
    print(f"  Predictions: {len(predictions)} rows")
    print(f"  Sweeps: {len(sweeps)} rows")

    print("\nGenerating plots...")
    plot_f1_delta_bar(summaries, output_dir, args.format, args.dpi)
    plot_threshold_sweep(sweeps, output_dir, args.format, args.dpi)
    plot_leakage_method_comparison(summaries, output_dir, args.format, args.dpi)
    plot_per_pair_f1(predictions, output_dir, args.format, args.dpi)
    plot_dimension_comparison(summaries, output_dir, args.format, args.dpi)

    print("\nDone!")


if __name__ == "__main__":
    main()
