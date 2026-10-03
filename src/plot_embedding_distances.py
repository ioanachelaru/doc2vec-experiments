#!/usr/bin/env python3
"""
plot_embedding_distances.py
===========================
Generate visualizations of embedding similarity distributions across
strategies and projects.

Reads the per-file similarity CSVs produced by analyze_embedding_distances.py
and generates publication-quality figures.

Figures:
  1. Similarity distribution (KDE) — one per project
  2. Per-pair mean similarity trend — one per project
  3. Threshold exceedance (% >= 0.99) — one per project
  4. Aggregate summary bar chart — one total
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

PROJECTS = ["django", "calcite"]
STRATEGIES = ["pairwise", "cumulative-fresh", "cumulative-carried"]

# Strategy -> (display name, color, marker)
STRATEGY_STYLE = {
    "pairwise": ("Pairwise", "#1f77b4", "o"),
    "cumulative-fresh": ("Cumulative-fresh", "#ff7f0e", "s"),
    "cumulative-carried": ("Cumulative-carried", "#2ca02c", "^"),
}

PROJECT_TITLES = {
    "django": "Django",
    "calcite": "Apache Calcite",
}


def _load_similarity_data(
    results_dir: Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load per-file similarity CSVs.

    Returns:
        {project: {strategy: DataFrame}}
    """
    data: dict[str, dict[str, pd.DataFrame]] = {}

    for project in PROJECTS:
        for strategy in STRATEGIES:
            filename = f"embedding_distances_{project}_{strategy.replace('-', '_')}.csv"
            path = results_dir / filename
            if not path.exists():
                continue

            df = pd.read_csv(path, comment="#")
            data.setdefault(project, {})[strategy] = df
            log.info(f"  Loaded {filename}: {len(df)} rows")

    return data


def _compute_pair_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pair statistics from per-file similarity data."""
    stats = (
        df.groupby("pair")
        .agg(
            mean_sim=("similarity", "mean"),
            median_sim=("similarity", "median"),
            matched_files=("similarity", "count"),
            pct_above_099=("similarity", lambda x: (x >= 0.99).mean() * 100),
            pct_above_095=("similarity", lambda x: (x >= 0.95).mean() * 100),
        )
        .reset_index()
    )
    return stats


def plot_similarity_distribution(
    project_data: dict[str, pd.DataFrame],
    project: str,
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Plot overlaid KDE curves for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in STRATEGIES:
        if strategy not in project_data:
            continue
        df = project_data[strategy]
        name, color, _ = STRATEGY_STYLE[strategy]
        sns.kdeplot(
            data=df["similarity"],
            ax=ax,
            label=name,
            color=color,
            linewidth=2,
            clip=(0, 1),
        )

    ax.axvline(x=0.99, color="red", linestyle="--", alpha=0.7, label="0.99 threshold")
    ax.axvline(x=0.95, color="gray", linestyle=":", alpha=0.7, label="0.95 threshold")

    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{PROJECT_TITLES[project]} — Embedding Similarity Distribution",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.02)

    plt.tight_layout()
    path = output_dir / f"similarity_distribution_{project}.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info(f"  Saved {path}")


def plot_mean_similarity_trend(
    project_data: dict[str, pd.DataFrame],
    project: str,
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Plot per-pair mean similarity as line chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in STRATEGIES:
        if strategy not in project_data:
            continue
        stats = _compute_pair_stats(project_data[strategy])
        name, color, marker = STRATEGY_STYLE[strategy]
        ax.plot(
            stats["pair"],
            stats["mean_sim"],
            label=name,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
        )

    ax.set_xlabel("Version Pair", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title(
        f"{PROJECT_TITLES[project]} — Mean Embedding Similarity per Version Pair",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    path = output_dir / f"mean_similarity_trend_{project}.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info(f"  Saved {path}")


def plot_threshold_exceedance(
    project_data: dict[str, pd.DataFrame],
    project: str,
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Plot grouped bar chart of % files above 0.99 per pair."""
    fig, ax = plt.subplots(figsize=(12, 6))

    available = [s for s in STRATEGIES if s in project_data]
    n_strategies = len(available)
    if n_strategies == 0:
        return

    # Get all pair numbers from the strategy with the most pairs
    all_pairs = sorted(
        set().union(*(project_data[s]["pair"].unique() for s in available))
    )
    n_pairs = len(all_pairs)
    bar_width = 0.8 / n_strategies
    x = np.arange(n_pairs)

    for j, strategy in enumerate(available):
        stats = _compute_pair_stats(project_data[strategy])
        name, color, _ = STRATEGY_STYLE[strategy]

        # Align stats to all_pairs (some strategies may have fewer)
        pct_values = []
        for pair in all_pairs:
            row = stats[stats["pair"] == pair]
            pct_values.append(row["pct_above_099"].values[0] if len(row) > 0 else 0)

        offset = (j - n_strategies / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            pct_values,
            bar_width,
            label=name,
            color=color,
            alpha=0.85,
        )

    ax.set_xlabel("Version Pair", fontsize=12)
    ax.set_ylabel("% Files with Similarity >= 0.99", fontsize=12)
    ax.set_title(
        f"{PROJECT_TITLES[project]} — Near-Duplicate Rate per Version Pair",
        fontsize=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_pairs, fontsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = output_dir / f"threshold_exceedance_{project}.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info(f"  Saved {path}")


def plot_aggregate_summary(
    all_data: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    fmt: str,
    dpi: int,
) -> None:
    """Plot aggregate summary: avg mean similarity by strategy and project."""
    fig, ax = plt.subplots(figsize=(10, 6))

    available_projects = [p for p in PROJECTS if p in all_data]
    n_projects = len(available_projects)
    if n_projects == 0:
        return

    x = np.arange(len(STRATEGIES))
    bar_width = 0.8 / n_projects
    project_colors = {"django": "#4878d0", "calcite": "#ee854a"}

    for i, project in enumerate(available_projects):
        values = []
        for strategy in STRATEGIES:
            if strategy in all_data.get(project, {}):
                df = all_data[project][strategy]
                values.append(df["similarity"].mean())
            else:
                values.append(0)

        offset = (i - n_projects / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            values,
            bar_width,
            label=PROJECT_TITLES[project],
            color=project_colors[project],
            alpha=0.85,
        )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Average Cosine Similarity", fontsize=12)
    ax.set_title("Aggregate Embedding Similarity by Strategy and Project", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_STYLE[s][0] for s in STRATEGIES],
        fontsize=11,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = output_dir / f"aggregate_summary.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot embedding similarity distributions."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing similarity CSVs (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory for output figures (default: results/plots/)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf"],
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150, use 300 for paper)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading similarity data...")
    all_data = _load_similarity_data(results_dir)

    if not all_data:
        log.error("No data found. Run analyze_embedding_distances.py first.")
        sys.exit(1)

    log.info("\nGenerating figures...")

    for project in PROJECTS:
        if project not in all_data:
            continue

        project_data = all_data[project]
        plot_similarity_distribution(
            project_data, project, output_dir, args.format, args.dpi
        )
        plot_mean_similarity_trend(
            project_data, project, output_dir, args.format, args.dpi
        )
        plot_threshold_exceedance(
            project_data, project, output_dir, args.format, args.dpi
        )

    plot_aggregate_summary(all_data, output_dir, args.format, args.dpi)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
