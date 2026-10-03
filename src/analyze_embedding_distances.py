#!/usr/bin/env python3
"""
analyze_embedding_distances.py
==============================
Compute cosine similarity distributions for same-file pairs across versions.

For every file that exists in both train and test (same relative path across
consecutive versions), compute the cosine similarity between their Doc2Vec
embedding vectors — no threshold cutoff. This produces a continuous distribution
showing how much files change between versions in embedding space.

Supports three strategies:
  - Pairwise: fresh base model per consecutive pair
  - Cumulative-fresh: fresh base model, growing version window
  - Cumulative-carried: model carried forward across versions
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

REPO = "ioanachelaru/doc2vec-experiments"

# Artifact directories and their strategies/projects
ARTIFACT_CONFIGS = {
    "pairwise": {
        "django": "pairwise-django",
        "calcite": "pairwise-calcite",
    },
    "cumulative-fresh": {
        "django": "cumulative-fresh-django",
        "calcite": "cumulative-fresh-calcite",
    },
    "cumulative-carried": {
        "django": "cross-version-django",
        "calcite": "cross-version-calcite",
    },
}


def _strip_version_prefix(file_path: str) -> str:
    """Strip the version prefix directory from a file path.

    Examples:
        '1.0/django/__init__.py' -> 'django/__init__.py'
        'calcite-1.0.0-incubating/spark/src/...' -> 'spark/src/...'
    """
    parts = file_path.split("/", 1)
    return parts[1] if len(parts) > 1 else file_path


def _find_metadata(artifact_dir: Path, strategy: str) -> dict | None:
    """Find and load the metadata JSON from an artifact directory."""
    if strategy == "cumulative-carried":
        candidates = list(artifact_dir.glob("*_cross_version_metadata.json"))
    else:
        candidates = [
            f for f in artifact_dir.glob("*_metadata.json") if "chunk" not in f.name
        ]
        if not candidates:
            candidates = list(artifact_dir.glob("*_metadata_chunk*.json"))

    if not candidates:
        log.warning(f"  No metadata JSON found in {artifact_dir}")
        return None

    with open(candidates[0]) as f:
        return json.load(f)


def _get_version_pairs(metadata: dict, strategy: str) -> list[tuple[str, str]]:
    """Extract consecutive version pairs from metadata.

    Returns:
        List of (version_a, version_b) tuples
    """
    if strategy == "cumulative-carried":
        return [
            (r["version_a"], r["version_b"])
            for r in metadata.get("consecutive_pair_results", [])
        ]
    elif strategy == "pairwise":
        return [(r["version_a"], r["version_b"]) for r in metadata.get("results", [])]
    else:  # cumulative-fresh
        pairs = []
        for r in metadata.get("results", []):
            train_versions = r["train_versions"]
            if train_versions:
                pairs.append((train_versions[-1], r["test_version"]))
        return pairs


def _compute_pair_similarities(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    dim_cols: list[str],
) -> pd.DataFrame:
    """Compute cosine similarity for matched files between two version sets.

    Args:
        df_a: Embeddings for version A with 'relative_path' column
        df_b: Embeddings for version B with 'relative_path' column
        dim_cols: List of dimension column names

    Returns:
        DataFrame with columns: relative_path, similarity
    """
    merged = pd.merge(
        df_a[["relative_path"] + dim_cols],
        df_b[["relative_path"] + dim_cols],
        on="relative_path",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return pd.DataFrame(columns=["relative_path", "similarity"])

    vec_a = merged[[f"{c}_a" for c in dim_cols]].values
    vec_b = merged[[f"{c}_b" for c in dim_cols]].values

    # Row-wise cosine similarity (diagonal of the full matrix)
    # Clamp to [-1, 1] to handle floating-point imprecision
    norms_a = np.linalg.norm(vec_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(vec_b, axis=1, keepdims=True)
    # Avoid division by zero for zero-length vectors
    norms_a = np.maximum(norms_a, 1e-10)
    norms_b = np.maximum(norms_b, 1e-10)
    sims = np.sum((vec_a / norms_a) * (vec_b / norms_b), axis=1)
    sims = np.clip(sims, -1.0, 1.0)

    return pd.DataFrame(
        {"relative_path": merged["relative_path"].values, "similarity": sims}
    )


def _pair_stats(sims: np.ndarray) -> dict:
    """Compute summary statistics for a set of similarity values."""
    return {
        "matched_files": len(sims),
        "mean_sim": round(float(np.mean(sims)), 4),
        "median_sim": round(float(np.median(sims)), 4),
        "min_sim": round(float(np.min(sims)), 4),
        "max_sim": round(float(np.max(sims)), 4),
        "pct_above_099": round(float(np.mean(sims >= 0.99) * 100), 2),
        "pct_above_095": round(float(np.mean(sims >= 0.95) * 100), 2),
        "pct_above_090": round(float(np.mean(sims >= 0.90) * 100), 2),
    }


def _extract_version_prefix(file_path: str) -> str:
    """Extract the version prefix directory from a file path.

    Examples:
        '1.0/django/__init__.py' -> '1.0'
        'calcite-1.0.0-incubating/spark/src/...' -> 'calcite-1.0.0-incubating'
    """
    return file_path.split("/", 1)[0]


def _load_embeddings_with_version_col(
    csv_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Load an embeddings CSV that has file_path, version, label, dim_* columns.

    Uses the file_path prefix as the version string (not the version column)
    because the version column is float and loses precision for versions
    like 1.10 (stored as 1.1).

    Returns:
        (DataFrame with added 'relative_path' and corrected 'version' columns, dim_cols)
    """
    df = pd.read_csv(csv_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    df["version"] = df["file_path"].apply(_extract_version_prefix)
    df["relative_path"] = df["file_path"].apply(_strip_version_prefix)
    return df, dim_cols


def _load_embeddings_no_version_col(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load an embeddings CSV that has only file_path and dim_* columns.

    Returns:
        (DataFrame with added 'relative_path' column, dim_cols)
    """
    df = pd.read_csv(csv_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    df["relative_path"] = df["file_path"].apply(_strip_version_prefix)
    return df, dim_cols


def analyze_pairwise(
    artifact_dir: Path,
    metadata: dict,
    version_pairs: list[tuple[str, str]],
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Analyze pairwise strategy embeddings.

    Returns:
        (list of per-pair stat dicts, list of per-pair similarity DataFrames)
    """
    stats = []
    all_sims = []

    for i, (ver_a, ver_b) in enumerate(version_pairs):
        pair_num = i + 1
        candidates = list(artifact_dir.glob(f"*_pair{pair_num}_embeddings.csv"))
        if not candidates:
            log.warning(f"    Pair {pair_num}: embeddings CSV not found")
            continue

        df, dim_cols = _load_embeddings_with_version_col(candidates[0])
        df_a = df[df["version"] == str(ver_a)]
        df_b = df[df["version"] == str(ver_b)]

        sim_df = _compute_pair_similarities(df_a, df_b, dim_cols)
        if sim_df.empty:
            log.warning(f"    Pair {pair_num}: no matched files")
            continue

        pair_stat = {"pair": pair_num, "train_ver": ver_a, "test_ver": ver_b}
        pair_stat.update(_pair_stats(sim_df["similarity"].values))
        stats.append(pair_stat)

        sim_df["pair"] = pair_num
        sim_df["version_a"] = ver_a
        sim_df["version_b"] = ver_b
        all_sims.append(sim_df)

        log.info(
            f"    Pair {pair_num} ({ver_a} -> {ver_b}): "
            f"{pair_stat['matched_files']} files, "
            f"mean={pair_stat['mean_sim']:.4f}, "
            f">0.99={pair_stat['pct_above_099']:.1f}%"
        )

    return stats, all_sims


def analyze_cumfresh(
    artifact_dir: Path,
    metadata: dict,
    version_pairs: list[tuple[str, str]],
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Analyze cumulative-fresh strategy embeddings.

    Each iter CSV contains all versions for that iteration. We compare the
    last train version vs the test version (same consecutive pair as pairwise).

    Returns:
        (list of per-pair stat dicts, list of per-pair similarity DataFrames)
    """
    stats = []
    all_sims = []

    for i, (ver_a, ver_b) in enumerate(version_pairs):
        iter_num = i + 1
        candidates = list(artifact_dir.glob(f"*_iter{iter_num}_embeddings.csv"))
        if not candidates:
            log.warning(f"    Iter {iter_num}: embeddings CSV not found")
            continue

        df, dim_cols = _load_embeddings_with_version_col(candidates[0])
        df_a = df[df["version"] == str(ver_a)]
        df_b = df[df["version"] == str(ver_b)]

        sim_df = _compute_pair_similarities(df_a, df_b, dim_cols)
        if sim_df.empty:
            log.warning(f"    Iter {iter_num}: no matched files")
            continue

        pair_stat = {"pair": iter_num, "train_ver": ver_a, "test_ver": ver_b}
        pair_stat.update(_pair_stats(sim_df["similarity"].values))
        stats.append(pair_stat)

        sim_df["pair"] = iter_num
        sim_df["version_a"] = ver_a
        sim_df["version_b"] = ver_b
        all_sims.append(sim_df)

        log.info(
            f"    Iter {iter_num} ({ver_a} -> {ver_b}): "
            f"{pair_stat['matched_files']} files, "
            f"mean={pair_stat['mean_sim']:.4f}, "
            f">0.99={pair_stat['pct_above_099']:.1f}%"
        )

    return stats, all_sims


def analyze_carried(
    artifact_dir: Path,
    metadata: dict,
    version_pairs: list[tuple[str, str]],
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Analyze cumulative-carried strategy embeddings.

    Per-version CSVs: *_{version}_embeddings.csv (no version column).

    Returns:
        (list of per-pair stat dicts, list of per-pair similarity DataFrames)
    """
    stats = []
    all_sims = []

    for i, (ver_a, ver_b) in enumerate(version_pairs):
        pair_num = i + 1
        candidates_a = list(artifact_dir.glob(f"*_{ver_a}_embeddings.csv"))
        candidates_b = list(artifact_dir.glob(f"*_{ver_b}_embeddings.csv"))
        if not candidates_a or not candidates_b:
            log.warning(
                f"    Pair {pair_num}: embeddings CSV not found for {ver_a} or {ver_b}"
            )
            continue

        df_a, dim_cols = _load_embeddings_no_version_col(candidates_a[0])
        df_b, _ = _load_embeddings_no_version_col(candidates_b[0])

        sim_df = _compute_pair_similarities(df_a, df_b, dim_cols)
        if sim_df.empty:
            log.warning(f"    Pair {pair_num}: no matched files")
            continue

        pair_stat = {"pair": pair_num, "train_ver": ver_a, "test_ver": ver_b}
        pair_stat.update(_pair_stats(sim_df["similarity"].values))
        stats.append(pair_stat)

        sim_df["pair"] = pair_num
        sim_df["version_a"] = ver_a
        sim_df["version_b"] = ver_b
        all_sims.append(sim_df)

        log.info(
            f"    Pair {pair_num} ({ver_a} -> {ver_b}): "
            f"{pair_stat['matched_files']} files, "
            f"mean={pair_stat['mean_sim']:.4f}, "
            f">0.99={pair_stat['pct_above_099']:.1f}%"
        )

    return stats, all_sims


ANALYZERS = {
    "pairwise": analyze_pairwise,
    "cumulative-fresh": analyze_cumfresh,
    "cumulative-carried": analyze_carried,
}


_CSV_HEADER = (
    "# Cosine similarity between Doc2Vec embeddings of same-path files across consecutive versions.\n"
    "# Each row = one file that exists in both versions, with the cosine similarity of its embeddings.\n"
    "# No threshold cutoff — all matched files are included regardless of similarity.\n"
)


def _write_csv_with_header(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV with a comment header explaining the methodology."""
    with open(path, "w", newline="") as f:
        f.write(_CSV_HEADER)
        df.to_csv(f, index=False)


def run_analysis(
    results_dir: Path,
    strategies: list[str],
    projects: list[str],
) -> dict[str, dict[str, list[dict]]]:
    """Run similarity analysis for given strategies and projects.

    Returns:
        {project: {strategy: [pair_stat_dicts]}}
    """
    all_stats: dict[str, dict[str, list[dict]]] = {}

    for strategy in strategies:
        analyzer = ANALYZERS[strategy]

        for project in projects:
            artifact_name = ARTIFACT_CONFIGS.get(strategy, {}).get(project)
            if not artifact_name:
                continue

            artifact_dir = results_dir / artifact_name
            if not artifact_dir.exists():
                log.warning(f"  {artifact_name}: directory not found, skipping")
                continue

            metadata = _find_metadata(artifact_dir, strategy)
            if metadata is None:
                continue

            version_pairs = _get_version_pairs(metadata, strategy)
            if not version_pairs:
                log.warning(f"  {artifact_name}: no version pairs found")
                continue

            log.info(f"\n  {project} / {strategy} ({len(version_pairs)} pairs)")
            pair_stats, sim_dfs = analyzer(artifact_dir, metadata, version_pairs)

            all_stats.setdefault(project, {})[strategy] = pair_stats

            # Write per-file similarities CSV
            if sim_dfs:
                combined = pd.concat(sim_dfs, ignore_index=True)
                combined = combined[
                    ["pair", "relative_path", "version_a", "version_b", "similarity"]
                ]
                csv_path = (
                    results_dir
                    / f"embedding_distances_{project}_{strategy.replace('-', '_')}.csv"
                )
                _write_csv_with_header(combined, csv_path)
                log.info(f"    -> {csv_path} ({len(combined)} rows)")

    return all_stats


def print_pair_tables(all_stats: dict[str, dict[str, list[dict]]]) -> None:
    """Print per-pair comparison tables for each project."""
    for project in sorted(all_stats):
        log.info(f"\n{'=' * 80}")
        log.info(
            f"  {project.upper()} — Per-pair embedding similarity (same-path files)"
        )
        log.info(f"{'=' * 80}")

        for strategy in sorted(all_stats[project]):
            pair_stats = all_stats[project][strategy]
            if not pair_stats:
                continue

            log.info(f"\n  Strategy: {strategy}")
            df = pd.DataFrame(pair_stats)
            log.info(df.to_string(index=False))


def build_summary_table(
    all_stats: dict[str, dict[str, list[dict]]],
) -> pd.DataFrame:
    """Build aggregate summary across projects and strategies."""
    rows = []

    for project in sorted(all_stats):
        for strategy in sorted(all_stats[project]):
            pair_stats = all_stats[project][strategy]
            if not pair_stats:
                continue

            matched_total = sum(p["matched_files"] for p in pair_stats)
            mean_sims = [p["mean_sim"] for p in pair_stats]
            pct_099 = [p["pct_above_099"] for p in pair_stats]
            pct_095 = [p["pct_above_095"] for p in pair_stats]

            rows.append(
                {
                    "project": project,
                    "strategy": strategy,
                    "num_pairs": len(pair_stats),
                    "total_matched": matched_total,
                    "avg_mean_sim": round(sum(mean_sims) / len(mean_sims), 4),
                    "avg_pct_above_099": round(sum(pct_099) / len(pct_099), 2),
                    "avg_pct_above_095": round(sum(pct_095) / len(pct_095), 2),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute embedding similarity distributions for same-file pairs across versions."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing downloaded artifacts (default: results/)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="all",
        choices=["django", "calcite", "all"],
        help="Project to analyze (default: all)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["pairwise", "cumulative-fresh", "cumulative-carried", "all"],
        help="Strategy to analyze (default: all)",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    projects = ["django", "calcite"] if args.project == "all" else [args.project]
    strategies = (
        ["pairwise", "cumulative-fresh", "cumulative-carried"]
        if args.strategy == "all"
        else [args.strategy]
    )

    log.info("Computing embedding similarity distributions...")
    all_stats = run_analysis(results_dir, strategies, projects)

    if not all_stats:
        log.error("No data found. Ensure artifacts are downloaded in --results-dir.")
        sys.exit(1)

    # Per-pair tables
    print_pair_tables(all_stats)

    # Aggregate summary
    log.info(f"\n{'=' * 80}")
    log.info("  AGGREGATE SUMMARY")
    log.info(f"{'=' * 80}")

    summary_df = build_summary_table(all_stats)
    if not summary_df.empty:
        log.info(summary_df.to_string(index=False))

        csv_path = results_dir / "embedding_distance_summary.csv"
        _write_csv_with_header(summary_df, csv_path)
        log.info(f"\n  Saved -> {csv_path}")


if __name__ == "__main__":
    main()
