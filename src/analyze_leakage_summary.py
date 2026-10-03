#!/usr/bin/env python3
"""
analyze_leakage_summary.py
==========================
Download and compare embedding-based leakage across strategies and projects.

Compares three strategies:
  - Pairwise: fresh base model per consecutive pair
  - Cumulative-fresh: fresh base model, growing version window
  - Cumulative-carried: model carried forward across versions

Produces per-pair comparison tables and aggregate summaries.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger(__name__)

# Artifact name -> (run_id, strategy, project)
ARTIFACTS = {
    "pairwise-django": (36435916600, "pairwise", "django"),
    "pairwise-calcite": (36436651515, "pairwise", "calcite"),
    "cumulative-fresh-django": (36702999931, "cumulative-fresh", "django"),
    "cumulative-fresh-calcite": (36529018897, "cumulative-fresh", "calcite"),
    "cross-version-django": (33485534592, "cumulative-carried", "django"),
    "cross-version-calcite": (33080537845, "cumulative-carried", "calcite"),
}

REPO = "ioanachelaru/doc2vec-experiments"


def download_artifacts(results_dir: Path) -> None:
    """Download all artifacts from GitHub Actions."""
    results_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name, (run_id, strategy, project) in ARTIFACTS.items():
        dest = results_dir / artifact_name
        if dest.exists() and any(dest.iterdir()):
            log.info(f"  {artifact_name}: already downloaded, skipping")
            continue

        log.info(f"  Downloading {artifact_name} (run {run_id})...")
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "-R",
                REPO,
                "-n",
                artifact_name,
                "-D",
                str(dest),
            ],
            check=True,
        )
    log.info("All artifacts downloaded.")


def find_metadata(artifact_dir: Path, strategy: str) -> dict | None:
    """Find and load the metadata JSON from an artifact directory.

    Args:
        artifact_dir: Path to the downloaded artifact directory
        strategy: Strategy name to help disambiguate metadata files
    """
    if strategy == "cumulative-carried":
        # cross_version_pipeline.py outputs *_cross_version_metadata.json
        # alongside per-pair *_vA_vs_vB_metadata.json files — need the main one
        candidates = list(artifact_dir.glob("*_cross_version_metadata.json"))
    else:
        # Pairwise/cumulative-fresh: prefer non-chunk, then chunk metadata
        candidates = [
            f for f in artifact_dir.glob("*_metadata.json") if "chunk" not in f.name
        ]
        if not candidates:
            candidates = list(artifact_dir.glob("*_metadata_chunk*.json"))

    if not candidates:
        log.warning(f"No metadata JSON found in {artifact_dir}")
        return None

    with open(candidates[0]) as f:
        return json.load(f)


def extract_pairs_pairwise(metadata: dict) -> list[dict]:
    """Extract per-pair leakage from pairwise metadata."""
    pairs = []
    for r in metadata.get("results", []):
        pairs.append(
            {
                "version_a": r["version_a"],
                "version_b": r["version_b"],
                "test_size": r.get("files_b", 0),
                "leakage_pct": r["test_leakage_pct"],
                "same_file_pairs": r["same_file_pairs"],
                "collision_pairs": r["collision_pairs"],
            }
        )
    return pairs


def extract_pairs_cumfresh(metadata: dict) -> list[dict]:
    """Extract per-pair leakage from cumulative-fresh metadata."""
    pairs = []
    for r in metadata.get("results", []):
        train_versions = r["train_versions"]
        pairs.append(
            {
                "version_a": train_versions[-1] if train_versions else "",
                "version_b": r["test_version"],
                "test_size": r.get("test_size", 0),
                "leakage_pct": r["test_leakage_pct"],
                "same_file_pairs": r["same_file_pairs"],
                "collision_pairs": r["collision_pairs"],
            }
        )
    return pairs


def extract_pairs_carried(metadata: dict) -> list[dict]:
    """Extract per-pair leakage from cumulative-carried metadata."""
    pairs = []
    for r in metadata.get("consecutive_pair_results", []):
        pairs.append(
            {
                "version_a": r["version_a"],
                "version_b": r["version_b"],
                "test_size": r.get("test_set_size", 0),
                "leakage_pct": r.get("test_leakage_percentage", 0),
                "same_file_pairs": r.get("same_file_leakage_pairs", 0),
                "collision_pairs": r.get("collision_leakage_pairs", 0),
            }
        )
    return pairs


EXTRACTORS = {
    "pairwise": extract_pairs_pairwise,
    "cumulative-fresh": extract_pairs_cumfresh,
    "cumulative-carried": extract_pairs_carried,
}


def load_all_pairs(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Load pairs from all artifacts, grouped by project and strategy.

    Returns:
        {project: {strategy: [pair_dicts]}}
    """
    data: dict[str, dict[str, list[dict]]] = {}

    for artifact_name, (_, strategy, project) in ARTIFACTS.items():
        artifact_dir = results_dir / artifact_name
        if not artifact_dir.exists():
            log.warning(f"  {artifact_name}: not found, skipping")
            continue

        metadata = find_metadata(artifact_dir, strategy)
        if metadata is None:
            continue

        extractor = EXTRACTORS[strategy]
        pairs = extractor(metadata)
        log.info(f"  {artifact_name}: {len(pairs)} pairs loaded")

        data.setdefault(project, {})[strategy] = pairs

    return data


def build_comparison_table(
    project_data: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build a per-pair comparison table across strategies.

    Args:
        project_data: {strategy: [pair_dicts]} for one project

    Returns:
        DataFrame with columns per strategy
    """
    # Use pairwise as the reference for version pairs (it always has them)
    # Fall back to cumulative-fresh or cumulative-carried
    ref_strategy = None
    for s in ["pairwise", "cumulative-fresh", "cumulative-carried"]:
        if s in project_data:
            ref_strategy = s
            break

    if ref_strategy is None:
        return pd.DataFrame()

    ref_pairs = project_data[ref_strategy]
    rows = []

    for i, ref in enumerate(ref_pairs):
        row = {
            "pair": i + 1,
            "train_version": ref["version_a"],
            "test_version": ref["version_b"],
            "test_size": ref["test_size"],
        }

        for strategy, pairs in project_data.items():
            if i < len(pairs):
                p = pairs[i]
                prefix = strategy.replace("-", "_")
                row[f"{prefix}_leakage_pct"] = p["leakage_pct"]
                row[f"{prefix}_same_file"] = p["same_file_pairs"]
                row[f"{prefix}_collision"] = p["collision_pairs"]

        rows.append(row)

    return pd.DataFrame(rows)


def build_summary_table(
    all_data: dict[str, dict[str, list[dict]]],
) -> pd.DataFrame:
    """Build aggregate summary across projects and strategies."""
    rows = []

    for project, strategies in sorted(all_data.items()):
        for strategy, pairs in sorted(strategies.items()):
            if not pairs:
                continue
            leakage_pcts = [p["leakage_pct"] for p in pairs]
            same_total = sum(p["same_file_pairs"] for p in pairs)
            coll_total = sum(p["collision_pairs"] for p in pairs)

            rows.append(
                {
                    "project": project,
                    "strategy": strategy,
                    "num_pairs": len(pairs),
                    "avg_leakage_pct": round(sum(leakage_pcts) / len(leakage_pcts), 2),
                    "median_leakage_pct": round(
                        sorted(leakage_pcts)[len(leakage_pcts) // 2], 2
                    ),
                    "max_leakage_pct": max(leakage_pcts),
                    "min_leakage_pct": min(leakage_pcts),
                    "total_same_file_pairs": same_total,
                    "total_collision_pairs": coll_total,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding-based leakage across strategies and projects."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for artifacts (default: results/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download artifacts from GitHub Actions before analysis",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if args.download:
        log.info("Downloading artifacts...")
        download_artifacts(results_dir)

    log.info("\nLoading metadata from artifacts...")
    all_data = load_all_pairs(results_dir)

    if not all_data:
        log.error("No data found. Run with --download to fetch artifacts.")
        sys.exit(1)

    # Per-project comparison tables
    for project, strategies in sorted(all_data.items()):
        log.info(f"\n{'=' * 70}")
        log.info(f"  {project.upper()} — Per-pair leakage comparison")
        log.info(f"{'=' * 70}")

        df = build_comparison_table(strategies)
        if df.empty:
            log.info("  No data available")
            continue

        log.info(df.to_string(index=False))

        csv_path = results_dir / f"leakage_comparison_{project}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"\n  Saved -> {csv_path}")

    # Aggregate summary
    log.info(f"\n{'=' * 70}")
    log.info("  AGGREGATE SUMMARY")
    log.info(f"{'=' * 70}")

    summary_df = build_summary_table(all_data)
    if not summary_df.empty:
        log.info(summary_df.to_string(index=False))

        csv_path = results_dir / "leakage_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        log.info(f"\n  Saved -> {csv_path}")


if __name__ == "__main__":
    main()
