#!/usr/bin/env python3
"""
enrich_leakage.py
=================
Join cross-version leakage pairs with bug labels to determine whether
near-duplicate files share the same defect status.

Reads the pipeline's leakage CSVs and a directory of per-version label
files, producing enriched CSVs and a summary breakdown by label.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_labels(labels_dir: Path, version: str) -> dict[str, str]:
    """Load bug labels for a single version.

    Args:
        labels_dir: Directory containing django-{version}.csv files
        version: Version string (e.g., '1.0')

    Returns:
        Dict mapping filepath -> label ('buggy' or 'clean')
    """
    label_file = labels_dir / f"django-{version}.csv"
    if not label_file.exists():
        return {}
    df = pd.read_csv(label_file)
    return dict(zip(df["filepath"], df["label"]))


def extract_version_and_path(tagged_path: str) -> tuple[str, str]:
    """Split a tagged path like '1.0/django/foo.py' into ('1.0', 'django/foo.py')."""
    parts = tagged_path.split("/", 1)
    return parts[0], parts[1]


def enrich_pair(
    pair_idx: int,
    version_a: str,
    version_b: str,
    leakage_csv: Path,
    labels_dir: Path,
    output_prefix: str,
) -> dict | None:
    """Enrich one pair's leakage CSV with bug labels.

    Args:
        pair_idx: 1-based pair index
        version_a: Training boundary version
        version_b: Test version
        leakage_csv: Path to the pair's leakage CSV
        labels_dir: Directory containing label files
        output_prefix: Prefix for output files

    Returns:
        Summary dict for this pair, or None if leakage CSV doesn't exist
    """
    if not leakage_csv.exists():
        return None

    df = pd.read_csv(leakage_csv)
    if df.empty:
        return None

    # Load labels for all training versions up to version_a and the test version
    # file_a comes from any training version, file_b comes from version_b
    label_cache: dict[str, dict[str, str]] = {}

    def get_label(tagged_path: str) -> str:
        ver, fpath = extract_version_and_path(tagged_path)
        if ver not in label_cache:
            label_cache[ver] = load_labels(labels_dir, ver)
        return label_cache[ver].get(fpath, "unknown")

    df["label_a"] = df["file_a"].apply(get_label)
    df["label_b"] = df["file_b"].apply(get_label)
    df["same_label"] = df["label_a"] == df["label_b"]

    # Save enriched CSV
    out_path = f"{output_prefix}_pair{pair_idx}_leakage_labeled.csv"
    df.to_csv(out_path, index=False)

    # Load test version labels for full stats (not just leaked files)
    test_labels = load_labels(labels_dir, version_b)
    test_total = len(test_labels)
    test_buggy = sum(1 for v in test_labels.values() if v == "buggy")
    test_clean = test_total - test_buggy

    # Leaked test files (unique file_b entries) and their labels
    leaked_files = df.drop_duplicates(subset="file_b")
    leaked_labels = leaked_files["label_b"]

    leaked_buggy = int((leaked_labels == "buggy").sum())
    leaked_clean = int((leaked_labels == "clean").sum())
    leaked_total = leaked_buggy + leaked_clean  # exclude 'unknown'

    # Same-label vs different-label among leaked pairs
    known_pairs = df[(df["label_a"] != "unknown") & (df["label_b"] != "unknown")]
    same_label = int(known_pairs["same_label"].sum())
    diff_label = len(known_pairs) - same_label

    buggy_leak_pct = round(leaked_buggy / test_buggy * 100, 2) if test_buggy > 0 else 0
    clean_leak_pct = round(leaked_clean / test_clean * 100, 2) if test_clean > 0 else 0

    summary = {
        "pair": pair_idx,
        "version_a": version_a,
        "version_b": version_b,
        "test_total": test_total,
        "test_buggy": test_buggy,
        "test_clean": test_clean,
        "leaked_total": leaked_total,
        "leaked_buggy": leaked_buggy,
        "leaked_clean": leaked_clean,
        "leaked_same_label_pairs": same_label,
        "leaked_diff_label_pairs": diff_label,
        "buggy_leak_pct": buggy_leak_pct,
        "clean_leak_pct": clean_leak_pct,
    }

    print(
        f"  Pair {pair_idx} ({version_a} vs {version_b}): "
        f"test={test_total} ({test_buggy}B/{test_clean}C), "
        f"leaked={leaked_total} ({leaked_buggy}B/{leaked_clean}C), "
        f"buggy_leak={buggy_leak_pct}%, clean_leak={clean_leak_pct}%"
    )

    return summary


def run(metadata_path: str, labels_dir: str, output_prefix: str) -> None:
    """Run the label enrichment pipeline.

    Args:
        metadata_path: Path to *_cross_version_metadata.json
        labels_dir: Directory containing django-{version}.csv label files
        output_prefix: Prefix for output files
    """
    meta = json.loads(Path(metadata_path).read_text())
    labels_path = Path(labels_dir)
    pairs = meta["consecutive_pair_results"]

    if not labels_path.is_dir():
        print(f"Error: labels directory not found: {labels_path}")
        sys.exit(1)

    print(f"Enriching {len(pairs)} pairs with labels from {labels_path}")
    print(f"{'=' * 70}")

    summaries = []
    skipped = []

    for idx, pair in enumerate(pairs, 1):
        va, vb = pair["version_a"], pair["version_b"]

        # Check if label files exist for this pair's test version
        if not (labels_path / f"django-{vb}.csv").exists():
            skipped.append(f"{va} vs {vb} (no labels for {vb})")
            continue

        leakage_csv = Path(f"{output_prefix}_pair{idx}_leakage.csv")
        result = enrich_pair(idx, va, vb, leakage_csv, labels_path, output_prefix)
        if result:
            summaries.append(result)

    if skipped:
        print(f"\nSkipped {len(skipped)} pairs (no label data):")
        for s in skipped:
            print(f"  - {s}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = f"{output_prefix}_leakage_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        # Print overall stats
        total_leaked_buggy = summary_df["leaked_buggy"].sum()
        total_leaked_clean = summary_df["leaked_clean"].sum()
        total_same = summary_df["leaked_same_label_pairs"].sum()
        total_diff = summary_df["leaked_diff_label_pairs"].sum()
        print(f"\nOverall across {len(summaries)} pairs:")
        print(f"  Leaked buggy test files: {total_leaked_buggy}")
        print(f"  Leaked clean test files: {total_leaked_clean}")
        print(f"  Same-label leakage pairs: {total_same}")
        print(f"  Diff-label leakage pairs: {total_diff}")
    else:
        print("\nNo pairs had both leakage data and labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich cross-version leakage pairs with bug labels."
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to *_cross_version_metadata.json",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing django-{version}.csv label files",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files (same as pipeline output prefix)",
    )

    args = parser.parse_args()
    run(args.metadata, args.labels_dir, args.output_prefix)
