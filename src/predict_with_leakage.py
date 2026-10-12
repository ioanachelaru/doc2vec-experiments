#!/usr/bin/env python3
"""
predict_with_leakage.py
=======================
Measure how train/test overlap affects defect prediction in CVDP.

For each version pair, trains a RandomForest on embedding vectors and evaluates
on three subsets:
  - baseline: full test set
  - new_files: test files not present in training (by filepath)
  - changed_label: test files present in training but with a different label

Usage:
    python src/predict_with_leakage.py --project django --strategy pairwise
    python src/predict_with_leakage.py --project calcite --strategy cumulative-fresh
    python src/predict_with_leakage.py --project django --strategy pairwise --dim-suffix 200
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings(csv_path: Path) -> pd.DataFrame:
    """Load an embeddings CSV and return DataFrame with file_path, version, label, and dim columns."""
    df = pd.read_csv(csv_path)
    df = df[df["label"] != "unknown"].copy()
    return df


def get_dim_columns(df: pd.DataFrame) -> list[str]:
    """Get the embedding dimension column names."""
    return [c for c in df.columns if c.startswith("dim_")]


def extract_relative_path(file_path: str) -> str:
    """Strip version prefix from file_path to get relative path.

    '1.0/django/utils/foo.py' -> 'django/utils/foo.py'
    'calcite-1.0.0-incubating/core/src/...' -> 'core/src/...'
    """
    parts = file_path.split("/", 1)
    return parts[1] if len(parts) > 1 else file_path


def _version_matches(series: pd.Series, target: str) -> pd.Series:
    """Match a version column against a target string, handling float precision.

    Django versions like '1.10' become float 1.1 in CSV, so str(1.1) != '1.10'.
    Try string match first, then float match.
    """
    str_match = series.astype(str) == target
    if str_match.any():
        return str_match
    try:
        return series == float(target)
    except (ValueError, TypeError):
        return str_match


def split_train_test(
    df: pd.DataFrame, metadata_pair: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split embeddings into train and test based on metadata version info."""
    version_a = metadata_pair["version_a"]
    version_b = metadata_pair["version_b"]

    train = df[_version_matches(df["version"], str(version_a))].copy()
    test = df[_version_matches(df["version"], str(version_b))].copy()

    return train, test


def split_cumfresh_train_test(
    df: pd.DataFrame, pair_meta: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cumulative-fresh embeddings: all versions except last = train, last = test."""
    test_version = str(pair_meta.get("version_b", pair_meta.get("test_version", "")))
    test_mask = _version_matches(df["version"], test_version)
    test = df[test_mask].copy()
    train = df[~test_mask].copy()
    return train, test


def compute_new_files(train: pd.DataFrame, test: pd.DataFrame) -> set[str]:
    """Find test files whose relative path does not appear in training."""
    train_paths = {extract_relative_path(fp) for fp in train["file_path"]}
    new = set()
    for _, row in test.iterrows():
        rel = extract_relative_path(row["file_path"])
        if rel not in train_paths:
            new.add(row["file_path"])
    return new


def compute_changed_label(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> set[str]:
    """Find test files present in train but with a different label.

    A test file is changed_label if:
    - Its label differs from ANY train instance with the same path, OR
    - The file has inconsistent labels across train versions (both clean and buggy).

    This means files with mixed labels in training are always included,
    regardless of the test label.
    """
    # Build map: relative_path -> set of all labels seen in training
    train_labels_by_path: dict[str, set[str]] = {}
    for _, row in train.iterrows():
        rel = extract_relative_path(row["file_path"])
        train_labels_by_path.setdefault(rel, set()).add(row["label"])

    changed = set()
    for _, row in test.iterrows():
        rel = extract_relative_path(row["file_path"])
        if rel in train_labels_by_path:
            train_labels = train_labels_by_path[rel]
            # Include if: test label differs from any train label, or train has mixed labels
            if row["label"] not in train_labels or len(train_labels) > 1:
                changed.add(row["file_path"])
    return changed


def evaluate_subset(clf, X: np.ndarray, y: np.ndarray, subset_name: str) -> dict | None:
    """Evaluate classifier on a subset, return metrics dict or None if subset is empty."""
    if len(X) == 0 or len(np.unique(y)) == 0:
        return None

    y_pred = clf.predict(X)
    buggy_count = int(np.sum(y == 1))
    clean_count = int(np.sum(y == 0))
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))

    metrics = {
        "classifier": "RandomForest",
        "subset": subset_name,
        "total": len(y),
        "support_buggy": buggy_count,
        "support_clean": clean_count,
    }

    metrics["accuracy"] = accuracy_score(y, y_pred)
    metrics["f1_macro"] = f1_score(y, y_pred, average="macro", zero_division=0)
    metrics["f1_weighted"] = f1_score(y, y_pred, average="weighted", zero_division=0)

    if len(np.unique(y)) < 2:
        nan_metrics = [
            "precision_buggy",
            "recall_buggy",
            "f1_buggy",
            "precision_clean",
            "recall_clean",
            "f1_clean",
            "far",
            "csi",
            "mcc",
            "auc",
            "auprc",
        ]
        for m in nan_metrics:
            metrics[m] = np.nan
    else:
        # Per-class precision, recall, F1
        metrics["precision_buggy"] = precision_score(
            y, y_pred, pos_label=1, zero_division=0
        )
        metrics["recall_buggy"] = recall_score(y, y_pred, pos_label=1, zero_division=0)
        metrics["f1_buggy"] = f1_score(y, y_pred, pos_label=1, zero_division=0)
        metrics["precision_clean"] = precision_score(
            y, y_pred, pos_label=0, zero_division=0
        )
        metrics["recall_clean"] = recall_score(y, y_pred, pos_label=0, zero_division=0)
        metrics["f1_clean"] = f1_score(y, y_pred, pos_label=0, zero_division=0)

        # Derived metrics
        metrics["far"] = 1.0 - metrics["recall_clean"]  # False Alarm Rate
        csi_denom = tp + fp + fn
        metrics["csi"] = (
            tp / csi_denom if csi_denom > 0 else np.nan
        )  # Critical Success Index

        metrics["mcc"] = matthews_corrcoef(y, y_pred)
        try:
            y_proba = clf.predict_proba(X)[:, 1]
            metrics["auc"] = roc_auc_score(y, y_proba)
            metrics["auprc"] = average_precision_score(y, y_proba)
        except (ValueError, IndexError):
            metrics["auc"] = np.nan
            metrics["auprc"] = np.nan

    return metrics


def compute_leaked_train_mask(
    X_train: np.ndarray, X_test: np.ndarray, threshold: float = 0.99
) -> np.ndarray:
    """Return boolean mask of train instances with cosine similarity >= threshold to any test instance."""
    sim_matrix = np.clip(cosine_similarity(X_train, X_test), -1.0, 1.0)
    return sim_matrix.max(axis=1) >= threshold


def run_pair(
    pair: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dim_cols: list[str],
    new_file_paths: set[str],
    changed_label_paths: set[str],
    n_seeds: int = 10,
) -> list[dict]:
    """Run classification for one pair with multiple seeds and cleaned-train variant."""
    X_train_full = train[dim_cols].values.astype(np.float64)
    y_train_full = (train["label"] == "buggy").astype(int).values

    if len(np.unique(y_train_full)) < 2:
        print(f"  Pair {pair}: skipping — single class in train")
        return []

    # Build test subsets
    new_mask = test["file_path"].isin(new_file_paths)
    changed_mask = test["file_path"].isin(changed_label_paths)

    subsets = {
        "baseline": test,
        "new_files": test[new_mask],
        "changed_label": test[changed_mask],
    }

    # Compute cleaned train set (remove instances with cosine >= 0.99 to any test)
    X_test_all = test[dim_cols].values.astype(np.float64)
    leaked_mask = compute_leaked_train_mask(X_train_full, X_test_all)
    n_leaked = int(leaked_mask.sum())

    X_train_clean = X_train_full[~leaked_mask]
    y_train_clean = y_train_full[~leaked_mask]
    clean_viable = len(X_train_clean) > 0 and len(np.unique(y_train_clean)) >= 2

    print(
        f"    leaked_train={n_leaked} ({n_leaked / len(X_train_full) * 100:.1f}%), "
        f"cleaned_train={len(X_train_clean)}"
    )

    seeds = list(range(42, 42 + n_seeds))
    results = []

    for seed in seeds:
        # Full train
        clf_full = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
        )
        clf_full.fit(X_train_full, y_train_full)

        for subset_name, subset_df in subsets.items():
            X_sub = subset_df[dim_cols].values.astype(np.float64)
            y_sub = (subset_df["label"] == "buggy").astype(int).values
            m = evaluate_subset(clf_full, X_sub, y_sub, subset_name)
            if m is not None:
                m["pair"] = pair
                m["seed"] = seed
                m["train_set"] = "full"
                m["n_leaked_train"] = n_leaked
                results.append(m)

        # Cleaned train
        if clean_viable:
            clf_clean = RandomForestClassifier(
                n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
            )
            clf_clean.fit(X_train_clean, y_train_clean)

            for subset_name, subset_df in subsets.items():
                X_sub = subset_df[dim_cols].values.astype(np.float64)
                y_sub = (subset_df["label"] == "buggy").astype(int).values
                m = evaluate_subset(clf_clean, X_sub, y_sub, subset_name)
                if m is not None:
                    m["pair"] = pair
                    m["seed"] = seed
                    m["train_set"] = "cleaned"
                    m["n_leaked_train"] = n_leaked
                    results.append(m)

    return results


def find_embedding_files(
    results_dir: Path, project: str, strategy: str, dim_suffix: str = ""
) -> list[tuple[int, Path]]:
    """Find all embedding CSV files for a project/strategy, return sorted (pair, path) list."""
    suffix = f"-{dim_suffix}" if dim_suffix else ""
    if strategy == "pairwise":
        pattern = f"{project}_pairwise_pair*_embeddings.csv"
        subdir = results_dir / f"pairwise-{project}{suffix}"
    else:
        pattern = f"{project}_cumulative-fresh_iter*_embeddings.csv"
        subdir = results_dir / f"cumulative-fresh-{project}{suffix}"

    files = []
    for p in subdir.glob(pattern):
        name = p.stem
        if strategy == "pairwise":
            m = re.search(r"_pair(\d+)_", name)
        else:
            m = re.search(r"_iter(\d+)_", name)
        if not m:
            continue
        num = int(m.group(1))
        files.append((num, p))

    return sorted(files)


def load_metadata(
    results_dir: Path, project: str, strategy: str, dim_suffix: str = ""
) -> dict:
    """Load the pipeline metadata JSON."""
    suffix = f"-{dim_suffix}" if dim_suffix else ""
    if strategy == "pairwise":
        meta_path = (
            results_dir
            / f"pairwise-{project}{suffix}"
            / f"{project}_pairwise_pairwise_metadata.json"
        )
    else:
        meta_path = (
            results_dir
            / f"cumulative-fresh-{project}{suffix}"
            / f"{project}_cumulative-fresh_cumulative_fresh_metadata.json"
        )

    with open(meta_path) as f:
        return json.load(f)


def get_pair_metadata(metadata: dict, pair_num: int, strategy: str) -> dict:
    """Get metadata for a specific pair/iteration."""
    key = "pair" if strategy == "pairwise" else "iteration"
    for r in metadata["results"]:
        if r.get(key) == pair_num or r.get("pair") == pair_num:
            result = dict(r)
            if "version_a" not in result:
                result["version_a"] = ", ".join(result.get("train_versions", []))
            if "version_b" not in result:
                result["version_b"] = result.get("test_version", "")
            return result
    return {}


def write_results_csv(results: list[dict], output_path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "seed",
        "train_set",
        "classifier",
        "subset",
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_buggy",
        "f1_clean",
        "auc",
        "auprc",
        "precision_buggy",
        "recall_buggy",
        "precision_clean",
        "recall_clean",
        "far",
        "csi",
        "mcc",
        "n_leaked_train",
        "support_buggy",
        "support_clean",
        "total",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Wrote {output_path}")


def write_summary(results: list[dict], output_path: Path) -> None:
    """Write aggregate summary (mean across pairs) to CSV."""
    if not results:
        return
    df = pd.DataFrame(results)
    numeric_cols = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "f1_buggy",
        "f1_clean",
        "auc",
        "auprc",
        "precision_buggy",
        "recall_buggy",
        "precision_clean",
        "recall_clean",
        "far",
        "csi",
        "mcc",
    ]
    summary = df.groupby(["classifier", "train_set", "subset"])[numeric_cols].mean()
    summary = summary.round(4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    print(f"  Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure impact of train/test overlap on defect prediction in CVDP."
    )
    parser.add_argument("--project", required=True, choices=["django", "calcite"])
    parser.add_argument(
        "--strategy", default="pairwise", choices=["pairwise", "cumulative-fresh"]
    )
    parser.add_argument(
        "--results-dir", default="results", help="Directory with embedding CSVs"
    )
    parser.add_argument(
        "--output", default="results/prediction", help="Output directory"
    )
    parser.add_argument(
        "--dim-suffix",
        type=str,
        default="",
        help="Suffix for alternate dimension directories (e.g., '200' for pairwise-django-200)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds for RF (default 10)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    metadata = load_metadata(results_dir, args.project, args.strategy, args.dim_suffix)
    emb_files = find_embedding_files(
        results_dir, args.project, args.strategy, args.dim_suffix
    )

    if not emb_files:
        print(f"No embedding files found for {args.project}/{args.strategy}")
        sys.exit(1)

    print(f"Project: {args.project}, Strategy: {args.strategy}")
    print(f"Found {len(emb_files)} pairs/iterations")

    all_results = []

    for pair_num, emb_path in emb_files:
        pair_meta = get_pair_metadata(metadata, pair_num, args.strategy)
        if not pair_meta:
            print(f"  Pair {pair_num}: no metadata, skipping")
            continue

        df = load_embeddings(emb_path)
        dim_cols = get_dim_columns(df)

        if args.strategy == "pairwise":
            train, test = split_train_test(df, pair_meta)
        else:
            train, test = split_cumfresh_train_test(df, pair_meta)

        if len(train) == 0 or len(test) == 0:
            print(f"  Pair {pair_num}: empty train or test, skipping")
            continue

        new_files = compute_new_files(train, test)
        changed_label = compute_changed_label(train, test)

        pct_new = len(new_files) / len(test) * 100
        pct_changed = len(changed_label) / len(test) * 100
        print(
            f"  Pair {pair_num}: train={len(train)}, test={len(test)}, "
            f"new_files={len(new_files)} ({pct_new:.1f}%), "
            f"changed_label={len(changed_label)} ({pct_changed:.1f}%)"
        )

        pair_results = run_pair(
            pair_num, train, test, dim_cols, new_files, changed_label, args.seeds
        )
        all_results.extend(pair_results)

    # Write results
    dim_tag = f"_dim{args.dim_suffix}" if args.dim_suffix else ""
    out_prefix = f"{args.project}_{args.strategy}{dim_tag}"
    write_results_csv(all_results, output_dir / f"{out_prefix}_predictions.csv")
    write_summary(all_results, output_dir / f"{out_prefix}_summary.csv")

    # Print summary table
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {args.project} {args.strategy} ({args.seeds} seeds)")
        print(f"{'=' * 80}\n")
        for train_set in ["full", "cleaned"]:
            ts = df_res[df_res["train_set"] == train_set]
            if len(ts) == 0:
                continue
            print(f"  [{train_set} train]")
            for subset in ["baseline", "new_files", "changed_label"]:
                s = ts[ts["subset"] == subset]
                if len(s) > 0:
                    f1 = s["f1_macro"].mean()
                    auc = s["auc"].mean()
                    mcc = s["mcc"].mean()
                    n = s["total"].mean()
                    print(
                        f"    {subset:<16} F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  avg_n={n:.0f}"
                    )
            if train_set == "full":
                leaked = ts["n_leaked_train"].iloc[0] if len(ts) > 0 else 0
                print(f"    (leaked train instances removed: {leaked})")
            print()


if __name__ == "__main__":
    main()
