#!/usr/bin/env python3
"""
predict_with_leakage.py
=======================
Measure how train/test overlap inflates defect prediction in CVDP.

For each version pair, trains classifiers on embedding vectors and evaluates
on three subsets: full test (baseline), non-leaked files (cleaned), and
leaked-only files. The delta between baseline and cleaned quantifies the
inflation caused by code overlap.

Supports two definitions of "leaked":
  - Embedding-based: cosine similarity >= threshold between same-path files
  - Same-code: ground truth from source code comparison (external CSV)

Usage:
    # Basic run
    python src/predict_with_leakage.py --project django --strategy pairwise

    # With same-code ground truth comparison
    python src/predict_with_leakage.py --project django --strategy pairwise \
        --same-code-zip "resources/django 1/django-same-code.zip"

    # Threshold sweep
    python src/predict_with_leakage.py --project django --strategy pairwise --sweep

    # Both projects
    python src/predict_with_leakage.py --project django --strategy pairwise
    python src/predict_with_leakage.py --project calcite --strategy pairwise
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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
    # Drop unknown labels (Calcite has these)
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
    # Try float comparison (handles 1.10 -> 1.1 case)
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


def compute_leaked_files_embedding(
    train: pd.DataFrame, test: pd.DataFrame, dim_cols: list[str], threshold: float
) -> set[str]:
    """Find test files with a same-path near-duplicate in train (cosine sim >= threshold)."""
    train_by_path = {}
    for _, row in train.iterrows():
        rel = extract_relative_path(row["file_path"])
        train_by_path[rel] = row[dim_cols].values.astype(np.float64)

    leaked = set()
    for _, row in test.iterrows():
        rel = extract_relative_path(row["file_path"])
        if rel in train_by_path:
            test_vec = row[dim_cols].values.astype(np.float64).reshape(1, -1)
            train_vec = train_by_path[rel].reshape(1, -1)
            sim = cosine_similarity(test_vec, train_vec)[0, 0]
            # Clamp to [-1, 1] for floating-point safety
            sim = np.clip(sim, -1.0, 1.0)
            if sim >= threshold:
                leaked.add(row["file_path"])

    return leaked


def load_same_code_files(zip_path: str, pair: int, project: str) -> set[str]:
    """Load the set of test file relative paths from same-code ground truth."""
    fname = f"{project}_pairwise_pair{pair}_same_code.csv"
    result = subprocess.run(
        ["unzip", "-p", zip_path, fname],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip().count("\n") < 1:
        return set()

    reader = csv.DictReader(result.stdout.strip().split("\n"))
    paths = set()
    for row in reader:
        # filename2 is the test version file; strip version prefix
        rel = "/".join(row["filename2"].split("/")[1:])
        paths.add(rel)
    return paths


def compute_leaked_files_same_code(
    test: pd.DataFrame, same_code_paths: set[str]
) -> set[str]:
    """Find test files that are in the same-code ground truth set."""
    leaked = set()
    for _, row in test.iterrows():
        rel = extract_relative_path(row["file_path"])
        if rel in same_code_paths:
            leaked.add(row["file_path"])
    return leaked


def evaluate_subset(
    clf, X: np.ndarray, y: np.ndarray, clf_name: str, subset_name: str
) -> dict | None:
    """Evaluate classifier on a subset, return metrics dict or None if subset is empty."""
    if len(X) == 0 or len(np.unique(y)) == 0:
        return None

    y_pred = clf.predict(X)
    buggy_count = int(np.sum(y == 1))
    clean_count = int(np.sum(y == 0))

    metrics = {
        "classifier": clf_name,
        "subset": subset_name,
        "total": len(y),
        "support_buggy": buggy_count,
        "support_clean": clean_count,
    }

    if len(np.unique(y)) < 2:
        # Single class in subset — limited metrics
        metrics["f1_macro"] = f1_score(y, y_pred, average="macro", zero_division=0)
        metrics["precision_buggy"] = np.nan
        metrics["recall_buggy"] = np.nan
        metrics["precision_clean"] = np.nan
        metrics["recall_clean"] = np.nan
        metrics["mcc"] = np.nan
        metrics["auc"] = np.nan
    else:
        metrics["f1_macro"] = f1_score(y, y_pred, average="macro", zero_division=0)
        metrics["precision_buggy"] = precision_score(
            y, y_pred, pos_label=1, zero_division=0
        )
        metrics["recall_buggy"] = recall_score(y, y_pred, pos_label=1, zero_division=0)
        metrics["precision_clean"] = precision_score(
            y, y_pred, pos_label=0, zero_division=0
        )
        metrics["recall_clean"] = recall_score(y, y_pred, pos_label=0, zero_division=0)
        metrics["mcc"] = matthews_corrcoef(y, y_pred)
        try:
            y_proba = clf.predict_proba(X)[:, 1]
            metrics["auc"] = roc_auc_score(y, y_proba)
        except (ValueError, IndexError):
            metrics["auc"] = np.nan

    return metrics


def run_pair(
    pair: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    dim_cols: list[str],
    leaked_files: set[str],
    leakage_method: str,
) -> list[dict]:
    """Run classification for one pair, return list of metric dicts."""
    X_train = train[dim_cols].values.astype(np.float64)
    y_train = (train["label"] == "buggy").astype(int).values

    # Split test into leaked and cleaned
    test_leaked_mask = test["file_path"].isin(leaked_files)
    test_cleaned = test[~test_leaked_mask]
    test_leaked = test[test_leaked_mask]

    X_test_full = test[dim_cols].values.astype(np.float64)
    y_test_full = (test["label"] == "buggy").astype(int).values

    X_test_cleaned = test_cleaned[dim_cols].values.astype(np.float64)
    y_test_cleaned = (test_cleaned["label"] == "buggy").astype(int).values

    X_test_leaked = test_leaked[dim_cols].values.astype(np.float64)
    y_test_leaked = (test_leaked["label"] == "buggy").astype(int).values

    # Check we have enough training data
    if len(np.unique(y_train)) < 2:
        print(f"  Pair {pair}: skipping — single class in train")
        return []

    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
    }

    results = []
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        for subset_name, X_sub, y_sub in [
            ("baseline", X_test_full, y_test_full),
            ("cleaned", X_test_cleaned, y_test_cleaned),
            ("leaked-only", X_test_leaked, y_test_leaked),
        ]:
            m = evaluate_subset(clf, X_sub, y_sub, clf_name, subset_name)
            if m is not None:
                m["pair"] = pair
                m["leakage_method"] = leakage_method
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
        # Extract pair/iter number from filename
        # e.g. django_pairwise_pair12_embeddings -> 12
        # e.g. django_cumulative-fresh_iter5_embeddings -> 5
        import re

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
        # Cumulative-fresh has a merged metadata
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
            # Normalize field names
            if "version_a" not in result:
                result["version_a"] = ", ".join(result.get("train_versions", []))
            if "version_b" not in result:
                result["version_b"] = result.get("test_version", "")
            return result
    return {}


def split_cumfresh_train_test(
    df: pd.DataFrame, pair_meta: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cumulative-fresh embeddings: all versions except last = train, last = test."""
    test_version = str(pair_meta.get("version_b", pair_meta.get("test_version", "")))
    test_mask = _version_matches(df["version"], test_version)
    test = df[test_mask].copy()
    train = df[~test_mask].copy()
    return train, test


def write_results_csv(results: list[dict], output_path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pair",
        "leakage_method",
        "classifier",
        "subset",
        "f1_macro",
        "auc",
        "precision_buggy",
        "recall_buggy",
        "precision_clean",
        "recall_clean",
        "mcc",
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
        "f1_macro",
        "auc",
        "precision_buggy",
        "recall_buggy",
        "precision_clean",
        "recall_clean",
        "mcc",
    ]
    summary = df.groupby(["leakage_method", "classifier", "subset"])[
        numeric_cols
    ].mean()
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
        "--threshold", type=float, default=0.99, help="Cosine similarity threshold"
    )
    parser.add_argument(
        "--same-code-zip",
        type=str,
        default=None,
        help="Path to same-code ground truth ZIP archive",
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Run threshold sweep from 0.90 to 1.00"
    )
    parser.add_argument(
        "--dim-suffix",
        type=str,
        default="",
        help="Suffix for alternate dimension directories (e.g., '200' for pairwise-django-200)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    # Load metadata and find embedding files
    metadata = load_metadata(results_dir, args.project, args.strategy, args.dim_suffix)
    emb_files = find_embedding_files(
        results_dir, args.project, args.strategy, args.dim_suffix
    )

    if not emb_files:
        print(f"No embedding files found for {args.project}/{args.strategy}")
        sys.exit(1)

    print(f"Project: {args.project}, Strategy: {args.strategy}")
    print(f"Found {len(emb_files)} pairs/iterations")
    print(f"Threshold: {args.threshold}")

    # ── Main experiment ──
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

        # Embedding-based leakage
        leaked_emb = compute_leaked_files_embedding(
            train, test, dim_cols, args.threshold
        )
        pct = len(leaked_emb) / len(test) * 100
        print(
            f"  Pair {pair_num}: train={len(train)}, test={len(test)}, "
            f"leaked(emb@{args.threshold})={len(leaked_emb)} ({pct:.1f}%)"
        )

        pair_results = run_pair(
            pair_num, train, test, dim_cols, leaked_emb, f"embedding_{args.threshold}"
        )
        all_results.extend(pair_results)

        # Same-code leakage (if provided)
        if args.same_code_zip:
            sc_paths = load_same_code_files(args.same_code_zip, pair_num, args.project)
            if sc_paths:
                leaked_sc = compute_leaked_files_same_code(test, sc_paths)
                pct_sc = len(leaked_sc) / len(test) * 100
                print(f"           leaked(same-code)={len(leaked_sc)} ({pct_sc:.1f}%)")
                sc_results = run_pair(
                    pair_num, train, test, dim_cols, leaked_sc, "same_code"
                )
                all_results.extend(sc_results)

    # Write main results
    dim_tag = f"_dim{args.dim_suffix}" if args.dim_suffix else ""
    out_prefix = f"{args.project}_{args.strategy}{dim_tag}"
    write_results_csv(all_results, output_dir / f"{out_prefix}_predictions.csv")
    write_summary(all_results, output_dir / f"{out_prefix}_summary.csv")

    # Print summary table
    if all_results:
        df_res = pd.DataFrame(all_results)
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {args.project} {args.strategy}")
        print(f"{'=' * 80}")
        for method in df_res["leakage_method"].unique():
            print(f"\n  Leakage method: {method}")
            sub = df_res[df_res["leakage_method"] == method]
            for clf in sub["classifier"].unique():
                print(f"  {clf}:")
                for subset in ["baseline", "cleaned", "leaked-only"]:
                    s = sub[(sub["classifier"] == clf) & (sub["subset"] == subset)]
                    if len(s) > 0:
                        f1 = s["f1_macro"].mean()
                        auc = s["auc"].mean()
                        mcc = s["mcc"].mean()
                        n = s["total"].mean()
                        print(
                            f"    {subset:<14} F1={f1:.4f}  AUC={auc:.4f}  MCC={mcc:.4f}  avg_n={n:.0f}"
                        )

    # ── Threshold sweep ──
    if args.sweep:
        print(f"\n{'=' * 80}")
        print("THRESHOLD SWEEP")
        print(f"{'=' * 80}")
        thresholds = [round(0.90 + i * 0.01, 2) for i in range(11)]
        sweep_results = []

        for pair_num, emb_path in emb_files:
            pair_meta = get_pair_metadata(metadata, pair_num, args.strategy)
            if not pair_meta:
                continue

            df = load_embeddings(emb_path)
            dim_cols = get_dim_columns(df)

            if args.strategy == "pairwise":
                train, test = split_train_test(df, pair_meta)
            else:
                train, test = split_cumfresh_train_test(df, pair_meta)

            if len(train) == 0 or len(test) == 0:
                continue

            # Precompute all similarities for this pair
            train_by_path = {}
            for _, row in train.iterrows():
                rel = extract_relative_path(row["file_path"])
                train_by_path[rel] = row[dim_cols].values.astype(np.float64)

            test_sims = {}  # file_path -> similarity
            for _, row in test.iterrows():
                rel = extract_relative_path(row["file_path"])
                if rel in train_by_path:
                    test_vec = row[dim_cols].values.astype(np.float64).reshape(1, -1)
                    train_vec = train_by_path[rel].reshape(1, -1)
                    sim = np.clip(
                        cosine_similarity(test_vec, train_vec)[0, 0], -1.0, 1.0
                    )
                    test_sims[row["file_path"]] = sim

            for t in thresholds:
                leaked = {fp for fp, sim in test_sims.items() if sim >= t}
                pair_results = run_pair(
                    pair_num, train, test, dim_cols, leaked, f"embedding_{t}"
                )
                for r in pair_results:
                    r["threshold"] = t
                    sweep_results.append(r)

            print(f"  Pair {pair_num}: sweep done")

        # Write sweep results
        if sweep_results:
            sweep_path = output_dir / f"{out_prefix}_threshold_sweep.csv"
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "pair",
                "threshold",
                "leakage_method",
                "classifier",
                "subset",
                "f1_macro",
                "auc",
                "mcc",
                "support_buggy",
                "support_clean",
                "total",
            ]
            with open(sweep_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(sweep_results)
            print(f"  Wrote {sweep_path}")

            # Print sweep summary
            df_sweep = pd.DataFrame(sweep_results)
            print("\n  Threshold sweep summary (RandomForest, F1 macro):")
            print(
                f"  {'Threshold':<11} {'Baseline':<10} {'Cleaned':<10} {'Delta':<10} {'Leaked-only':<12}"
            )
            print(f"  {'-' * 55}")
            for t in thresholds:
                sub = df_sweep[
                    (df_sweep["threshold"] == t)
                    & (df_sweep["classifier"] == "RandomForest")
                ]
                bl = sub[sub["subset"] == "baseline"]["f1_macro"].mean()
                cl = sub[sub["subset"] == "cleaned"]["f1_macro"].mean()
                lo = sub[sub["subset"] == "leaked-only"]["f1_macro"].mean()
                delta = bl - cl
                print(
                    f"  {t:<11.2f} {bl:<10.4f} {cl:<10.4f} {delta:<+10.4f} {lo:<12.4f}"
                )


if __name__ == "__main__":
    main()
