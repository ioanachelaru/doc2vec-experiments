#!/usr/bin/env python3
"""
method_level_pipeline.py
========================
Method-level cross-version leakage analysis using Doc2Vec embeddings.

Extracts method bodies from pre-extracted source code zips and AST-level
data, trains Doc2Vec cumulatively across versions, and analyzes cross-version
duplicate/leakage at the method granularity.
"""

import argparse
import json
import logging
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
from gensim.models.doc2vec import TaggedDocument

from utils import tokenize_code
from finetune_and_embed import (
    load_base_model,
    finetune_model,
    generate_embeddings_infer,
)
from analyze_duplicates import (
    find_cross_version_duplicates,
    find_duplicates,
    generate_report,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like '1.10' into a sortable tuple (1, 10)."""
    return tuple(int(x) for x in version_str.split("."))


def discover_versions(data_dir: Path, version_prefix: str) -> list[str]:
    """Discover available versions from AST-level CSV filenames.

    Args:
        data_dir: Root data directory containing ast_level/ subdirectory
        version_prefix: Filename prefix (e.g., 'django' for 'django-1.0.csv')

    Returns:
        List of version strings sorted by semver
    """
    ast_dir = data_dir / "ast_level"
    pattern = re.compile(rf"^{re.escape(version_prefix)}-(.+)\.csv$")
    versions = []
    for f in ast_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            versions.append(m.group(1))
    versions.sort(key=parse_version)
    return versions


def _resolve_class_name(ast_df: pd.DataFrame, func_row: pd.Series) -> str | None:
    """Resolve the parent ClassDef name for a FunctionDef node.

    Args:
        ast_df: Full AST DataFrame for the file
        func_row: Row for the FunctionDef node

    Returns:
        Class name string, or None if parent is Module/not a class
    """
    parent_id = func_row["parent_id"]
    if pd.isna(parent_id):
        return None
    parent = ast_df[
        (ast_df["filepath"] == func_row["filepath"])
        & (ast_df["node_id"] == int(parent_id))
    ]
    if parent.empty:
        return None
    parent_row = parent.iloc[0]
    if parent_row["node_type"] == "ClassDef":
        return str(parent_row["node_info"])
    return None


def extract_methods_for_version(
    version: str,
    data_dir: Path,
    version_prefix: str,
) -> tuple[list[TaggedDocument], dict[str, dict]]:
    """Extract method bodies and metadata for a single version.

    Args:
        version: Version string (e.g., '1.0')
        data_dir: Root data directory
        version_prefix: Filename prefix (e.g., 'django')

    Returns:
        Tuple of (documents, metadata_dict) where metadata_dict maps
        tag -> {has_bugs, line_start, line_end, filepath, version, method_name}
    """
    ast_path = data_dir / "ast_level" / f"{version_prefix}-{version}.csv"
    zip_path = data_dir / "source_code" / f"{version_prefix}-{version}.zip"

    ast_df = pd.read_csv(ast_path).drop_duplicates(subset=["filepath", "node_id"])
    funcs = ast_df[ast_df["node_type"] == "FunctionDef"].copy()

    # Pre-compute class names for all functions (keyed by filepath+node_id)
    class_names = {}
    for _, row in funcs.iterrows():
        key = (row["filepath"], row["node_id"])
        class_names[key] = _resolve_class_name(ast_df, row)

    # Read all source files from zip into memory
    file_lines: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for filepath in funcs["filepath"].unique():
            try:
                with z.open(filepath) as f:
                    file_lines[filepath] = (
                        f.read().decode("utf-8", errors="ignore").split("\n")
                    )
            except KeyError:
                pass

    documents = []
    metadata = {}
    tag_counts: dict[str, int] = {}

    for _, row in funcs.iterrows():
        filepath = row["filepath"]
        method_name = str(row["node_info"])
        line_start = int(row["line_start"])
        line_end = int(row["line_end"])
        has_bugs = row["has_bugs"]

        lines = file_lines.get(filepath)
        if lines is None:
            continue

        body = "\n".join(lines[line_start - 1 : line_end])
        tokens = tokenize_code(body)
        if len(tokens) <= 5:
            continue

        # Build qualified method name with class prefix
        class_name = class_names.get((filepath, row["node_id"]))
        qualified = f"{class_name}.{method_name}" if class_name else method_name

        # Build tag with disambiguation
        base_tag = f"{version}/{filepath}::{qualified}"
        count = tag_counts.get(base_tag, 0)
        tag_counts[base_tag] = count + 1
        tag = f"{base_tag}#{count}" if count > 0 else base_tag

        documents.append(TaggedDocument(words=tokens, tags=[tag]))
        metadata[tag] = {
            "has_bugs": bool(has_bugs) if not pd.isna(has_bugs) else False,
            "line_start": line_start,
            "line_end": line_end,
            "filepath": filepath,
            "version": version,
            "method_name": qualified,
        }

    return documents, metadata


def compute_leakage_stats(
    consecutive_results: list[dict],
    versions: list[str],
    version_embeddings: dict[str, pd.DataFrame],
    cross_version_dups: dict[tuple[str, str], list[dict]],
    within_version_dups: dict[str, list[dict]],
    output_prefix: str,
) -> None:
    """Compute train/test leakage stats for each consecutive pair boundary.

    At each boundary (vN, vN+1), versions v1..vN form the training set and
    vN+1 is the test set. Updates consecutive_results dicts in-place.

    Args:
        consecutive_results: List of per-pair result dicts to update
        versions: Ordered list of version strings
        version_embeddings: Dict mapping version -> embeddings DataFrame
        cross_version_dups: All pairwise cross-version duplicate pairs
        within_version_dups: Within-version duplicate pairs
        output_prefix: Prefix for output CSV files
    """
    log.info("Computing per-pair leakage stats...")

    for idx, cr in enumerate(consecutive_results):
        va = cr["version_a"]
        vb = cr["version_b"]
        train_versions = versions[: versions.index(va) + 1]

        train_size = sum(len(version_embeddings[v]) for v in train_versions)
        test_size = len(version_embeddings[vb])

        # Within-training duplicates
        train_dups = []
        for a in range(len(train_versions)):
            train_dups.extend(within_version_dups.get(train_versions[a], []))
            for b in range(a + 1, len(train_versions)):
                key = (train_versions[a], train_versions[b])
                train_dups.extend(cross_version_dups.get(key, []))

        # Train-test leakage
        leakage_dups = []
        for tv in train_versions:
            key = (tv, vb)
            leakage_dups.extend(cross_version_dups.get(key, []))

        test_methods_with_leakage = {d["file_b"] for d in leakage_dups}
        leakage_pct = (
            round(len(test_methods_with_leakage) / test_size * 100, 2)
            if test_size > 0
            else 0
        )

        # Split by duplicate type
        same_method_leakage = [
            d for d in leakage_dups if d.get("duplicate_type") == "same_file"
        ]
        collision_leakage = [
            d for d in leakage_dups if d.get("duplicate_type") == "collision"
        ]
        same_method_test = {d["file_b"] for d in same_method_leakage}
        collision_test = {d["file_b"] for d in collision_leakage}

        cr.update(
            {
                "training_set_size": train_size,
                "training_duplicate_pairs": len(train_dups),
                "test_set_size": test_size,
                "test_entries_with_leakage": len(test_methods_with_leakage),
                "test_leakage_percentage": leakage_pct,
                "leakage_pairs": len(leakage_dups),
                "same_method_leakage_methods": len(same_method_test),
                "collision_leakage_methods": len(collision_test),
                "same_method_leakage_pairs": len(same_method_leakage),
                "collision_leakage_pairs": len(collision_leakage),
            }
        )

        log.info(
            "  Pair %d (%s vs %s): train=%d (%d dups), test=%d, "
            "leakage=%d methods (%.1f%%) [same_method=%d, collision=%d]",
            idx + 1,
            va,
            vb,
            train_size,
            len(train_dups),
            test_size,
            len(test_methods_with_leakage),
            leakage_pct,
            len(same_method_test),
            len(collision_test),
        )

        if train_dups:
            pd.DataFrame(train_dups).to_csv(
                f"{output_prefix}_pair{idx + 1}_train_duplicates.csv", index=False
            )
        if leakage_dups:
            pd.DataFrame(leakage_dups).to_csv(
                f"{output_prefix}_pair{idx + 1}_method_leakage.csv", index=False
            )


def enrich_leakage_with_labels(
    consecutive_results: list[dict],
    all_metadata: dict[str, dict],
    output_prefix: str,
) -> list[dict]:
    """Enrich leakage pairs with bug labels from AST data.

    Args:
        consecutive_results: List of per-pair result dicts
        all_metadata: Combined metadata dict (tag -> {has_bugs, ...})
        output_prefix: Prefix for output files

    Returns:
        List of per-pair summary dicts with label breakdown
    """
    log.info("\nEnriching leakage pairs with bug labels...")
    summaries = []

    for idx, cr in enumerate(consecutive_results):
        va = cr["version_a"]
        vb = cr["version_b"]
        leakage_path = Path(f"{output_prefix}_pair{idx + 1}_method_leakage.csv")

        if not leakage_path.exists():
            continue

        df = pd.read_csv(leakage_path)
        if df.empty:
            continue

        def get_label(tag: str) -> str:
            meta = all_metadata.get(tag)
            if meta is None:
                return "unknown"
            return "buggy" if meta["has_bugs"] else "clean"

        df["label_a"] = df["file_a"].apply(get_label)
        df["label_b"] = df["file_b"].apply(get_label)
        df["same_label"] = df["label_a"] == df["label_b"]

        labeled_path = f"{output_prefix}_pair{idx + 1}_method_leakage_labeled.csv"
        df.to_csv(labeled_path, index=False)

        # Stats on leaked test methods
        leaked_methods = df.drop_duplicates(subset="file_b")
        leaked_buggy = int((leaked_methods["label_b"] == "buggy").sum())
        leaked_clean = int((leaked_methods["label_b"] == "clean").sum())

        # Total test methods and their labels
        test_buggy = sum(
            1
            for tag, meta in all_metadata.items()
            if meta["version"] == vb and meta["has_bugs"]
        )
        test_clean = sum(
            1
            for tag, meta in all_metadata.items()
            if meta["version"] == vb and not meta["has_bugs"]
        )
        test_total = test_buggy + test_clean

        # Same/diff label pairs
        known_pairs = df[(df["label_a"] != "unknown") & (df["label_b"] != "unknown")]
        same_label = int(known_pairs["same_label"].sum())
        diff_label = len(known_pairs) - same_label

        buggy_leak_pct = (
            round(leaked_buggy / test_buggy * 100, 2) if test_buggy > 0 else 0
        )
        clean_leak_pct = (
            round(leaked_clean / test_clean * 100, 2) if test_clean > 0 else 0
        )

        # Split by duplicate_type
        has_type = "duplicate_type" in df.columns
        if has_type:
            sm = df[df["duplicate_type"] == "same_file"]
            co = df[df["duplicate_type"] == "collision"]
            sm_methods = sm.drop_duplicates(subset="file_b")
            co_methods = co.drop_duplicates(subset="file_b")
            sm_buggy = int((sm_methods["label_b"] == "buggy").sum())
            sm_clean = int((sm_methods["label_b"] == "clean").sum())
            co_buggy = int((co_methods["label_b"] == "buggy").sum())
            co_clean = int((co_methods["label_b"] == "clean").sum())
        else:
            sm_buggy = sm_clean = co_buggy = co_clean = 0

        summary = {
            "pair": idx + 1,
            "version_a": va,
            "version_b": vb,
            "test_total_methods": test_total,
            "test_buggy": test_buggy,
            "test_clean": test_clean,
            "leaked_total": leaked_buggy + leaked_clean,
            "leaked_buggy": leaked_buggy,
            "leaked_clean": leaked_clean,
            "leaked_same_label_pairs": same_label,
            "leaked_diff_label_pairs": diff_label,
            "buggy_leak_pct": buggy_leak_pct,
            "clean_leak_pct": clean_leak_pct,
            "same_method_leaked_buggy": sm_buggy,
            "same_method_leaked_clean": sm_clean,
            "collision_leaked_buggy": co_buggy,
            "collision_leaked_clean": co_clean,
        }
        summaries.append(summary)

        type_info = ""
        if has_type:
            type_info = (
                f" [same_method={sm_buggy}B/{sm_clean}C,"
                f" collision={co_buggy}B/{co_clean}C]"
            )

        log.info(
            "  Pair %d (%s vs %s): test=%d (%dB/%dC), "
            "leaked=%d (%dB/%dC), buggy_leak=%.1f%%, clean_leak=%.1f%%%s",
            idx + 1,
            va,
            vb,
            test_total,
            test_buggy,
            test_clean,
            leaked_buggy + leaked_clean,
            leaked_buggy,
            leaked_clean,
            buggy_leak_pct,
            clean_leak_pct,
            type_info,
        )

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = f"{output_prefix}_method_leakage_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        log.info("\nSummary saved to %s", summary_path)

        total_leaked_buggy = summary_df["leaked_buggy"].sum()
        total_leaked_clean = summary_df["leaked_clean"].sum()
        total_same = summary_df["leaked_same_label_pairs"].sum()
        total_diff = summary_df["leaked_diff_label_pairs"].sum()
        log.info("\nOverall across %d pairs:", len(summaries))
        log.info("  Leaked buggy test methods: %d", total_leaked_buggy)
        log.info("  Leaked clean test methods: %d", total_leaked_clean)
        log.info("  Same-label leakage pairs: %d", total_same)
        log.info("  Diff-label leakage pairs: %d", total_diff)

    return summaries


def run_method_level_pipeline(
    base_model_path: str,
    data_dir: str,
    output_prefix: str,
    threshold: float = 0.99,
    epochs: int = 10,
    version_prefix: str = "django",
    max_versions: int | None = None,
) -> dict:
    """Run the method-level cross-version leakage analysis pipeline.

    Args:
        base_model_path: Path to pre-trained Doc2Vec model
        data_dir: Path to data directory (with source_code/, ast_level/ subdirs)
        output_prefix: Prefix for output files
        threshold: Cosine similarity threshold for duplicate detection
        epochs: Fine-tuning epochs per version
        version_prefix: Filename prefix for data files (e.g., 'django')
        max_versions: Optional limit on number of versions to process

    Returns:
        Dict with method-level metadata and results
    """
    start_time = time.time()
    data_path = Path(data_dir)

    # Step 1: Discover versions
    log.info("=" * 60)
    log.info("Step 1: Discovering versions")
    log.info("=" * 60)

    versions = discover_versions(data_path, version_prefix)
    if max_versions:
        versions = versions[:max_versions]

    if len(versions) < 2:
        log.error("Need at least 2 versions, found %d", len(versions))
        raise SystemExit(1)

    log.info("Versions to process (%d):", len(versions))
    for i, v in enumerate(versions):
        log.info("  %d. %s", i + 1, v)

    # Step 2: Extract methods and train cumulatively
    log.info("\n" + "=" * 60)
    log.info("Step 2: Extract methods and train Doc2Vec")
    log.info("=" * 60)

    model = load_base_model(base_model_path)
    version_docs: dict[str, list] = {}
    methods_per_version: dict[str, int] = {}
    all_metadata: dict[str, dict] = {}
    versions_with_docs = []

    for v in versions:
        log.info("\nProcessing version %s...", v)
        docs, meta = extract_methods_for_version(v, data_path, version_prefix)

        if not docs:
            log.info("  %s: no methods with enough tokens, skipping", v)
            methods_per_version[v] = 0
            continue

        model = finetune_model(model, docs, epochs=epochs, update_vocab=True)
        version_docs[v] = docs
        methods_per_version[v] = len(docs)
        all_metadata.update(meta)
        versions_with_docs.append(v)

        # Count buggy methods for this version
        buggy_count = sum(1 for d in docs if meta[d.tags[0]]["has_bugs"])
        log.info(
            "  %s: %d methods (%d buggy), vocab=%d",
            v,
            len(docs),
            buggy_count,
            len(model.wv),
        )

    if len(versions_with_docs) < 2:
        log.error(
            "Need at least 2 versions with methods, found %d",
            len(versions_with_docs),
        )
        raise SystemExit(1)

    total_methods = sum(len(docs) for docs in version_docs.values())
    log.info(
        "\nTotal methods: %d across %d versions",
        total_methods,
        len(versions_with_docs),
    )
    log.info("Final vocab size: %d", len(model.wv))

    # Step 3: Generate embeddings via infer_vector
    log.info("\n" + "=" * 60)
    log.info("Step 3: Generating embeddings via infer_vector")
    log.info("=" * 60)

    version_embeddings = {}
    for v in versions_with_docs:
        emb = generate_embeddings_infer(model, version_docs[v])
        version_embeddings[v] = emb
        csv_path = f"{output_prefix}_{v}_method_embeddings.csv"
        emb.to_csv(csv_path, index=False)
        log.info("  %s: %d embeddings -> %s", v, len(emb), csv_path)

    # Step 4: Duplicate & leakage analysis
    log.info("\n" + "=" * 60)
    log.info("Step 4: Duplicate & leakage analysis")
    log.info("=" * 60)

    # Consecutive pairs
    log.info("\nAnalyzing consecutive version pairs...")
    consecutive_results = []
    for i in range(len(versions_with_docs) - 1):
        va, vb = versions_with_docs[i], versions_with_docs[i + 1]
        result = find_cross_version_duplicates(
            version_embeddings[va],
            version_embeddings[vb],
            va,
            vb,
            threshold,
        )
        pair_prefix = f"{output_prefix}_{va}_vs_{vb}"
        stats = generate_report(
            result["duplicates"], result["total_files"], threshold, pair_prefix
        )
        consecutive_results.append({"version_a": va, "version_b": vb, **stats})

    # All pairwise cross-version duplicates
    log.info("\nComputing all pairwise cross-version duplicates...")
    cross_version_dups: dict[tuple[str, str], list[dict]] = {}
    for a in range(len(versions_with_docs)):
        for b in range(a + 1, len(versions_with_docs)):
            va, vb = versions_with_docs[a], versions_with_docs[b]
            result = find_cross_version_duplicates(
                version_embeddings[va],
                version_embeddings[vb],
                va,
                vb,
                threshold,
            )
            cross_version_dups[(va, vb)] = result["duplicates"]

    total_cross = sum(len(d) for d in cross_version_dups.values())
    log.info(
        "  %d version pairs, %d total duplicate pairs",
        len(cross_version_dups),
        total_cross,
    )

    # Within-version duplicates
    log.info("Computing within-version duplicates...")
    within_version_dups: dict[str, list[dict]] = {}
    for v in versions_with_docs:
        paths = version_embeddings[v]["file_path"].tolist()
        vectors = version_embeddings[v].drop("file_path", axis=1).values
        within_version_dups[v] = find_duplicates(paths, vectors, threshold)

    total_within = sum(len(d) for d in within_version_dups.values())
    log.info("  %d total within-version duplicate pairs", total_within)

    # Leakage stats
    compute_leakage_stats(
        consecutive_results,
        versions_with_docs,
        version_embeddings,
        cross_version_dups,
        within_version_dups,
        output_prefix,
    )

    # Step 5: Enrich with bug labels
    log.info("\n" + "=" * 60)
    log.info("Step 5: Enriching leakage with bug labels")
    log.info("=" * 60)

    label_summaries = enrich_leakage_with_labels(
        consecutive_results, all_metadata, output_prefix
    )

    # Step 6: Summary & metadata
    log.info("\n" + "=" * 60)
    log.info("Step 6: Summary")
    log.info("=" * 60)

    elapsed_time = time.time() - start_time
    metadata = {
        "analysis_level": "method",
        "data_dir": str(data_dir),
        "versions_analyzed": versions_with_docs,
        "methods_per_version": methods_per_version,
        "embedding_mode": "infer_vector (seed=42, epochs=200)",
        "finetune_epochs": epochs,
        "threshold": threshold,
        "consecutive_pair_results": consecutive_results,
        "label_summaries": label_summaries,
        "overall_result": {
            "total_pairs_analyzed": len(consecutive_results),
            "total_duplicate_pairs": sum(
                r["duplicate_pairs"] for r in consecutive_results
            ),
        },
        "elapsed_time_minutes": round(elapsed_time / 60, 1),
    }

    metadata_path = f"{output_prefix}_method_level_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("\nMetadata saved to %s", metadata_path)
    log.info("\n" + "=" * 60)
    log.info("Method-level analysis complete!")
    log.info("  Versions: %d", len(versions_with_docs))
    log.info("  Total methods: %d", total_methods)
    log.info("  Total time: %.1f minutes", elapsed_time / 60)
    log.info("=" * 60)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method-level cross-version leakage analysis with Doc2Vec."
    )
    parser.add_argument(
        "--base-model", required=True, help="Path to pre-trained Doc2Vec model"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data directory with source_code/ and ast_level/ subdirs",
    )
    parser.add_argument(
        "--output-prefix",
        default="method_level",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold (default: 0.99)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--version-prefix",
        default="django",
        help="Filename prefix for data files (default: django)",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        help="Max number of versions to process",
    )

    args = parser.parse_args()

    log.info("   Method-level leakage analysis pipeline")
    log.info("   Base model: %s", args.base_model)
    log.info("   Data dir: %s", args.data_dir)
    log.info("   Threshold: %s", args.threshold)
    log.info("   Epochs: %s", args.epochs)
    log.info("   Version prefix: %s", args.version_prefix)
    if args.max_versions:
        log.info("   Max versions: %s", args.max_versions)
    log.info("")

    run_method_level_pipeline(
        args.base_model,
        args.data_dir,
        args.output_prefix,
        threshold=args.threshold,
        epochs=args.epochs,
        version_prefix=args.version_prefix,
        max_versions=args.max_versions,
    )
