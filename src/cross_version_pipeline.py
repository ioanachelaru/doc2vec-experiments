#!/usr/bin/env python3
"""
cross_version_pipeline.py
=========================
Train a Doc2Vec model on all project versions, extract deterministic
embeddings from model.dv, and analyze cross-version duplicates + leakage.
"""

import sys
import shutil
import time
import argparse
import json
from pathlib import Path

import pandas as pd

from utils import (
    clone_repo,
    get_source_files,
    prepare_documents,
    get_version_tags,
    checkout_version,
)
from finetune_and_embed import (
    load_base_model,
    finetune_model,
    generate_embeddings_from_docvecs,
)
from analyze_duplicates import (
    find_cross_version_duplicates,
    find_duplicates,
    generate_report,
)


def _train_cumulatively(
    model,
    repo_dir: Path,
    versions: list[str],
    extensions: list[str],
    source_dir: str | None,
    finetune_epochs: int,
    update_vocab: bool,
) -> tuple[dict[str, list[str]], dict[str, int], list[str]]:
    """Checkout each version, train cumulatively, and store document tags.

    Trains on one version at a time to limit memory usage. After training,
    only the tag strings are kept — vectors are stored in model.dv.

    Args:
        model: Base Doc2Vec model to fine-tune
        repo_dir: Path to the cloned repository
        versions: List of version tags to process
        extensions: File extensions to include
        source_dir: Optional subdirectory to restrict file search
        finetune_epochs: Number of training epochs per version
        update_vocab: Whether to update vocabulary with new words

    Returns:
        Tuple of (version_tags, files_per_version, versions_with_docs)
        where version_tags maps version -> list of document tag strings
    """
    version_tags = {}
    files_per_version = {}
    search_path = repo_dir / source_dir if source_dir else repo_dir

    for v in versions:
        checkout_version(repo_dir, v)
        files = get_source_files(search_path, extensions)
        docs = prepare_documents(files, repo_dir, tag_prefix=v)

        if not docs:
            print(f"  {v}: no source files, skipping")
            files_per_version[v] = 0
            continue

        model = finetune_model(
            model, docs, epochs=finetune_epochs, update_vocab=update_vocab
        )
        version_tags[v] = [doc.tags[0] for doc in docs]
        files_per_version[v] = len(docs)
        print(f"  {v}: {len(docs)} files, vocab={len(model.wv)}")

    versions_with_docs = [v for v in versions if v in version_tags]
    return version_tags, files_per_version, versions_with_docs


def _analyze_consecutive_pairs(
    version_embeddings: dict[str, pd.DataFrame],
    versions: list[str],
    threshold: float,
    output_prefix: str,
) -> list[dict]:
    """Analyze duplicates between consecutive version pairs.

    Args:
        version_embeddings: Dict mapping version tag to embeddings DataFrame
        versions: Ordered list of version tags
        threshold: Cosine similarity threshold
        output_prefix: Prefix for output CSV files

    Returns:
        List of per-pair result dicts with duplicate stats
    """
    print("\nAnalyzing consecutive version pairs...")
    results = []

    for i in range(len(versions) - 1):
        va, vb = versions[i], versions[i + 1]

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

        results.append({"version_a": va, "version_b": vb, **stats})

    return results


def _compute_all_pairwise_duplicates(
    version_embeddings: dict[str, pd.DataFrame],
    versions: list[str],
    threshold: float,
) -> dict[tuple[str, str], list[dict]]:
    """Compute cross-version duplicates for all version pairs.

    Args:
        version_embeddings: Dict mapping version tag to embeddings DataFrame
        versions: Ordered list of version tags
        threshold: Cosine similarity threshold

    Returns:
        Dict mapping (version_a, version_b) tuples to lists of duplicate pairs
    """
    print("\nComputing all pairwise cross-version duplicates...")
    cross_version_dups = {}

    for a in range(len(versions)):
        for b in range(a + 1, len(versions)):
            va, vb = versions[a], versions[b]
            result = find_cross_version_duplicates(
                version_embeddings[va],
                version_embeddings[vb],
                va,
                vb,
                threshold,
            )
            cross_version_dups[(va, vb)] = result["duplicates"]

    total = sum(len(d) for d in cross_version_dups.values())
    print(f"  {len(cross_version_dups)} version pairs, {total} total duplicate pairs")
    return cross_version_dups


def _compute_within_version_duplicates(
    version_embeddings: dict[str, pd.DataFrame],
    versions: list[str],
    threshold: float,
) -> dict[str, list[dict]]:
    """Compute duplicates within each version.

    Args:
        version_embeddings: Dict mapping version tag to embeddings DataFrame
        versions: List of version tags
        threshold: Cosine similarity threshold

    Returns:
        Dict mapping version tag to list of duplicate pairs
    """
    print("Computing within-version duplicates...")
    within_dups = {}

    for v in versions:
        paths = version_embeddings[v]["file_path"].tolist()
        vectors = version_embeddings[v].drop("file_path", axis=1).values
        within_dups[v] = find_duplicates(paths, vectors, threshold)

    total = sum(len(d) for d in within_dups.values())
    print(f"  {total} total within-version duplicate pairs")
    return within_dups


def _compute_leakage_stats(
    consecutive_results: list[dict],
    versions: list[str],
    version_embeddings: dict[str, pd.DataFrame],
    cross_version_dups: dict[tuple[str, str], list[dict]],
    within_version_dups: dict[str, list[dict]],
    output_prefix: str,
) -> None:
    """Compute train/test leakage stats for each consecutive pair boundary.

    At each boundary (vN, vN+1), versions v1..vN form the training set and
    vN+1 is the test set. Computes within-training duplicates and test entries
    that have near-duplicates in the training set.

    Updates consecutive_results dicts in-place with leakage fields.

    Args:
        consecutive_results: List of per-pair result dicts to update
        versions: Ordered list of version tags
        version_embeddings: Dict mapping version tag to embeddings DataFrame
        cross_version_dups: All pairwise cross-version duplicate pairs
        within_version_dups: Within-version duplicate pairs
        output_prefix: Prefix for output CSV files
    """
    print("\nComputing per-pair leakage stats...")

    for idx, cr in enumerate(consecutive_results):
        va = cr["version_a"]
        vb = cr["version_b"]
        train_versions = versions[: versions.index(va) + 1]
        test_version = vb

        train_size = sum(len(version_embeddings[v]) for v in train_versions)
        test_size = len(version_embeddings[test_version])

        # Within-training duplicates: cross-version + within-version
        train_dups = []
        for a in range(len(train_versions)):
            train_dups.extend(within_version_dups.get(train_versions[a], []))
            for b in range(a + 1, len(train_versions)):
                key = (train_versions[a], train_versions[b])
                train_dups.extend(cross_version_dups.get(key, []))

        # Train-test leakage
        leakage_dups = []
        for tv in train_versions:
            key = (tv, test_version)
            leakage_dups.extend(cross_version_dups.get(key, []))

        test_files_with_leakage = {d["file_b"] for d in leakage_dups}
        leakage_pct = (
            round(len(test_files_with_leakage) / test_size * 100, 2)
            if test_size > 0
            else 0
        )

        # Split by duplicate type
        same_file_leakage = [
            d for d in leakage_dups if d.get("duplicate_type") == "same_file"
        ]
        collision_leakage = [
            d for d in leakage_dups if d.get("duplicate_type") == "collision"
        ]
        same_file_test = {d["file_b"] for d in same_file_leakage}
        collision_test = {d["file_b"] for d in collision_leakage}

        cr.update(
            {
                "training_set_size": train_size,
                "training_duplicate_pairs": len(train_dups),
                "test_set_size": test_size,
                "test_entries_with_leakage": len(test_files_with_leakage),
                "test_leakage_percentage": leakage_pct,
                "leakage_pairs": len(leakage_dups),
                "same_file_leakage_files": len(same_file_test),
                "collision_leakage_files": len(collision_test),
                "same_file_leakage_pairs": len(same_file_leakage),
                "collision_leakage_pairs": len(collision_leakage),
            }
        )

        print(
            f"  Pair {idx + 1} ({va} vs {vb}): train={train_size} ({len(train_dups)} dups), "
            f"test={test_size}, leakage={len(test_files_with_leakage)} files ({leakage_pct}%) "
            f"[same_file={len(same_file_test)}, collision={len(collision_test)}]"
        )

        if train_dups:
            pd.DataFrame(train_dups).to_csv(
                f"{output_prefix}_pair{idx + 1}_train_duplicates.csv", index=False
            )
        if leakage_dups:
            pd.DataFrame(leakage_dups).to_csv(
                f"{output_prefix}_pair{idx + 1}_leakage.csv", index=False
            )


def run_cross_version_pipeline(
    repo_url: str,
    base_model_path: str,
    tag_regex: str,
    extensions: list[str],
    output_prefix: str,
    finetune_epochs: int = 10,
    update_vocab: bool = True,
    threshold: float = 0.99,
    max_versions: int = None,
    source_dir: str = None,
) -> dict:
    """Run the cross-version embedding and duplicate analysis pipeline.

    Cumulatively trains a Doc2Vec model across versions (one at a time to limit
    memory), then extracts deterministic embeddings from model.dv (no
    infer_vector randomness). Analyzes duplicates between consecutive version
    pairs and computes train/test leakage.

    Args:
        repo_url: GitHub repository URL
        base_model_path: Path to pre-trained base Doc2Vec model
        tag_regex: Regex pattern for git tags
        extensions: File extensions to include
        output_prefix: Prefix for output files
        finetune_epochs: Number of fine-tuning epochs
        update_vocab: Whether to update vocabulary during fine-tuning
        threshold: Cosine similarity threshold for duplicate detection
        max_versions: Optional limit on number of versions to process
        source_dir: Optional subdirectory to restrict file search (e.g., 'django')

    Returns:
        Dict with cross-version metadata and results
    """
    start_time = time.time()

    # Step 1: Full clone to access all tags
    print(f"\n{'=' * 60}")
    print("Step 1: Cloning repository (full clone for tag access)")
    print(f"{'=' * 60}")
    repo_dir = clone_repo(repo_url, shallow=False)

    # Step 2: Extract and sort version tags
    print(f"\n{'=' * 60}")
    print("Step 2: Extracting version tags")
    print(f"{'=' * 60}")
    versions = get_version_tags(repo_dir, tag_regex)

    if len(versions) < 2:
        print(f"Error: Need at least 2 versions, found {len(versions)}")
        shutil.rmtree(repo_dir, ignore_errors=True)
        sys.exit(1)

    if max_versions:
        versions = versions[:max_versions]

    print(f"Versions to process ({len(versions)}):")
    for i, v in enumerate(versions):
        print(f"  {i + 1}. {v}")
    if source_dir:
        print(f"Source directory: {source_dir}")
    print("Embedding mode: model.dv (deterministic, cumulative training)")

    # Step 3: Cumulative training across versions
    # Train one version at a time to limit memory. After training each version,
    # only tag strings are kept — vectors are stored in model.dv.
    print(f"\n{'=' * 60}")
    print("Step 3: Cumulative training across versions")
    print(f"{'=' * 60}")

    model = load_base_model(base_model_path)
    version_tags, files_per_version, versions_with_docs = _train_cumulatively(
        model,
        repo_dir,
        versions,
        extensions,
        source_dir,
        finetune_epochs,
        update_vocab,
    )

    if len(versions_with_docs) < 2:
        print(
            f"Error: Need at least 2 versions with source files, "
            f"found {len(versions_with_docs)}"
        )
        shutil.rmtree(repo_dir, ignore_errors=True)
        sys.exit(1)

    shutil.rmtree(repo_dir, ignore_errors=True)

    total_docs = sum(len(tags) for tags in version_tags.values())
    print(f"\nTotal documents: {total_docs} across {len(versions_with_docs)} versions")
    print(f"Final vocab size: {len(model.wv)}")

    model_path = f"{output_prefix}_finetuned.d2v"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Step 4: Extract embeddings from model.dv
    print(f"\n{'=' * 60}")
    print("Step 4: Extracting embeddings from model.dv")
    print(f"{'=' * 60}")

    version_embeddings = {}
    for v in versions_with_docs:
        emb = generate_embeddings_from_docvecs(model, version_tags[v])
        version_embeddings[v] = emb
        csv_path = f"{output_prefix}_{v}_embeddings.csv"
        emb.to_csv(csv_path, index=False)
        print(f"  {v}: {len(emb)} embeddings -> {csv_path}")

    # Step 5: Duplicate & leakage analysis
    print(f"\n{'=' * 60}")
    print("Step 5: Duplicate & leakage analysis")
    print(f"{'=' * 60}")

    consecutive_results = _analyze_consecutive_pairs(
        version_embeddings, versions_with_docs, threshold, output_prefix
    )
    cross_version_dups = _compute_all_pairwise_duplicates(
        version_embeddings, versions_with_docs, threshold
    )
    within_version_dups = _compute_within_version_duplicates(
        version_embeddings, versions_with_docs, threshold
    )
    _compute_leakage_stats(
        consecutive_results,
        versions_with_docs,
        version_embeddings,
        cross_version_dups,
        within_version_dups,
        output_prefix,
    )

    # Step 6: Summary & metadata
    print(f"\n{'=' * 60}")
    print("Step 6: Summary")
    print(f"{'=' * 60}")

    overall_stats = {
        "total_pairs_analyzed": len(consecutive_results),
        "total_duplicate_pairs": sum(r["duplicate_pairs"] for r in consecutive_results),
    }

    elapsed_time = time.time() - start_time
    metadata = {
        "repo_url": repo_url,
        "tag_regex": tag_regex,
        "versions_analyzed": versions_with_docs,
        "files_per_version": files_per_version,
        "source_dir": source_dir,
        "embedding_mode": "model.dv (deterministic)",
        "finetune_epochs": finetune_epochs,
        "threshold": threshold,
        "consecutive_pair_results": consecutive_results,
        "overall_result": overall_stats,
        "training_approach": "train on all versions, extract from model.dv",
        "elapsed_time_minutes": round(elapsed_time / 60, 1),
    }

    metadata_path = f"{output_prefix}_cross_version_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nCross-version metadata saved to {metadata_path}")

    print(f"\n{'=' * 60}")
    print("Cross-version analysis complete!")
    print(f"  Versions: {len(versions_with_docs)}")
    print(f"  Total files: {sum(files_per_version.values())}")
    print(f"  Total time: {elapsed_time / 60:.1f} minutes")
    print(f"{'=' * 60}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-version Doc2Vec duplicate analysis pipeline."
    )
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument(
        "--base-model", required=True, help="Path to base Doc2Vec model"
    )
    parser.add_argument(
        "--tag-regex",
        required=True,
        help="Regex for version tags (e.g., '^[0-9]+\\.[0-9]+$')",
    )
    parser.add_argument(
        "--ext", nargs="+", default=[".java"], help="File extensions to include"
    )
    parser.add_argument(
        "--output", default="cross_version", help="Output prefix for files"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument(
        "--no-vocab-update", action="store_true", help="Don't update vocabulary"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.99, help="Duplicate similarity threshold"
    )
    parser.add_argument(
        "--max-versions", type=int, help="Max number of versions to process"
    )
    parser.add_argument(
        "--source-dir",
        help="Subdirectory within repo to restrict file search (e.g., 'django')",
    )

    args = parser.parse_args()

    print("   Cross-version duplicate analysis pipeline")
    print(f"   Repository: {args.repo}")
    print(f"   Tag regex: {args.tag_regex}")
    print(f"   Base model: {args.base_model}")
    print(f"   Extensions: {args.ext}")
    print(f"   Fine-tune epochs: {args.epochs}")
    print(f"   Threshold: {args.threshold}")
    if args.source_dir:
        print(f"   Source dir: {args.source_dir}")
    if args.max_versions:
        print(f"   Max versions: {args.max_versions}")
    print()

    run_cross_version_pipeline(
        args.repo,
        args.base_model,
        args.tag_regex,
        args.ext,
        args.output,
        finetune_epochs=args.epochs,
        update_vocab=not args.no_vocab_update,
        threshold=args.threshold,
        max_versions=args.max_versions,
        source_dir=args.source_dir,
    )
