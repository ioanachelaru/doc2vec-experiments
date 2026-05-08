#!/usr/bin/env python3
"""
cross_version_pipeline.py
=========================
Fine-tune a Doc2Vec model on the first version of a project,
generate embeddings for all versions, and analyze cross-version duplicates.
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
    get_repo_name_from_url,
    get_version_tags,
    checkout_version,
)
from finetune_and_embed import (
    load_base_model,
    finetune_model,
    generate_embeddings,
)
from analyze_duplicates import (
    find_duplicates,
    find_cross_version_duplicates,
    generate_report,
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
) -> dict:
    """Run the cross-version fine-tuning, embedding, and duplicate analysis pipeline.

    Args:
        repo_url: GitHub repository URL
        base_model_path: Path to pre-trained base Doc2Vec model
        tag_regex: Regex pattern for git tags (e.g., 'calcite-[0-9]+\\.[0-9]+\\.[0-9]+(-incubating)?$')
        extensions: File extensions to include
        output_prefix: Prefix for output files
        finetune_epochs: Number of fine-tuning epochs
        update_vocab: Whether to update vocabulary during fine-tuning
        threshold: Cosine similarity threshold for duplicate detection
        max_versions: Optional limit on number of versions to process

    Returns:
        Dict with cross-version metadata and results
    """
    start_time = time.time()

    # Step 1: Full clone to access all tags
    print(f"\n{'='*60}")
    print("Step 1: Cloning repository (full clone for tag access)")
    print(f"{'='*60}")
    repo_dir = clone_repo(repo_url, shallow=False)

    # Step 2: Extract and sort version tags
    print(f"\n{'='*60}")
    print("Step 2: Extracting version tags")
    print(f"{'='*60}")
    versions = get_version_tags(repo_dir, tag_regex)

    if len(versions) < 2:
        print(f"Error: Need at least 2 versions, found {len(versions)}")
        shutil.rmtree(repo_dir, ignore_errors=True)
        sys.exit(1)

    if max_versions:
        versions = versions[:max_versions]

    print(f"Versions to process ({len(versions)}):")
    for i, v in enumerate(versions):
        label = " (fine-tune target)" if i == 0 else ""
        print(f"  {i+1}. {v}{label}")

    # Step 3: Fine-tune on the first version
    print(f"\n{'='*60}")
    print(f"Step 3: Fine-tuning on {versions[0]}")
    print(f"{'='*60}")
    model = load_base_model(base_model_path)

    checkout_version(repo_dir, versions[0])
    files_v1 = get_source_files(repo_dir, extensions)
    docs_v1 = prepare_documents(files_v1, repo_dir, tag_prefix=versions[0])

    if not docs_v1:
        print(f"Error: No source files found in {versions[0]}")
        shutil.rmtree(repo_dir, ignore_errors=True)
        sys.exit(1)

    model = finetune_model(model, docs_v1, epochs=finetune_epochs, update_vocab=update_vocab)

    # Save fine-tuned model
    model_path = f"{output_prefix}_finetuned.d2v"
    model.save(model_path)
    print(f"Fine-tuned model saved to {model_path}")

    # Step 4: Generate embeddings for all versions
    print(f"\n{'='*60}")
    print("Step 4: Generating embeddings for all versions")
    print(f"{'='*60}")
    all_embeddings = []
    files_per_version = {}

    for i, version_tag in enumerate(versions):
        print(f"\n--- Version {i+1}/{len(versions)}: {version_tag} ---")
        checkout_version(repo_dir, version_tag)

        files = get_source_files(repo_dir, extensions)
        docs = prepare_documents(files, repo_dir, tag_prefix=version_tag)

        if not docs:
            print(f"Warning: No source files in {version_tag}, skipping")
            all_embeddings.append(None)
            files_per_version[version_tag] = 0
            continue

        embeddings_df = generate_embeddings(model, docs)
        files_per_version[version_tag] = len(embeddings_df)

        csv_path = f"{output_prefix}_{version_tag}_embeddings.csv"
        embeddings_df.to_csv(csv_path, index=False)
        print(f"Saved {len(embeddings_df)} embeddings to {csv_path}")

        all_embeddings.append(embeddings_df)

    # Cleanup the clone
    shutil.rmtree(repo_dir, ignore_errors=True)

    # Step 5: Analyze consecutive-pair duplicates
    print(f"\n{'='*60}")
    print("Step 5: Analyzing consecutive version pairs")
    print(f"{'='*60}")
    consecutive_results = []

    for i in range(len(versions) - 1):
        if all_embeddings[i] is None or all_embeddings[i + 1] is None:
            print(f"Skipping {versions[i]} vs {versions[i+1]} (missing embeddings)")
            continue

        v_a, v_b = versions[i], versions[i + 1]
        print(f"\n--- {v_a} vs {v_b} ---")

        result = find_cross_version_duplicates(
            all_embeddings[i], all_embeddings[i + 1],
            v_a, v_b, threshold
        )

        pair_prefix = f"{output_prefix}_{v_a}_vs_{v_b}"
        stats = generate_report(result['duplicates'], result['total_files'], threshold, pair_prefix)

        consecutive_results.append({
            'version_a': v_a,
            'version_b': v_b,
            **stats
        })

    # Step 6: Overall cross-version duplicate analysis
    # Compare every version pair (not within-version) to avoid building
    # a single N x N similarity matrix which can exceed memory limits.
    print(f"\n{'='*60}")
    print("Step 6: Overall cross-version duplicate analysis")
    print(f"{'='*60}")
    valid_indices = [i for i, df in enumerate(all_embeddings) if df is not None]

    overall_stats = {}
    if len(valid_indices) >= 2:
        try:
            all_duplicates = []
            total_files = sum(len(all_embeddings[i]) for i in valid_indices)

            for idx_a in range(len(valid_indices)):
                for idx_b in range(idx_a + 1, len(valid_indices)):
                    i, j = valid_indices[idx_a], valid_indices[idx_b]
                    v_a, v_b = versions[i], versions[j]
                    result = find_cross_version_duplicates(
                        all_embeddings[i], all_embeddings[j],
                        v_a, v_b, threshold
                    )
                    all_duplicates.extend(result['duplicates'])

            all_duplicates.sort(key=lambda x: x['similarity'], reverse=True)
            overall_stats = generate_report(
                all_duplicates, total_files, threshold,
                f"{output_prefix}_all_versions"
            )
        except MemoryError:
            print("WARNING: Overall analysis ran out of memory, skipping.")
            print("Consecutive pair results are still available.")
    else:
        print("Not enough valid versions for overall analysis")

    # Step 7: Save cross-version metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "repo_url": repo_url,
        "tag_regex": tag_regex,
        "versions_analyzed": versions,
        "files_per_version": files_per_version,
        "finetuned_on": versions[0],
        "finetune_epochs": finetune_epochs,
        "threshold": threshold,
        "consecutive_pair_results": consecutive_results,
        "overall_result": overall_stats,
        "elapsed_time_minutes": round(elapsed_time / 60, 1),
    }

    metadata_path = f"{output_prefix}_cross_version_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nCross-version metadata saved to {metadata_path}")

    print(f"\n{'='*60}")
    print(f"Cross-version analysis complete!")
    print(f"  Versions: {len(versions)}")
    print(f"  Total files: {sum(files_per_version.values())}")
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print(f"{'='*60}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-version Doc2Vec duplicate analysis pipeline."
    )
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--base-model", required=True, help="Path to base Doc2Vec model")
    parser.add_argument("--tag-regex", required=True, help="Regex for version tags (e.g., 'calcite-[0-9]+\\.[0-9]+\\.[0-9]+(-incubating)?$')")
    parser.add_argument("--ext", nargs="+", default=[".java"], help="File extensions to include")
    parser.add_argument("--output", default="cross_version", help="Output prefix for files")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--no-vocab-update", action="store_true", help="Don't update vocabulary")
    parser.add_argument("--threshold", type=float, default=0.99, help="Duplicate similarity threshold")
    parser.add_argument("--max-versions", type=int, help="Max number of versions to process")

    args = parser.parse_args()

    print(f"   Cross-version duplicate analysis pipeline")
    print(f"   Repository: {args.repo}")
    print(f"   Tag regex: {args.tag_regex}")
    print(f"   Base model: {args.base_model}")
    print(f"   Extensions: {args.ext}")
    print(f"   Fine-tune epochs: {args.epochs}")
    print(f"   Threshold: {args.threshold}")
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
    )
