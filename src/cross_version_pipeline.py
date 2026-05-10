#!/usr/bin/env python3
"""
cross_version_pipeline.py
=========================
Cumulatively train a Doc2Vec model across project versions,
generate embeddings for each version, and analyze cross-version duplicates.
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
    find_cross_version_duplicates,
    find_duplicates,
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
        print(f"  {i+1}. {v}")
    print(f"Training mode: cumulative (each pair embedded with same model)")

    # Step 3: Cumulative training, embedding, and pair analysis
    # For each consecutive pair (vN, vN+1):
    #   1. Train model on vN (cumulative)
    #   2. Embed both vN and vN+1 with the same model
    #   3. Compare the pair
    # This ensures each consecutive pair uses the same embedding space.
    print(f"\n{'='*60}")
    print("Step 3: Cumulative training, embedding, and pair analysis")
    print(f"{'='*60}")
    model = load_base_model(base_model_path)

    files_per_version = {}
    model_state_per_version = {}
    consecutive_results = []
    all_version_docs = {}  # keep docs in memory for end-of-pipeline leakage analysis

    for i in range(len(versions) - 1):
        v_current = versions[i]
        v_next = versions[i + 1]
        trained_on = [versions[j] for j in range(i + 1) if files_per_version.get(versions[j], 0) != 0 or j == i]

        print(f"\n{'='*60}")
        print(f"Pair {i+1}/{len(versions)-1}: {v_current} vs {v_next}")
        print(f"{'='*60}")

        # Train on current version
        print(f"\nTraining on {v_current}...")
        checkout_version(repo_dir, v_current)
        files_current = get_source_files(repo_dir, extensions)
        docs_current = prepare_documents(files_current, repo_dir, tag_prefix=v_current)

        if not docs_current:
            print(f"Warning: No source files in {v_current}, skipping pair")
            files_per_version[v_current] = 0
            continue

        model = finetune_model(model, docs_current, epochs=finetune_epochs, update_vocab=update_vocab)
        files_per_version[v_current] = len(docs_current)
        model_state_per_version[v_current] = f"trained on: {', '.join(trained_on)}"
        all_version_docs[v_current] = docs_current
        print(f"Vocab size: {len(model.wv)}")

        # Embed current version
        print(f"\nEmbedding {v_current}...")
        embeddings_current = generate_embeddings(model, docs_current)
        csv_current = f"{output_prefix}_{v_current}_pair{i+1}_embeddings.csv"
        embeddings_current.to_csv(csv_current, index=False)
        print(f"Saved {len(embeddings_current)} embeddings to {csv_current}")

        # Embed next version with the same model
        print(f"\nEmbedding {v_next}...")
        checkout_version(repo_dir, v_next)
        files_next = get_source_files(repo_dir, extensions)
        docs_next = prepare_documents(files_next, repo_dir, tag_prefix=v_next)

        if not docs_next:
            print(f"Warning: No source files in {v_next}, skipping pair")
            files_per_version[v_next] = 0
            continue

        embeddings_next = generate_embeddings(model, docs_next)
        files_per_version[v_next] = len(docs_next)
        all_version_docs[v_next] = docs_next
        csv_next = f"{output_prefix}_{v_next}_pair{i+1}_embeddings.csv"
        embeddings_next.to_csv(csv_next, index=False)
        print(f"Saved {len(embeddings_next)} embeddings to {csv_next}")

        # Compare the pair (consecutive version comparison)
        print(f"\nComparing {v_current} vs {v_next}...")
        result = find_cross_version_duplicates(
            embeddings_current, embeddings_next,
            v_current, v_next, threshold
        )

        pair_prefix = f"{output_prefix}_{v_current}_vs_{v_next}"
        stats = generate_report(result['duplicates'], result['total_files'], threshold, pair_prefix)

        consecutive_results.append({
            'version_a': v_current,
            'version_b': v_next,
            'model_trained_on': trained_on,
            **stats
        })

    # Train on the last version to complete the cumulative model
    last_v = versions[-1]
    if files_per_version.get(last_v, 0) > 0:
        print(f"\nTraining on final version {last_v}...")
        checkout_version(repo_dir, last_v)
        files_last = get_source_files(repo_dir, extensions)
        docs_last = prepare_documents(files_last, repo_dir, tag_prefix=last_v)
        if docs_last:
            all_version_docs[last_v] = docs_last
            model = finetune_model(model, docs_last, epochs=finetune_epochs, update_vocab=update_vocab)

    # Save final model
    model_path = f"{output_prefix}_finetuned.d2v"
    model.save(model_path)
    print(f"\nFinal model saved to {model_path}")

    # Cleanup the clone (docs are kept in memory for leakage analysis)
    shutil.rmtree(repo_dir, ignore_errors=True)

    # Step 4: Train/Test Leakage Analysis
    # Re-embed all versions with the final model so all embeddings share
    # the same embedding space, then compute within-training duplicates
    # and train-test leakage for each pair boundary.
    print(f"\n{'='*60}")
    print("Step 4: Train/Test Leakage Analysis")
    print(f"{'='*60}")

    versions_with_docs = [v for v in versions if v in all_version_docs and all_version_docs[v]]
    print(f"\nRe-embedding {len(versions_with_docs)} versions with final model...")

    version_embeddings = {}
    for v in versions_with_docs:
        emb = generate_embeddings(model, all_version_docs[v])
        version_embeddings[v] = emb
        print(f"  {v}: {len(emb)} files")

    # Free docs from memory now that we have embeddings
    all_version_docs.clear()

    # Compute all pairwise cross-version duplicates (once)
    print(f"\nComputing all pairwise cross-version duplicates...")
    cross_version_dups = {}
    for a in range(len(versions_with_docs)):
        for b in range(a + 1, len(versions_with_docs)):
            va, vb = versions_with_docs[a], versions_with_docs[b]
            result_pair = find_cross_version_duplicates(
                version_embeddings[va], version_embeddings[vb],
                va, vb, threshold
            )
            cross_version_dups[(va, vb)] = result_pair['duplicates']
    total_cross_dups = sum(len(d) for d in cross_version_dups.values())
    print(f"  {len(cross_version_dups)} version pairs, {total_cross_dups} total duplicate pairs")

    # Compute within-version duplicates (once)
    print(f"Computing within-version duplicates...")
    within_version_dups = {}
    for v in versions_with_docs:
        paths = version_embeddings[v]['file_path'].tolist()
        vectors = version_embeddings[v].drop('file_path', axis=1).values
        within_version_dups[v] = find_duplicates(paths, vectors, threshold)
    total_within_dups = sum(len(d) for d in within_version_dups.values())
    print(f"  {total_within_dups} total within-version duplicate pairs")

    # For each pair boundary, compute training dups and leakage
    print(f"\nComputing per-pair leakage stats...")
    for idx, cr in enumerate(consecutive_results):
        va = cr['version_a']
        vb = cr['version_b']
        va_idx = versions.index(va)
        train_versions = [v for v in versions[:va_idx + 1] if v in version_embeddings]
        test_version = vb

        if test_version not in version_embeddings or not train_versions:
            cr.update({
                'training_set_size': 0, 'training_duplicate_pairs': 0,
                'test_set_size': 0, 'test_entries_with_leakage': 0,
                'test_leakage_percentage': 0.0, 'leakage_pairs': 0,
            })
            continue

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

        test_files_with_leakage = set(d['file_b'] for d in leakage_dups)
        leakage_pct = round(len(test_files_with_leakage) / test_size * 100, 2) if test_size > 0 else 0

        cr.update({
            'training_set_size': train_size,
            'training_duplicate_pairs': len(train_dups),
            'test_set_size': test_size,
            'test_entries_with_leakage': len(test_files_with_leakage),
            'test_leakage_percentage': leakage_pct,
            'leakage_pairs': len(leakage_dups),
        })

        print(f"  Pair {idx+1} ({va} vs {vb}): train={train_size} ({len(train_dups)} dups), "
              f"test={test_size}, leakage={len(test_files_with_leakage)} files ({leakage_pct}%)")

        # Save CSVs
        if train_dups:
            pd.DataFrame(train_dups).to_csv(f"{output_prefix}_pair{idx+1}_train_duplicates.csv", index=False)
        if leakage_dups:
            pd.DataFrame(leakage_dups).to_csv(f"{output_prefix}_pair{idx+1}_leakage.csv", index=False)

    # Step 5: Overall summary
    print(f"\n{'='*60}")
    print("Step 5: Summary")
    print(f"{'='*60}")
    overall_stats = {
        'total_pairs_analyzed': len(consecutive_results),
        'total_duplicate_pairs': sum(r['duplicate_pairs'] for r in consecutive_results),
    }

    # Step 6: Save cross-version metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "repo_url": repo_url,
        "tag_regex": tag_regex,
        "versions_analyzed": versions,
        "files_per_version": files_per_version,
        "training_mode": "cumulative",
        "model_state_per_version": model_state_per_version,
        "finetune_epochs": finetune_epochs,
        "threshold": threshold,
        "consecutive_pair_results": consecutive_results,
        "overall_result": overall_stats,
        "training_approach": "train on v1..vN, embed vN and vN+1 with same model per pair",
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
