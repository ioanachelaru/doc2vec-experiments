#!/usr/bin/env python3
"""
pairwise_pipeline.py
====================
Cross-version embedding pipeline with fresh base model at each iteration.

Supports two strategies:
  pairwise         - Fine-tune on each consecutive pair of versions independently.
                     Each pair starts from a fresh copy of the base model.
  cumulative-fresh - Fine-tune on growing version windows (v0..v1, v0..v2, ...),
                     resetting to the base model each time.

Both strategies differ from cross_version_pipeline.py, which carries the
fine-tuned model forward across versions (cumulative-carried).

Output CSVs include version and label columns for downstream ML.

Tokenized documents are cached to disk (pickle) so that memory holds only
one version's data at a time, avoiding OOM on standard CI runners.
"""

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from utils import (
    clone_repo,
    get_version_tags,
)
from finetune_and_embed import (
    load_base_model,
    finetune_model,
    generate_embeddings_infer,
)
from analyze_duplicates import find_cross_version_duplicates

# Inline script run as a subprocess to tokenize one version.
# Each invocation starts a fresh Python process — when it exits, the OS
# reclaims ALL memory, eliminating heap fragmentation that accumulates
# across 20+ alloc/free cycles in a single long-lived process.
_TOKENIZE_WORKER = """\
import json, pickle, sys
from pathlib import Path
from utils import checkout_version, get_source_files, prepare_documents

repo_dir = Path(sys.argv[1])
version = sys.argv[2]
extensions = sys.argv[3].split(",")
search_path = Path(sys.argv[4])
pkl_path = sys.argv[5]
meta_path = sys.argv[6]

checkout_version(repo_dir, version)
files = get_source_files(search_path, extensions)
docs = prepare_documents(files, repo_dir, tag_prefix=version)
count = len(docs) if docs else 0
if docs:
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(meta_path, "w") as f:
    json.dump({"count": count}, f)
"""


class DiskBackedCorpus:
    """Streams TaggedDocuments from pickled version files.

    Loads one version at a time from disk during iteration, so peak memory
    is O(largest_version) instead of O(all_versions).  Supports multiple
    iterations (gensim calls __iter__ once per training epoch).
    """

    def __init__(self, paths: list[str], total_docs: int):
        self.paths = paths
        self._total_docs = total_docs

    def __iter__(self):
        for p in self.paths:
            with open(p, "rb") as f:
                yield from pickle.load(f)

    def __len__(self):
        return self._total_docs


def _load_labels(labels_dir: Path, version: str) -> dict[str, str]:
    """Load bug labels for a single version.

    Args:
        labels_dir: Directory containing label CSVs
        version: Version string (e.g., '1.0', 'calcite-1.0.0-incubating')

    Returns:
        Dict mapping filepath -> label ('buggy' or 'clean')
    """
    label_file = labels_dir / f"{version}.csv"
    if not label_file.exists():
        return {}
    df = pd.read_csv(label_file)
    return dict(zip(df["filepath"], df["label"]))


def _tokenize_to_disk(
    repo_dir: Path,
    versions: list[str],
    extensions: list[str],
    source_dir: str | None,
    cache_dir: Path,
) -> dict[str, dict]:
    """Checkout each version, tokenize files, and save docs to disk.

    Each version is tokenized in a **separate subprocess** so the OS reclaims
    all memory when the subprocess exits.  This prevents heap fragmentation
    from accumulating across 20+ versions in a single process (the root cause
    of OOM/SIGSEGV on 7 GB CI runners).

    Args:
        repo_dir: Path to cloned repository
        versions: List of version tags
        extensions: File extensions to include
        source_dir: Optional subdirectory filter
        cache_dir: Directory to store pickled document files

    Returns:
        Dict mapping version tag -> {"path": str, "count": int}
    """
    version_meta: dict[str, dict] = {}
    search_path = str(repo_dir / source_dir) if source_dir else str(repo_dir)
    src_dir = str(Path(__file__).parent)

    for v in versions:
        pkl_path = str(cache_dir / f"{v}.pkl")
        meta_path = str(cache_dir / f"{v}.meta.json")

        subprocess.run(
            [
                sys.executable,
                "-c",
                _TOKENIZE_WORKER,
                str(repo_dir),
                v,
                ",".join(extensions),
                search_path,
                pkl_path,
                meta_path,
            ],
            check=True,
            cwd=src_dir,
        )

        with open(meta_path) as f:
            count = json.load(f)["count"]

        if count > 0:
            version_meta[v] = {"path": pkl_path, "count": count}
            print(f"  {v}: {count} files (cached to disk)")
        else:
            print(f"  {v}: no source files, skipping")

    return version_meta


def _load_version_docs(meta: dict) -> list:
    """Load a single version's documents from its pickle cache file."""
    with open(meta["path"], "rb") as f:
        return pickle.load(f)


def _add_version_and_label(
    emb_df: pd.DataFrame,
    version: str,
    label_cache: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Add version and label columns to an embeddings DataFrame.

    Args:
        emb_df: DataFrame with file_path and dim_* columns
        version: Version string
        label_cache: Dict mapping version -> {filepath: label}

    Returns:
        DataFrame with version and label columns inserted after file_path
    """
    emb_df = emb_df.copy()
    emb_df.insert(1, "version", version)

    labels = label_cache.get(version, {})

    def get_label(file_path: str) -> str:
        _, rel_path = file_path.split("/", 1)
        return labels.get(rel_path, "unknown")

    emb_df.insert(2, "label", emb_df["file_path"].apply(get_label))
    return emb_df


def _leakage_stats(
    duplicates: list[dict], test_size: int
) -> tuple[int, float, int, int]:
    """Compute leakage stats from a list of duplicate pairs.

    Returns:
        (leakage_files, leakage_pct, same_file_count, collision_count)
    """
    test_files = {d["file_b"] for d in duplicates}
    pct = round(len(test_files) / test_size * 100, 2) if test_size > 0 else 0
    same = sum(1 for d in duplicates if d["duplicate_type"] == "same_file")
    coll = len(duplicates) - same
    return len(test_files), pct, same, coll


def _run_pairwise(
    versions: list[str],
    version_meta: dict[str, dict],
    base_model_path: str,
    label_cache: dict[str, dict[str, str]],
    epochs: int,
    threshold: float,
    output_prefix: str,
) -> list[dict]:
    """Run pairwise strategy: fresh base model for each consecutive pair.

    For each pair (vN, vN+1):
      1. Load fresh base model
      2. Fine-tune on vN + vN+1 documents together (streamed from disk)
      3. Generate embeddings for both versions via infer_vector
      4. Merge bug labels into the embeddings CSV
      5. Run duplicate/leakage analysis (vN = train, vN+1 = test)
    """
    results = []

    for i in range(len(versions) - 1):
        va, vb = versions[i], versions[i + 1]
        pair_num = i + 1
        print(f"\n{'=' * 60}")
        print(f"Pair {pair_num}: {va} + {vb}")
        print(f"{'=' * 60}")

        # Fresh base model
        model = load_base_model(base_model_path)

        # Fine-tune on both versions (streamed from disk)
        total_documents = version_meta[va]["count"] + version_meta[vb]["count"]
        corpus = DiskBackedCorpus(
            [version_meta[va]["path"], version_meta[vb]["path"]], total_documents
        )
        model = finetune_model(model, corpus, epochs=epochs, update_vocab=True)
        print(f"  Fine-tuned on {total_documents} documents, vocab={len(model.wv)}")
        del corpus

        # Embeddings via infer_vector (load one version at a time)
        docs_a = _load_version_docs(version_meta[va])
        emb_a = generate_embeddings_infer(model, docs_a)
        del docs_a

        docs_b = _load_version_docs(version_meta[vb])
        emb_b = generate_embeddings_infer(model, docs_b)
        del docs_b

        # Add version + label columns
        emb_a_labeled = _add_version_and_label(emb_a, va, label_cache)
        emb_b_labeled = _add_version_and_label(emb_b, vb, label_cache)

        # Save combined embeddings with labels
        combined = pd.concat([emb_a_labeled, emb_b_labeled], ignore_index=True)
        csv_path = f"{output_prefix}_pair{pair_num}_embeddings.csv"
        combined.to_csv(csv_path, index=False)
        print(f"  Saved {len(combined)} embeddings -> {csv_path}")

        # Duplicate analysis (va = train, vb = test)
        dup_result = find_cross_version_duplicates(emb_a, emb_b, va, vb, threshold)
        duplicates = dup_result["duplicates"]

        leak_files, leak_pct, same_n, coll_n = _leakage_stats(duplicates, len(emb_b))

        result = {
            "pair": pair_num,
            "version_a": va,
            "version_b": vb,
            "files_a": len(emb_a),
            "files_b": len(emb_b),
            "total_documents": total_documents,
            "vocab_size": len(model.wv),
            "duplicate_pairs": len(duplicates),
            "same_file_pairs": same_n,
            "collision_pairs": coll_n,
            "test_leakage_files": leak_files,
            "test_leakage_pct": leak_pct,
        }
        results.append(result)

        if duplicates:
            pd.DataFrame(duplicates).to_csv(
                f"{output_prefix}_pair{pair_num}_duplicates.csv", index=False
            )

        print(
            f"  Leakage: {leak_files}/{len(emb_b)} ({leak_pct}%) "
            f"[same_file={same_n}, collision={coll_n}]"
        )

        del model

    return results


def _run_cumulative_fresh(
    versions: list[str],
    version_meta: dict[str, dict],
    base_model_path: str,
    label_cache: dict[str, dict[str, str]],
    epochs: int,
    threshold: float,
    output_prefix: str,
    start_iter: int | None = None,
    end_iter: int | None = None,
) -> list[dict]:
    """Run cumulative-fresh strategy: fresh base model, growing version window.

    For each iteration i (2 versions, 3 versions, ...):
      1. Load fresh base model
      2. Fine-tune on all documents from v0..vi (streamed from disk)
      3. Generate embeddings for all versions via infer_vector
      4. Merge bug labels into the embeddings CSV
      5. Run leakage analysis (v0..vi-1 = train, vi = test)

    Args:
        start_iter: If set, skip iterations before this (1-based, inclusive)
        end_iter: If set, stop after this iteration (1-based, inclusive)
    """
    results = []

    for i in range(1, len(versions)):
        if start_iter is not None and i < start_iter:
            continue
        if end_iter is not None and i > end_iter:
            break

        current_versions = versions[: i + 1]
        iter_num = i
        train_versions = current_versions[:-1]
        test_version = current_versions[-1]

        print(f"\n{'=' * 60}")
        print(f"Iteration {iter_num}: {' + '.join(current_versions)}")
        print(f"  Train: {', '.join(train_versions)} | Test: {test_version}")
        print(f"{'=' * 60}")

        # Fresh base model
        model = load_base_model(base_model_path)

        # Fine-tune on all current versions (streamed from disk)
        total_documents = sum(version_meta[v]["count"] for v in current_versions)
        corpus = DiskBackedCorpus(
            [version_meta[v]["path"] for v in current_versions], total_documents
        )
        model = finetune_model(model, corpus, epochs=epochs, update_vocab=True)
        print(f"  Fine-tuned on {total_documents} documents, vocab={len(model.wv)}")
        del corpus

        # Embeddings: infer test version first, then stream train versions
        # to disk to limit peak memory
        test_docs = _load_version_docs(version_meta[test_version])
        test_emb = generate_embeddings_infer(model, test_docs)
        del test_docs

        test_size = len(test_emb)
        files_per_version = {test_version: test_size}

        # Write combined CSV incrementally (test version last)
        csv_path = f"{output_prefix}_iter{iter_num}_embeddings.csv"
        first_written = False
        train_size = 0
        all_leakage_dups = []

        for tv in train_versions:
            train_docs = _load_version_docs(version_meta[tv])
            train_emb = generate_embeddings_infer(model, train_docs)
            del train_docs

            files_per_version[tv] = len(train_emb)
            train_size += len(train_emb)

            # Leakage: this train version vs test
            dup_result = find_cross_version_duplicates(
                train_emb, test_emb, tv, test_version, threshold
            )
            all_leakage_dups.extend(dup_result["duplicates"])

            # Append labeled embeddings to CSV
            labeled = _add_version_and_label(train_emb, tv, label_cache)
            labeled.to_csv(csv_path, index=False, mode="a", header=not first_written)
            first_written = True
            del train_emb, labeled

        # Append test version
        test_labeled = _add_version_and_label(test_emb, test_version, label_cache)
        test_labeled.to_csv(csv_path, index=False, mode="a", header=not first_written)
        del test_labeled, test_emb

        total_rows = train_size + test_size
        print(f"  Saved {total_rows} embeddings -> {csv_path}")

        leak_files, leak_pct, same_n, coll_n = _leakage_stats(
            all_leakage_dups, test_size
        )

        result = {
            "iteration": iter_num,
            "versions": current_versions,
            "train_versions": train_versions,
            "test_version": test_version,
            "files_per_version": files_per_version,
            "total_documents": total_documents,
            "vocab_size": len(model.wv),
            "train_size": train_size,
            "test_size": test_size,
            "duplicate_pairs": len(all_leakage_dups),
            "same_file_pairs": same_n,
            "collision_pairs": coll_n,
            "test_leakage_files": leak_files,
            "test_leakage_pct": leak_pct,
        }
        results.append(result)

        if all_leakage_dups:
            pd.DataFrame(all_leakage_dups).to_csv(
                f"{output_prefix}_iter{iter_num}_leakage.csv", index=False
            )

        print(
            f"  Leakage: {leak_files}/{test_size} ({leak_pct}%) "
            f"[same_file={same_n}, collision={coll_n}]"
        )

        del model

    return results


def run_pipeline(
    strategy: str,
    repo_url: str,
    base_model_path: str,
    tag_regex: str,
    extensions: list[str],
    output_prefix: str,
    labels_dir: str | None = None,
    finetune_epochs: int = 10,
    threshold: float = 0.99,
    max_versions: int | None = None,
    source_dir: str | None = None,
    start_iter: int | None = None,
    end_iter: int | None = None,
) -> dict:
    """Run the embedding strategy pipeline.

    Args:
        strategy: 'pairwise' or 'cumulative-fresh'
        repo_url: GitHub repository URL
        base_model_path: Path to pre-trained base Doc2Vec model
        tag_regex: Regex pattern for git tags
        extensions: File extensions to include
        output_prefix: Prefix for output files
        labels_dir: Optional path to bug label CSVs
        finetune_epochs: Number of fine-tuning epochs
        threshold: Cosine similarity threshold
        max_versions: Optional limit on versions
        source_dir: Optional subdirectory filter
        start_iter: Start iteration for cumulative-fresh (1-based, inclusive)
        end_iter: End iteration for cumulative-fresh (1-based, inclusive)

    Returns:
        Metadata dict with configuration and results
    """
    start_time = time.time()

    # Step 1: Clone repo (full clone for tag access)
    print(f"\n{'=' * 60}")
    print("Step 1: Cloning repository")
    print(f"{'=' * 60}")
    repo_dir = clone_repo(repo_url, shallow=False)

    # Step 2: Discover version tags
    print(f"\n{'=' * 60}")
    print("Step 2: Discovering version tags")
    print(f"{'=' * 60}")
    versions = get_version_tags(repo_dir, tag_regex)

    if max_versions:
        versions = versions[:max_versions]

    if len(versions) < 2:
        print(f"Error: Need at least 2 versions, found {len(versions)}")
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise SystemExit(1)

    # For chunked cumulative-fresh, only tokenize the versions we need.
    # Iteration i uses versions 0..i, so end_iter needs versions 0..end_iter.
    if strategy == "cumulative-fresh" and end_iter is not None:
        needed = min(end_iter + 1, len(versions))
        if needed < len(versions):
            print(
                f"Chunk mode: trimming to {needed}/{len(versions)} versions "
                f"(iterations {start_iter or 1}-{end_iter})"
            )
            versions = versions[:needed]

    print(f"Versions to process ({len(versions)}):")
    for i, v in enumerate(versions):
        print(f"  {i + 1}. {v}")

    # Step 3: Tokenize all versions to disk (one at a time, memory-safe)
    print(f"\n{'=' * 60}")
    print("Step 3: Tokenizing all versions (disk-backed)")
    print(f"{'=' * 60}")
    cache_dir = Path(tempfile.mkdtemp(prefix="d2v_cache_"))
    version_meta = _tokenize_to_disk(
        repo_dir, versions, extensions, source_dir, cache_dir
    )
    shutil.rmtree(repo_dir, ignore_errors=True)

    versions_with_docs = [v for v in versions if v in version_meta]
    if len(versions_with_docs) < 2:
        print(
            f"Error: Need at least 2 versions with files, "
            f"found {len(versions_with_docs)}"
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        raise SystemExit(1)

    total_docs = sum(m["count"] for m in version_meta.values())
    print(f"\nTotal documents: {total_docs} across {len(versions_with_docs)} versions")

    # Step 4: Load labels (if provided)
    label_cache: dict[str, dict[str, str]] = {}
    if labels_dir:
        print(f"\n{'=' * 60}")
        print("Step 4: Loading bug labels")
        print(f"{'=' * 60}")
        labels_path = Path(labels_dir)
        for v in versions_with_docs:
            labels = _load_labels(labels_path, v)
            label_cache[v] = labels
            if labels:
                buggy = sum(1 for lbl in labels.values() if lbl == "buggy")
                print(
                    f"  {v}: {len(labels)} labels "
                    f"({buggy} buggy, {len(labels) - buggy} clean)"
                )
            else:
                print(f"  {v}: no labels found")

    # Step 5: Run strategy
    print(f"\n{'=' * 60}")
    print(f"Step 5: Running {strategy} strategy")
    print(f"{'=' * 60}")

    if strategy == "pairwise":
        results = _run_pairwise(
            versions_with_docs,
            version_meta,
            base_model_path,
            label_cache,
            finetune_epochs,
            threshold,
            output_prefix,
        )
    else:
        results = _run_cumulative_fresh(
            versions_with_docs,
            version_meta,
            base_model_path,
            label_cache,
            finetune_epochs,
            threshold,
            output_prefix,
            start_iter=start_iter,
            end_iter=end_iter,
        )

    # Cleanup cache
    shutil.rmtree(cache_dir, ignore_errors=True)

    # Step 6: Save metadata
    elapsed_time = time.time() - start_time
    metadata = {
        "strategy": strategy,
        "repo_url": repo_url,
        "tag_regex": tag_regex,
        "source_dir": source_dir,
        "labels_dir": labels_dir,
        "versions_analyzed": versions_with_docs,
        "files_per_version": {v: version_meta[v]["count"] for v in versions_with_docs},
        "embedding_mode": "infer_vector (epochs=200)",
        "finetune_epochs": finetune_epochs,
        "threshold": threshold,
        "results": results,
        "elapsed_time_minutes": round(elapsed_time / 60, 1),
    }

    strategy_slug = strategy.replace("-", "_")
    if start_iter is not None or end_iter is not None:
        s = start_iter or 1
        e = end_iter or "end"
        metadata_path = f"{output_prefix}_{strategy_slug}_metadata_chunk{s}_{e}.json"
    else:
        metadata_path = f"{output_prefix}_{strategy_slug}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"{strategy} analysis complete!")
    print(f"  Strategy: {strategy}")
    print(f"  Versions: {len(versions_with_docs)}")
    print(f"  Iterations: {len(results)}")
    print(f"  Total time: {elapsed_time / 60:.1f} minutes")
    print(f"  Metadata: {metadata_path}")
    print(f"{'=' * 60}")

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-version embedding pipeline with fresh base model strategies."
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["pairwise", "cumulative-fresh"],
        help="Training strategy: pairwise (fresh model per pair) or "
        "cumulative-fresh (fresh model, growing window)",
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
        "--ext", nargs="+", default=[".py"], help="File extensions to include"
    )
    parser.add_argument("--output", default="strategy", help="Output prefix for files")
    parser.add_argument(
        "--labels-dir",
        help="Directory containing {version}.csv bug label files",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Duplicate similarity threshold (default: 0.99)",
    )
    parser.add_argument(
        "--max-versions", type=int, help="Max number of versions to process"
    )
    parser.add_argument(
        "--source-dir",
        help="Subdirectory within repo to restrict file search (e.g., 'django')",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        help="Start iteration for cumulative-fresh (1-based, inclusive)",
    )
    parser.add_argument(
        "--end-iter",
        type=int,
        help="End iteration for cumulative-fresh (1-based, inclusive)",
    )

    args = parser.parse_args()

    print(f"   {args.strategy} embedding pipeline")
    print(f"   Repository: {args.repo}")
    print(f"   Tag regex: {args.tag_regex}")
    print(f"   Base model: {args.base_model}")
    print(f"   Extensions: {args.ext}")
    print(f"   Fine-tune epochs: {args.epochs}")
    print(f"   Threshold: {args.threshold}")
    if args.labels_dir:
        print(f"   Labels dir: {args.labels_dir}")
    if args.source_dir:
        print(f"   Source dir: {args.source_dir}")
    if args.max_versions:
        print(f"   Max versions: {args.max_versions}")
    if args.start_iter:
        print(f"   Start iteration: {args.start_iter}")
    if args.end_iter:
        print(f"   End iteration: {args.end_iter}")
    print()

    run_pipeline(
        args.strategy,
        args.repo,
        args.base_model,
        args.tag_regex,
        args.ext,
        args.output,
        labels_dir=args.labels_dir,
        finetune_epochs=args.epochs,
        threshold=args.threshold,
        max_versions=args.max_versions,
        source_dir=args.source_dir,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
    )
