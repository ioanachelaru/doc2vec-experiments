#!/usr/bin/env python3
"""
train_base_model.py
====================
Train a base Doc2Vec model on multiple popular GitHub repositories.
This model can later be fine-tuned on specific repositories.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import argparse
import json
import time
from multiprocessing import Pool, cpu_count
import traceback

from utils import (
    clone_repo,
    get_source_files,
    prepare_documents,
    get_repo_name_from_url
)


def process_repository(args):
    """Process a single repository and return documents."""
    repo_url, extensions = args
    try:
        print(f"Processing: {repo_url}")
        repo_dir = clone_repo(repo_url)

        source_files = get_source_files(repo_dir, extensions)
        documents = []
        if source_files:
            repo_name = get_repo_name_from_url(repo_url)
            documents = prepare_documents(source_files, repo_dir, tag_prefix=repo_name)
            print(f"✓ Added {len(documents)} documents from {repo_url}")
        else:
            print(f"⚠ No source files found in {repo_url}")

        # Cleanup immediately to save disk space
        shutil.rmtree(repo_dir, ignore_errors=True)
        return documents
    except Exception as e:
        print(f"✗ Error processing {repo_url}: {e}")
        return []


def train_base_model(
    repo_urls: list[str],
    extensions: list[str],
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 20,
    dm: int = 1,
    batch_size: int = 10,
    parallel_workers: int = None,
) -> Doc2Vec:
    """Train a base Doc2Vec model on multiple repositories.

    Args:
        repo_urls: List of repository URLs to train on
        extensions: File extensions to include
        vector_size: Embedding dimension
        window: Context window size
        min_count: Minimum word frequency
        epochs: Training epochs
        dm: Training algorithm (1=PV-DM, 0=PV-DBOW)
        batch_size: Number of repos to process before incremental training
        parallel_workers: Number of parallel workers for repo processing
    """

    all_documents = []

    # Determine number of parallel workers
    if parallel_workers is None:
        parallel_workers = min(4, cpu_count() or 1)

    print(f"\nProcessing {len(repo_urls)} repositories with {parallel_workers} parallel workers...")
    print(f"Batch size: {batch_size} repos per training batch\n")

    # Process repositories in batches to manage memory
    model = None
    total_processed = 0

    for batch_start in range(0, len(repo_urls), batch_size):
        batch_end = min(batch_start + batch_size, len(repo_urls))
        batch_repos = repo_urls[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start//batch_size + 1} (repos {batch_start+1}-{batch_end}/{len(repo_urls)})")
        print(f"{'='*60}")

        # Process batch in parallel
        batch_documents = []
        with Pool(processes=parallel_workers) as pool:
            args_list = [(repo, extensions) for repo in batch_repos]
            results = pool.map(process_repository, args_list)

            for docs in results:
                batch_documents.extend(docs)

        all_documents.extend(batch_documents)
        total_processed += len(batch_repos)

        print(f"\nBatch complete: {len(batch_documents)} documents from {len(batch_repos)} repos")
        print(f"Total progress: {total_processed}/{len(repo_urls)} repos processed")
        print(f"Total documents so far: {len(all_documents)}")

        # For very large datasets, consider incremental training
        if len(all_documents) > 10000 and model is not None:
            print("Performing incremental training on current batch...")
            model.build_vocab(batch_documents, update=True)
            model.train(batch_documents, total_examples=len(batch_documents), epochs=model.epochs)

    print(f"\n{'='*60}")
    print(f"Training final model on {len(all_documents)} total documents...")
    print(f"{'='*60}")

    # Train the model (or create if first time)
    if model is None:
        model = Doc2Vec(
            all_documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            dm=dm,
            workers=cpu_count() or 2,
        )
    else:
        # Final training on all documents
        model.train(all_documents, total_examples=len(all_documents), epochs=epochs//2)

    print("\n✅ Base model training complete")
    return model, all_documents


def save_model_and_metadata(model: Doc2Vec, documents: list, output_path: str, repo_urls: list[str]):
    """Save the model and metadata about training repos."""
    # Save the model
    model.save(output_path)
    print(f"Model saved to {output_path}")

    # Save metadata
    metadata = {
        "training_repos": repo_urls,
        "total_documents": len(documents),
        "vector_size": model.vector_size,
        "window": model.window,
        "min_count": model.min_count,
        "training_epochs": model.epochs,
        "unique_repos": len(set(doc.tags[0].split("/")[0] for doc in documents))
    }

    metadata_path = Path(output_path).with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # Save sample embeddings from base model
    sample_df = export_sample_embeddings(model, documents[:100])  # First 100 docs as sample
    sample_path = Path(output_path).with_suffix(".sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample embeddings saved to {sample_path}")


def export_sample_embeddings(model: Doc2Vec, documents: list) -> pd.DataFrame:
    """Export sample embeddings to CSV."""
    data = []
    for doc in documents:
        vec = model.infer_vector(doc.words)
        data.append([doc.tags[0]] + vec.tolist())

    df = pd.DataFrame(data, columns=["file_path"] + [f"dim_{i}" for i in range(model.vector_size)])
    return df


def load_repo_list(repo_file: str) -> list[str]:
    """Load repository URLs from a file."""
    with open(repo_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train base Doc2Vec model on multiple repositories.")
    parser.add_argument("--repos", help="File containing repository URLs (one per line)")
    parser.add_argument("--repo-urls", nargs="+", help="Repository URLs directly")
    parser.add_argument("--ext", nargs="+", default=[".java"], help="File extensions to include")
    parser.add_argument("--output", default="base_model.d2v", help="Output model file")
    parser.add_argument("--vector-size", type=int, default=200, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of repos to process per batch")
    parser.add_argument("--parallel-workers", type=int, help="Number of parallel workers (default: auto)")
    parser.add_argument("--max-repos", type=int, help="Maximum number of repos to process (for testing)")

    args = parser.parse_args()

    # Get repository list
    if args.repos:
        repo_urls = load_repo_list(args.repos)
    elif args.repo_urls:
        repo_urls = args.repo_urls
    else:
        print("Error: Please provide repository URLs via --repos file or --repo-urls")
        sys.exit(1)

    # Limit repos if requested (useful for testing)
    if args.max_repos:
        repo_urls = repo_urls[:args.max_repos]

    print(f"   Starting base model training")
    print(f"   Repositories: {len(repo_urls)}")
    print(f"   Extensions: {args.ext}")
    print(f"   Vector size: {args.vector_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Output: {args.output}\n")

    start_time = time.time()

    model, documents = train_base_model(
        repo_urls,
        args.ext,
        vector_size=args.vector_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        parallel_workers=args.parallel_workers
    )

    save_model_and_metadata(model, documents, args.output, repo_urls)

    elapsed_time = time.time() - start_time
    print(f"   Base model training pipeline finished successfully!")
    print(f"   Total time: {elapsed_time/60:.1f} minutes")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Model saved to: {args.output}")