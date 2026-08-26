#!/usr/bin/env python3
"""
train_base_model.py
====================
Train a base Doc2Vec model on multiple popular GitHub repositories.
This model can later be fine-tuned on specific repositories.
"""

import sys
import shutil
from pathlib import Path
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import argparse
import json
import time
from multiprocessing import cpu_count

from utils import (
    clone_repo,
    get_source_files,
    prepare_documents,
    get_repo_name_from_url,
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
    **_kwargs,
) -> Doc2Vec:
    """Train a base Doc2Vec model on multiple repositories.

    Collects all documents first, then trains in a single Doc2Vec() call
    to avoid gensim's build_vocab(update=True) which segfaults with large
    Java vocabularies.

    Args:
        repo_urls: List of repository URLs to train on
        extensions: File extensions to include
        vector_size: Embedding dimension
        window: Context window size
        min_count: Minimum word frequency
        epochs: Training epochs
        dm: Training algorithm (1=PV-DM, 0=PV-DBOW)
    """
    print(f"\nCollecting documents from {len(repo_urls)} repositories...\n")

    all_documents = []
    sample_documents = []

    for i, repo_url in enumerate(repo_urls, 1):
        print(f"\n[{i}/{len(repo_urls)}] ", end="")
        docs = process_repository((repo_url, extensions))
        all_documents.extend(docs)

        if len(sample_documents) < 100:
            sample_documents.extend(docs[: 100 - len(sample_documents)])

        print(f"  Running total: {len(all_documents)} documents")

    total_document_count = len(all_documents)

    if not all_documents:
        print("Error: No documents collected from any repository")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Training Doc2Vec on {total_document_count} documents...")
    print(f"{'=' * 60}")

    model = Doc2Vec(
        all_documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=cpu_count() or 2,
    )

    del all_documents

    print(f"\n{'=' * 60}")
    print("Base model training complete")
    print(f"   Total documents: {total_document_count}")
    print(f"   Final vocab size: {len(model.wv)}")
    print(f"{'=' * 60}")

    return model, sample_documents, total_document_count


def save_model_and_metadata(
    model: Doc2Vec,
    sample_documents: list,
    total_document_count: int,
    output_path: str,
    repo_urls: list[str],
):
    """Save the model and metadata about training repos."""
    # Save the model
    model.save(output_path)
    print(f"Model saved to {output_path}")

    # Save metadata
    metadata = {
        "training_repos": repo_urls,
        "total_documents": total_document_count,
        "vector_size": model.vector_size,
        "window": model.window,
        "min_count": model.min_count,
        "training_epochs": model.epochs,
        "vocab_size": len(model.wv),
        "unique_repos": len(set(doc.tags[0].split("/")[0] for doc in sample_documents))
        if sample_documents
        else 0,
    }

    metadata_path = Path(output_path).with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # Save sample embeddings from base model
    sample_df = export_sample_embeddings(
        model, sample_documents
    )  # Use saved sample docs
    sample_path = Path(output_path).with_suffix(".sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample embeddings saved to {sample_path}")


def export_sample_embeddings(model: Doc2Vec, documents: list) -> pd.DataFrame:
    """Export sample embeddings to CSV."""
    data = []
    for doc in documents:
        vec = model.infer_vector(doc.words)
        data.append([doc.tags[0]] + vec.tolist())

    df = pd.DataFrame(
        data, columns=["file_path"] + [f"dim_{i}" for i in range(model.vector_size)]
    )
    return df


def load_repo_list(repo_file: str) -> list[str]:
    """Load repository URLs from a file."""
    with open(repo_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train base Doc2Vec model on multiple repositories."
    )
    parser.add_argument(
        "--repos", help="File containing repository URLs (one per line)"
    )
    parser.add_argument("--repo-urls", nargs="+", help="Repository URLs directly")
    parser.add_argument(
        "--ext", nargs="+", default=[".java"], help="File extensions to include"
    )
    parser.add_argument("--output", default="base_model.d2v", help="Output model file")
    parser.add_argument(
        "--vector-size", type=int, default=200, help="Embedding dimension"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument(
        "--max-repos", type=int, help="Maximum number of repos to process (for testing)"
    )

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
        repo_urls = repo_urls[: args.max_repos]

    print("   Starting base model training")
    print(f"   Repositories: {len(repo_urls)}")
    print(f"   Extensions: {args.ext}")
    print(f"   Vector size: {args.vector_size}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Output: {args.output}\n")

    start_time = time.time()

    model, sample_documents, total_document_count = train_base_model(
        repo_urls,
        args.ext,
        vector_size=args.vector_size,
        epochs=args.epochs,
    )

    save_model_and_metadata(
        model, sample_documents, total_document_count, args.output, repo_urls
    )

    elapsed_time = time.time() - start_time
    print("   Base model training pipeline finished successfully!")
    print(f"   Total time: {elapsed_time / 60:.1f} minutes")
    print(f"   Documents processed: {total_document_count}")
    print(f"   Model saved to: {args.output}")
