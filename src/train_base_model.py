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

from utils import (
    clone_repo,
    get_source_files,
    prepare_documents,
    get_repo_name_from_url
)


def train_base_model(
    repo_urls: list[str],
    extensions: list[str],
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 20,
    dm: int = 1,
) -> Doc2Vec:
    """Train a base Doc2Vec model on multiple repositories."""

    all_documents = []
    temp_dirs = []

    # Process each repository
    for repo_url in repo_urls:
        print(f"\nProcessing repository: {repo_url}")
        repo_dir = clone_repo(repo_url)
        temp_dirs.append(repo_dir)

        source_files = get_source_files(repo_dir, extensions)
        if source_files:
            repo_name = get_repo_name_from_url(repo_url)
            documents = prepare_documents(source_files, repo_dir, tag_prefix=repo_name)
            all_documents.extend(documents)
            print(f"Added {len(documents)} documents from {repo_url}")
        else:
            print(f"Warning: No source files found in {repo_url}")

    print(f"\nTraining base model on {len(all_documents)} total documents...")

    # Train the model
    model = Doc2Vec(
        all_documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=os.cpu_count() or 2,
    )

    # Cleanup temp directories
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("Base model training complete")
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

    args = parser.parse_args()

    # Get repository list
    if args.repos:
        repo_urls = load_repo_list(args.repos)
    elif args.repo_urls:
        repo_urls = args.repo_urls
    else:
        print("Error: Please provide repository URLs via --repos file or --repo-urls")
        sys.exit(1)

    print(f"Training base model on {len(repo_urls)} repositories")

    model, documents = train_base_model(
        repo_urls,
        args.ext,
        vector_size=args.vector_size,
        epochs=args.epochs
    )

    save_model_and_metadata(model, documents, args.output, repo_urls)
    print("Base model training pipeline finished successfully!")