#!/usr/bin/env python3
"""
train_base_model.py
====================
Train a base Doc2Vec model on multiple popular GitHub repositories.
This model can later be fine-tuned on specific repositories.
"""

import os
import re
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse
import json


def clone_repo(github_url: str, dest_dir: str = None) -> Path:
    """Clone a GitHub repository."""
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="repo_")
    subprocess.run(["git", "clone", "--depth", "1", github_url, dest_dir], check=True)
    print(f"✅ Repository cloned to: {dest_dir}")
    return Path(dest_dir)


def get_source_files(repo_path: Path, extensions: list[str]) -> list[Path]:
    """Collect source files matching given extensions."""
    files = []
    for ext in extensions:
        files.extend(repo_path.rglob(f"*{ext}"))
    print(f"📄 Found {len(files)} source files ({', '.join(extensions)})")
    return files


def tokenize_code(code: str) -> list[str]:
    """Simple regex-based code tokenizer."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
    return [t.lower() for t in tokens if len(t) > 1]


def prepare_documents_from_repo(repo_url: str, files: list[Path], repo_root: Path) -> list[TaggedDocument]:
    """Prepare TaggedDocument objects for training from a single repo."""
    documents = []
    repo_name = repo_url.split("/")[-1].replace(".git", "")

    for file_path in tqdm(files, desc=f"Tokenizing {repo_name}"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                tokens = tokenize_code(f.read())
                if len(tokens) > 5:
                    # Tag includes repo name and relative path
                    relative_path = file_path.relative_to(repo_root)
                    tag = f"{repo_name}/{relative_path}"
                    documents.append(TaggedDocument(words=tokens, tags=[tag]))
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")

    return documents


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
        print(f"\n🔍 Processing repository: {repo_url}")
        repo_dir = clone_repo(repo_url)
        temp_dirs.append(repo_dir)

        source_files = get_source_files(repo_dir, extensions)
        if source_files:
            documents = prepare_documents_from_repo(repo_url, source_files, repo_dir)
            all_documents.extend(documents)
            print(f"📚 Added {len(documents)} documents from {repo_url}")
        else:
            print(f"⚠️ No source files found in {repo_url}")

    print(f"\n🚀 Training base model on {len(all_documents)} total documents...")

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

    print("✅ Base model training complete")
    return model, all_documents


def save_model_and_metadata(model: Doc2Vec, documents: list[TaggedDocument], output_path: str, repo_urls: list[str]):
    """Save the model and metadata about training repos."""
    # Save the model
    model.save(output_path)
    print(f"💾 Model saved to {output_path}")

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
    print(f"📋 Metadata saved to {metadata_path}")

    # Save sample embeddings from base model
    sample_df = export_sample_embeddings(model, documents[:100])  # First 100 docs as sample
    sample_path = Path(output_path).with_suffix(".sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"📊 Sample embeddings saved to {sample_path}")


def export_sample_embeddings(model: Doc2Vec, documents: list[TaggedDocument]) -> pd.DataFrame:
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
        print("❌ Please provide repository URLs via --repos file or --repo-urls")
        sys.exit(1)

    print(f"🎯 Training base model on {len(repo_urls)} repositories")

    model, documents = train_base_model(
        repo_urls,
        args.ext,
        vector_size=args.vector_size,
        epochs=args.epochs
    )

    save_model_and_metadata(model, documents, args.output, repo_urls)
    print("🎉 Base model training pipeline finished successfully!")