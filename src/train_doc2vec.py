#!/usr/bin/env python3
"""
train_doc2vec.py
================
End-to-end Doc2Vec training pipeline for any GitHub repository.

Features:
- Clones a GitHub repo
- Extracts source code files (configurable by extension)
- Tokenizes code
- Trains a Doc2Vec model (Gensim)
- Saves embeddings and model for downstream use

Usage:
  python train_doc2vec.py \
      --repo https://github.com/apache/calcite.git \
      --ext .java \
      --output calcite_embeddings.csv
"""

import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import argparse


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


def prepare_documents(files: list[Path], repo_root: Path) -> list[TaggedDocument]:
    """Prepare TaggedDocument objects for training."""
    documents = []
    for file_path in tqdm(files, desc="Tokenizing files"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                tokens = tokenize_code(f.read())
                if len(tokens) > 5:
                    # Store relative path from repo root instead of absolute path
                    relative_path = file_path.relative_to(repo_root)
                    documents.append(TaggedDocument(words=tokens, tags=[str(relative_path)]))
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
    print(f"📚 Prepared {len(documents)} documents for training")
    return documents


def train_doc2vec(
    documents: list[TaggedDocument],
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 20,
    dm: int = 1,
) -> Doc2Vec:
    """Train a Doc2Vec model using the given documents."""
    print("🚀 Training Doc2Vec model ...")
    model = Doc2Vec(
        documents,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=os.cpu_count() or 2,
    )
    print("✅ Doc2Vec model training complete")
    return model


def export_embeddings(model: Doc2Vec, documents: list[TaggedDocument], output_file: str) -> pd.DataFrame:
    """Infer and export embeddings to CSV."""
    print("💾 Exporting embeddings ...")
    data = []
    for doc in tqdm(documents, desc="Inferring embeddings"):
        vec = model.infer_vector(doc.words)
        data.append([doc.tags[0]] + vec.tolist())

    df = pd.DataFrame(data, columns=["file_path"] + [f"dim_{i}" for i in range(len(data[0]) - 1)])
    df.to_csv(output_file, index=False)
    print(f"✅ Embeddings saved to {output_file}")
    return df


def run_pipeline(github_url: str, extensions: list[str], output_file: str):
    repo_dir = clone_repo(github_url)
    source_files = get_source_files(repo_dir, extensions)
    if not source_files:
        print("❌ No source files found. Check your extensions or repo path.")
        sys.exit(1)

    documents = prepare_documents(source_files, repo_dir)
    model = train_doc2vec(documents)
    export_embeddings(model, documents, output_file)

    model_path = Path(output_file).with_suffix(".model")
    model.save(str(model_path))
    print(f"✅ Model saved as {model_path}")
    print("🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Doc2Vec on a GitHub repository.")
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--ext", nargs="+", default=[".java"], help="File extensions to include (e.g., .py .java)")
    parser.add_argument("--output", default="embeddings.csv", help="Output CSV file for embeddings")

    args = parser.parse_args()
    run_pipeline(args.repo, args.ext, args.output)
