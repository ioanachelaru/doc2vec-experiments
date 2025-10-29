#!/usr/bin/env python3
"""
finetune_and_embed.py
=====================
Fine-tune an existing Doc2Vec model on a new repository and generate embeddings.
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


def prepare_documents(files: list[Path], repo_root: Path) -> list[TaggedDocument]:
    """Prepare TaggedDocument objects for training."""
    documents = []
    for file_path in tqdm(files, desc="Tokenizing files"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                tokens = tokenize_code(f.read())
                if len(tokens) > 5:
                    # Store relative path from repo root
                    relative_path = file_path.relative_to(repo_root)
                    documents.append(TaggedDocument(words=tokens, tags=[str(relative_path)]))
        except Exception as e:
            print(f"⚠️ Skipping {file_path}: {e}")
    print(f"📚 Prepared {len(documents)} documents")
    return documents


def load_base_model(model_path: str) -> Doc2Vec:
    """Load a pre-trained Doc2Vec model."""
    print(f"📂 Loading base model from {model_path}")
    model = Doc2Vec.load(model_path)
    print(f"✅ Model loaded (vector_size={model.vector_size}, vocab_size={len(model.wv)})")
    return model


def finetune_model(
    model: Doc2Vec,
    documents: list[TaggedDocument],
    epochs: int = 10,
    update_vocab: bool = True
) -> Doc2Vec:
    """Fine-tune the model on new documents."""
    print(f"🔧 Fine-tuning model on {len(documents)} new documents...")

    if update_vocab:
        # Build vocabulary from new documents
        print("📖 Updating vocabulary with new documents...")
        model.build_vocab(documents, update=True)

    # Train on new documents
    print(f"🚀 Training for {epochs} epochs...")
    model.train(documents, total_examples=len(documents), epochs=epochs)

    print("✅ Fine-tuning complete")
    return model


def generate_embeddings(model: Doc2Vec, documents: list[TaggedDocument]) -> pd.DataFrame:
    """Generate embeddings for all documents using the fine-tuned model."""
    print("💾 Generating embeddings...")
    data = []

    for doc in tqdm(documents, desc="Generating embeddings"):
        # Infer vector for the document
        vec = model.infer_vector(doc.words, epochs=20)
        data.append([doc.tags[0]] + vec.tolist())

    df = pd.DataFrame(data, columns=["file_path"] + [f"dim_{i}" for i in range(model.vector_size)])
    return df


def save_outputs(
    model: Doc2Vec,
    embeddings_df: pd.DataFrame,
    output_prefix: str,
    repo_url: str,
    base_model_path: str
):
    """Save the fine-tuned model, embeddings, and metadata."""
    # Save embeddings
    embeddings_path = f"{output_prefix}_embeddings.csv"
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f"📊 Embeddings saved to {embeddings_path}")

    # Save fine-tuned model
    model_path = f"{output_prefix}_finetuned.d2v"
    model.save(model_path)
    print(f"💾 Fine-tuned model saved to {model_path}")

    # Save metadata
    metadata = {
        "base_model": base_model_path,
        "finetuned_on": repo_url,
        "total_documents": len(embeddings_df),
        "vector_size": model.vector_size,
        "vocab_size": len(model.wv)
    }

    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📋 Metadata saved to {metadata_path}")


def run_pipeline(
    repo_url: str,
    base_model_path: str,
    extensions: list[str],
    output_prefix: str,
    finetune_epochs: int = 10,
    update_vocab: bool = True
):
    """Run the complete fine-tuning and embedding pipeline."""
    # Load base model
    model = load_base_model(base_model_path)

    # Clone and process repository
    repo_dir = clone_repo(repo_url)
    source_files = get_source_files(repo_dir, extensions)

    if not source_files:
        print("❌ No source files found. Check your extensions or repo path.")
        shutil.rmtree(repo_dir, ignore_errors=True)
        sys.exit(1)

    # Prepare documents
    documents = prepare_documents(source_files, repo_dir)

    # Fine-tune model
    model = finetune_model(model, documents, epochs=finetune_epochs, update_vocab=update_vocab)

    # Generate embeddings
    embeddings_df = generate_embeddings(model, documents)

    # Save outputs
    save_outputs(model, embeddings_df, output_prefix, repo_url, base_model_path)

    # Cleanup
    shutil.rmtree(repo_dir, ignore_errors=True)

    print("🎉 Fine-tuning pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Doc2Vec model and generate embeddings.")
    parser.add_argument("--repo", required=True, help="GitHub repository URL to analyze")
    parser.add_argument("--base-model", required=True, help="Path to base Doc2Vec model")
    parser.add_argument("--ext", nargs="+", default=[".java"], help="File extensions to include")
    parser.add_argument("--output", default="finetuned", help="Output prefix for files")
    parser.add_argument("--epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--no-vocab-update", action="store_true", help="Don't update vocabulary")

    args = parser.parse_args()

    run_pipeline(
        args.repo,
        args.base_model,
        args.ext,
        args.output,
        finetune_epochs=args.epochs,
        update_vocab=not args.no_vocab_update
    )