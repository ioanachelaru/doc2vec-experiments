#!/usr/bin/env python3
"""
utils.py
========
Shared utilities for Doc2Vec training pipelines.
"""

import re
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument


def clone_repo(github_url: str, dest_dir: str = None, version: str = None, shallow: bool = True) -> Path:
    """Clone a GitHub repository and optionally checkout a specific version.

    Args:
        github_url: The GitHub repository URL
        dest_dir: Optional destination directory
        version: Optional commit SHA, tag, or branch to checkout
        shallow: Use shallow clone when no version is specified (default True)

    Returns:
        Path to the cloned repository
    """
    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="repo_")

    # Clone the repository
    if version:
        # Full clone when checking out specific version
        subprocess.run(["git", "clone", github_url, dest_dir], check=True)
        print(f"Repository cloned to: {dest_dir}")

        # Checkout the specific version
        subprocess.run(
            ["git", "checkout", version],
            cwd=dest_dir,
            check=True
        )
        print(f"Checked out version: {version}")
    elif shallow:
        # Shallow clone for latest version (faster)
        subprocess.run(["git", "clone", "--depth", "1", github_url, dest_dir], check=True)
        print(f"Repository cloned to: {dest_dir}")
    else:
        # Full clone without checking out a specific version
        subprocess.run(["git", "clone", github_url, dest_dir], check=True)
        print(f"Repository cloned to: {dest_dir}")

    return Path(dest_dir)


def get_source_files(repo_path: Path, extensions: list[str]) -> list[Path]:
    """Collect source files matching given extensions."""
    files = []
    for ext in extensions:
        files.extend(repo_path.rglob(f"*{ext}"))
    print(f"Found {len(files)} source files ({', '.join(extensions)})")
    return files


def tokenize_code(code: str) -> list[str]:
    """Simple regex-based code tokenizer."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)
    return [t.lower() for t in tokens if len(t) > 1]


def prepare_documents(files: list[Path], repo_root: Path, tag_prefix: str = None) -> list[TaggedDocument]:
    """
    Prepare TaggedDocument objects for training.

    Args:
        files: List of file paths to process
        repo_root: Root directory of the repository
        tag_prefix: Optional prefix for document tags (e.g., repo name)

    Returns:
        List of TaggedDocument objects
    """
    documents = []
    desc = f"Tokenizing {tag_prefix}" if tag_prefix else "Tokenizing files"

    for file_path in tqdm(files, desc=desc):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                tokens = tokenize_code(f.read())
                if len(tokens) > 5:
                    relative_path = file_path.relative_to(repo_root)

                    if tag_prefix:
                        tag = f"{tag_prefix}/{relative_path}"
                    else:
                        tag = str(relative_path)

                    documents.append(TaggedDocument(words=tokens, tags=[tag]))
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Prepared {len(documents)} documents")
    return documents


def get_repo_name_from_url(repo_url: str) -> str:
    """Extract repository name from GitHub URL."""
    return repo_url.split("/")[-1].replace(".git", "")


def get_version_tags(repo_path: str | Path, tag_pattern: str) -> list[str]:
    """List git tags matching a pattern, sorted by version order.

    Uses git's built-in version:refname sorting which correctly handles
    semver-like tags (e.g., 1.9.0 before 1.10.0) and suffixes like -incubating.

    Args:
        repo_path: Path to a cloned git repository (must be a full clone)
        tag_pattern: Glob pattern for tags (e.g., 'calcite-*')

    Returns:
        List of matching tag names sorted by version
    """
    result = subprocess.run(
        ["git", "tag", "--list", tag_pattern, "--sort=version:refname"],
        cwd=str(repo_path),
        capture_output=True, text=True, check=True
    )
    tags = [t.strip() for t in result.stdout.strip().split('\n') if t.strip()]
    print(f"Found {len(tags)} tags matching '{tag_pattern}'")
    return tags


def checkout_version(repo_path: str | Path, version: str):
    """Checkout a specific version and clean the working tree.

    Args:
        repo_path: Path to the git repository
        version: Tag, branch, or commit SHA to checkout
    """
    subprocess.run(
        ["git", "checkout", version],
        cwd=str(repo_path), check=True,
        capture_output=True
    )
    subprocess.run(
        ["git", "clean", "-fdx"],
        cwd=str(repo_path), check=True,
        capture_output=True
    )
    print(f"Checked out version: {version}")