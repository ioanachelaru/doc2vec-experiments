#!/usr/bin/env python3
"""
analyze_duplicates.py
=====================
Analyze embeddings to find duplicate/near-duplicate pairs based on cosine similarity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
from pathlib import Path


def load_embeddings(csv_path: str) -> tuple[list[str], np.ndarray]:
    """Load embeddings CSV and return file paths + vectors.

    Args:
        csv_path: Path to embeddings CSV file

    Returns:
        Tuple of (file_paths list, vectors numpy array)
    """
    print(f"Loading embeddings from {csv_path}...")
    df = pd.read_csv(csv_path)
    file_paths = df['file_path'].tolist()
    vectors = df.drop('file_path', axis=1).values
    print(f"Loaded {len(file_paths)} embeddings with {vectors.shape[1]} dimensions")
    return file_paths, vectors


def find_duplicates(file_paths: list[str], vectors: np.ndarray, threshold: float = 0.99) -> list[dict]:
    """Find pairs of embeddings with similarity >= threshold.

    Args:
        file_paths: List of file paths corresponding to each embedding
        vectors: numpy array of embedding vectors
        threshold: Minimum cosine similarity to consider as duplicate

    Returns:
        List of duplicate pairs with their similarity scores
    """
    print(f"Computing cosine similarity matrix for {len(file_paths)} embeddings...")
    sim_matrix = cosine_similarity(vectors)

    print(f"Finding duplicate pairs with similarity >= {threshold}...")
    duplicates = []
    n = len(file_paths)

    # Only check upper triangle (avoid duplicate pairs and self-comparison)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                duplicates.append({
                    'file_a': file_paths[i],
                    'file_b': file_paths[j],
                    'similarity': float(sim_matrix[i, j])
                })

    # Sort by similarity descending
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)

    return duplicates


def generate_report(
    duplicates: list[dict],
    total_files: int,
    threshold: float,
    output_prefix: str
):
    """Generate duplicate analysis report.

    Args:
        duplicates: List of duplicate pairs
        total_files: Total number of files analyzed
        threshold: Similarity threshold used
        output_prefix: Prefix for output files
    """
    # Calculate statistics
    unique_files_in_duplicates = set()
    exact_duplicates = 0

    for d in duplicates:
        unique_files_in_duplicates.add(d['file_a'])
        unique_files_in_duplicates.add(d['file_b'])
        if d['similarity'] >= 0.9999:  # Effectively 1.0
            exact_duplicates += 1

    stats = {
        'total_files': total_files,
        'duplicate_pairs': len(duplicates),
        'exact_duplicate_pairs': exact_duplicates,
        'files_with_duplicates': len(unique_files_in_duplicates),
        'percentage_files_with_duplicates': round(len(unique_files_in_duplicates) / total_files * 100, 2) if total_files > 0 else 0,
        'threshold': threshold
    }

    # Print summary to console
    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS REPORT")
    print("=" * 60)
    print(f"Total files analyzed:        {stats['total_files']}")
    print(f"Duplicate pairs found:       {stats['duplicate_pairs']}")
    print(f"  - Exact duplicates (1.0):  {stats['exact_duplicate_pairs']}")
    print(f"Files involved in duplicates: {stats['files_with_duplicates']}")
    print(f"Percentage with duplicates:  {stats['percentage_files_with_duplicates']}%")
    print(f"Similarity threshold:        {stats['threshold']}")
    print("=" * 60)

    # Save duplicates to CSV
    if duplicates:
        duplicates_df = pd.DataFrame(duplicates)
        csv_path = f"{output_prefix}_duplicates.csv"
        duplicates_df.to_csv(csv_path, index=False)
        print(f"\nDuplicate pairs saved to: {csv_path}")

        # Show top 10 duplicates
        print("\nTop 10 duplicate pairs:")
        for i, d in enumerate(duplicates[:10]):
            print(f"  {i+1}. {d['file_a']}")
            print(f"     {d['file_b']}")
            print(f"     Similarity: {d['similarity']:.6f}")
    else:
        print("\nNo duplicate pairs found!")

    # Save metadata JSON
    metadata_path = f"{output_prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze embeddings for duplicate/near-duplicate pairs."
    )
    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to embeddings CSV file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity to consider as duplicate (default: 0.99)"
    )
    parser.add_argument(
        "--output",
        default="duplicates_report",
        help="Output prefix for report files"
    )

    args = parser.parse_args()

    # Load embeddings
    file_paths, vectors = load_embeddings(args.embeddings)

    # Find duplicates
    duplicates = find_duplicates(file_paths, vectors, args.threshold)

    # Generate report
    generate_report(duplicates, len(file_paths), args.threshold, args.output)

    print("\nDuplicate analysis complete!")


if __name__ == "__main__":
    main()
