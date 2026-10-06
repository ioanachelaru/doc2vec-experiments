#!/usr/bin/env python3
"""
compare_dimensions.py
=====================
Compare leakage results between 200-dim and 400-dim Doc2Vec embeddings.

Downloads artifacts from GitHub Actions if not already present, extracts
leakage statistics from metadata JSONs and comparison CSVs, and produces
a summary table.

Usage:
    # Download new artifacts and compare
    python src/compare_dimensions.py --download

    # Compare using already-downloaded artifacts
    python src/compare_dimensions.py \
        --dim200-django /tmp/dim200-django \
        --dim400-calcite /tmp/dim400-calcite
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# Workflow run IDs for each base model
RUN_IDS = {
    "django": {
        200: 37302006634,  # embedding-strategies run with dim200 base
        400: None,  # original runs, data in local results/
    },
    "calcite": {
        200: None,  # original runs, data in local results/
        400: 37302039751,  # embedding-strategies run with dim400 base
    },
}

# Artifact names per strategy
ARTIFACT_NAMES = {
    "pairwise": "pairwise-{project}-dim{dim}",
    "cumulative-fresh": "cumulative-fresh-{project}-dim{dim}",
}


def load_metadata_leakage(metadata_path: Path) -> list[dict]:
    """Extract per-pair leakage stats from a pipeline metadata JSON."""
    with open(metadata_path) as f:
        meta = json.load(f)

    pairs = []
    for r in meta["results"]:
        pairs.append(
            {
                "pair": r.get("pair", r.get("iteration")),
                "version_a": r.get("version_a", ", ".join(r.get("train_versions", []))),
                "version_b": r.get("version_b", r.get("test_version", "")),
                "test_leakage_pct": r["test_leakage_pct"],
                "same_file_pairs": r["same_file_pairs"],
                "collision_pairs": r["collision_pairs"],
            }
        )
    return pairs


def load_existing_leakage(comparison_csv: Path, strategy: str) -> dict[int, dict]:
    """Load per-pair leakage from the existing comparison CSV."""
    col_prefix = strategy.replace("-", "_")
    results = {}

    with open(comparison_csv) as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)

    for row in reader:
        pair = int(row["pair"])
        results[pair] = {
            "pair": pair,
            "version_a": row["train_version"],
            "version_b": row["test_version"],
            "test_leakage_pct": float(row[f"{col_prefix}_leakage_pct"]),
            "same_file_pairs": int(row[f"{col_prefix}_same_file"]),
            "collision_pairs": int(row[f"{col_prefix}_collision"]),
        }
    return results


def print_comparison(
    project: str,
    strategy: str,
    dim_a: int,
    data_a: dict[int, dict],
    dim_b: int,
    data_b: dict[int, dict],
) -> None:
    """Print per-pair comparison table."""
    print(f"\n{'=' * 70}")
    print(f"{project.upper()} — {strategy} — dim{dim_a} vs dim{dim_b}")
    print(f"{'=' * 70}")
    print(
        f"{'Pair':<5} {'Train':<22} {'Test':<22} "
        f"{'dim' + str(dim_a):<10} {'dim' + str(dim_b):<10} {'Delta':>8}"
    )
    print("-" * 80)

    all_pairs = sorted(set(data_a.keys()) | set(data_b.keys()))
    sum_a, sum_b, count = 0.0, 0.0, 0

    for p in all_pairs:
        a = data_a.get(p, {})
        b = data_b.get(p, {})
        pct_a = a.get("test_leakage_pct", 0)
        pct_b = b.get("test_leakage_pct", 0)
        delta = pct_b - pct_a
        va = a.get("version_a", b.get("version_a", "?"))
        vb = a.get("version_b", b.get("version_b", "?"))

        print(
            f"{p:<5} {str(va):<22} {str(vb):<22} "
            f"{pct_a:<10.2f} {pct_b:<10.2f} {delta:>+8.2f}"
        )
        sum_a += pct_a
        sum_b += pct_b
        count += 1

    if count > 0:
        avg_a = sum_a / count
        avg_b = sum_b / count
        print("-" * 80)
        print(
            f"{'AVG':<5} {'':<22} {'':<22} "
            f"{avg_a:<10.2f} {avg_b:<10.2f} {avg_b - avg_a:>+8.2f}"
        )

    # Totals
    total_same_a = sum(d.get("same_file_pairs", 0) for d in data_a.values())
    total_same_b = sum(d.get("same_file_pairs", 0) for d in data_b.values())
    total_coll_a = sum(d.get("collision_pairs", 0) for d in data_a.values())
    total_coll_b = sum(d.get("collision_pairs", 0) for d in data_b.values())
    print(
        f"\n  Total same-file pairs: dim{dim_a}={total_same_a}, "
        f"dim{dim_b}={total_same_b}"
    )
    print(
        f"  Total collision pairs: dim{dim_a}={total_coll_a}, dim{dim_b}={total_coll_b}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare leakage between 200-dim and 400-dim embeddings."
    )
    parser.add_argument(
        "--dim200-django",
        type=str,
        help="Directory with dim200 Django artifacts (pairwise/ and cumfresh/ subdirs)",
    )
    parser.add_argument(
        "--dim400-calcite",
        type=str,
        help="Directory with dim400 Calcite artifacts (pairwise/ and cumfresh/ subdirs)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory with existing comparison CSVs (default: results/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download artifacts from GitHub Actions before comparing",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Resolve artifact directories
    if args.download:
        import subprocess

        for project, dim, run_id in [
            ("django", 200, RUN_IDS["django"][200]),
            ("calcite", 400, RUN_IDS["calcite"][400]),
        ]:
            for strategy in ["pairwise", "cumulative-fresh"]:
                artifact = ARTIFACT_NAMES[strategy].format(project=project, dim=dim)
                dest = Path(f"/tmp/dim{dim}-{project}/{strategy.split('-')[0]}")
                if strategy == "cumulative-fresh":
                    dest = Path(f"/tmp/dim{dim}-{project}/cumfresh")

                if dest.exists() and any(dest.iterdir()):
                    print(f"  {artifact} already downloaded to {dest}")
                    continue

                dest.mkdir(parents=True, exist_ok=True)
                print(f"  Downloading {artifact} from run {run_id}...")
                subprocess.run(
                    [
                        "gh",
                        "run",
                        "download",
                        str(run_id),
                        "-n",
                        artifact,
                        "-D",
                        str(dest),
                    ],
                    check=True,
                )
        args.dim200_django = "/tmp/dim200-django"
        args.dim400_calcite = "/tmp/dim400-calcite"

    if not args.dim200_django or not args.dim400_calcite:
        print("Error: provide --dim200-django and --dim400-calcite, or use --download")
        sys.exit(1)

    dim200_django = Path(args.dim200_django)
    dim400_calcite = Path(args.dim400_calcite)

    # ── Django: dim400 (existing) vs dim200 (new) ──

    django_comparison = results_dir / "embedding_leakage_comparison_django.csv"

    for strategy, subdir, meta_pattern in [
        ("pairwise", "pairwise", "*_pairwise_metadata.json"),
        ("cumulative-fresh", "cumfresh", "*_cumulative_fresh_metadata.json"),
    ]:
        # dim400 from existing comparison CSV
        existing = load_existing_leakage(django_comparison, strategy)

        # dim200 from new metadata
        meta_dir = dim200_django / subdir
        meta_files = list(meta_dir.glob(meta_pattern))
        # Use the non-chunk metadata if available
        meta_files = [f for f in meta_files if "chunk" not in f.name] or meta_files
        if not meta_files:
            print(f"Warning: no metadata found in {meta_dir} for {strategy}")
            continue

        new_pairs = load_metadata_leakage(meta_files[0])
        new_data = {p["pair"]: p for p in new_pairs}

        print_comparison("Django", strategy, 400, existing, 200, new_data)

    # ── Calcite: dim200 (existing) vs dim400 (new) ──

    calcite_comparison = results_dir / "embedding_leakage_comparison_calcite.csv"

    for strategy, subdir, meta_pattern in [
        ("pairwise", "pairwise", "*_pairwise_metadata.json"),
        ("cumulative-fresh", "cumfresh", "*_cumulative_fresh_metadata.json"),
    ]:
        existing = load_existing_leakage(calcite_comparison, strategy)

        meta_dir = dim400_calcite / subdir
        meta_files = list(meta_dir.glob(meta_pattern))
        meta_files = [f for f in meta_files if "chunk" not in f.name] or meta_files
        if not meta_files:
            print(f"Warning: no metadata found in {meta_dir} for {strategy}")
            continue

        new_pairs = load_metadata_leakage(meta_files[0])
        new_data = {p["pair"]: p for p in new_pairs}

        print_comparison("Calcite", strategy, 200, existing, 400, new_data)

    # ── Summary ──

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'Project':<10} {'Strategy':<18} {'Dim':<5} {'Avg Leak%':<12} "
        f"{'Same-file':<12} {'Collision':<10}"
    )
    print("-" * 70)

    for strategy in ["pairwise", "cumulative-fresh"]:
        # Django dim400
        d400 = load_existing_leakage(django_comparison, strategy)
        avg = sum(v["test_leakage_pct"] for v in d400.values()) / len(d400)
        sf = sum(v["same_file_pairs"] for v in d400.values())
        co = sum(v["collision_pairs"] for v in d400.values())
        print(
            f"{'Django':<10} {strategy:<18} {'400':<5} {avg:<12.2f} {sf:<12} {co:<10}"
        )

        # Django dim200
        subdir = "pairwise" if strategy == "pairwise" else "cumfresh"
        pat = (
            "*_pairwise_metadata.json"
            if strategy == "pairwise"
            else "*_cumulative_fresh_metadata.json"
        )
        mf = [f for f in (dim200_django / subdir).glob(pat) if "chunk" not in f.name]
        if mf:
            pairs = load_metadata_leakage(mf[0])
            avg = sum(p["test_leakage_pct"] for p in pairs) / len(pairs)
            sf = sum(p["same_file_pairs"] for p in pairs)
            co = sum(p["collision_pairs"] for p in pairs)
            print(
                f"{'Django':<10} {strategy:<18} {'200':<5} {avg:<12.2f} {sf:<12} {co:<10}"
            )

    for strategy in ["pairwise", "cumulative-fresh"]:
        # Calcite dim200
        c200 = load_existing_leakage(calcite_comparison, strategy)
        avg = sum(v["test_leakage_pct"] for v in c200.values()) / len(c200)
        sf = sum(v["same_file_pairs"] for v in c200.values())
        co = sum(v["collision_pairs"] for v in c200.values())
        print(
            f"{'Calcite':<10} {strategy:<18} {'200':<5} {avg:<12.2f} {sf:<12} {co:<10}"
        )

        # Calcite dim400
        subdir = "pairwise" if strategy == "pairwise" else "cumfresh"
        pat = (
            "*_pairwise_metadata.json"
            if strategy == "pairwise"
            else "*_cumulative_fresh_metadata.json"
        )
        mf = [f for f in (dim400_calcite / subdir).glob(pat) if "chunk" not in f.name]
        if mf:
            pairs = load_metadata_leakage(mf[0])
            avg = sum(p["test_leakage_pct"] for p in pairs) / len(pairs)
            sf = sum(p["same_file_pairs"] for p in pairs)
            co = sum(p["collision_pairs"] for p in pairs)
            print(
                f"{'Calcite':<10} {strategy:<18} {'400':<5} {avg:<12.2f} {sf:<12} {co:<10}"
            )


if __name__ == "__main__":
    main()
