# Doc2Vec Code Embeddings Pipeline

A pipeline for training Doc2Vec models on source code and detecting cross-version duplicate/near-duplicate files and methods. Designed for measuring train/test leakage in Cross-Version Defect Prediction (CVDP) setups.

## Features

- **Two-Stage Training**: Train a base model on popular repos, then fine-tune cumulatively on a target codebase's version history
- **File-Level Cross-Version Analysis**: Clone a repo, extract files per version tag, detect duplicates and train/test leakage (runs in CI)
- **Method-Level Cross-Version Analysis**: Extract method bodies from AST data, detect duplicates and leakage at method granularity (runs locally)
- **Bug Label Enrichment**: Join leakage pairs with SDP labels (buggy/clean) to measure how leakage affects defect prediction
- **Duplicate Classification**: Distinguish **same_file/same_method** (same path across versions) from **collision** (different files/methods with similar embeddings)
- **Deterministic Embeddings**: Uses `model.dv` stored vectors instead of `infer_vector` to avoid randomness

## Quick Start

### GitHub Actions (File-Level)

#### 1. Train Base Model
Go to **Actions** → **"Train Base Doc2Vec Model"**, configure language, repo count, and file extensions.

#### 2. Cross-Version Analysis
Go to **Actions** → **"Cross-Version Duplicate Analysis"** (defaults configured for Django):
- **repo_url**: Target repository
- **tag_regex**: Regex to match version tags (e.g., `^[0-9]+\.[0-9]+$`)
- **base_model_run_id**: Run ID from the base model workflow
- **source_dir**: Subdirectory to restrict file search (e.g., `django`)
- **labels_dir**: Path to bug label CSVs for leakage enrichment

Job summary shows a single table with train/test counts, buggy/clean breakdown, and same_file/collision split per label.

### Local Usage

```bash
pip install -r requirements.txt

# File-level cross-version analysis
python src/cross_version_pipeline.py \
    --repo https://github.com/django/django.git \
    --base-model base_model_python.d2v \
    --tag-regex "^[0-9]+\.[0-9]+$" \
    --ext .py \
    --source-dir django \
    --threshold 0.99

# Method-level analysis (requires pre-extracted AST + source code data)
python src/method_level_pipeline.py \
    --base-model base_model_python.d2v \
    --data-dir "resources/django 1" \
    --output-prefix django_method \
    --threshold 0.99 \
    --epochs 10 \
    --version-prefix django
```

## Project Structure

```
doc2vec-experiments/
├── .github/workflows/
│   ├── train-base-model.yaml         # Train base model on popular repos
│   ├── finetune-model.yaml           # Fine-tune + embed (single version)
│   └── cross-version-analysis.yaml   # File-level cross-version analysis
├── src/
│   ├── train_base_model.py           # Train base model on multiple repos
│   ├── finetune_and_embed.py         # Fine-tune and generate embeddings
│   ├── cross_version_pipeline.py     # File-level cross-version pipeline
│   ├── method_level_pipeline.py      # Method-level cross-version pipeline
│   ├── analyze_duplicates.py         # Duplicate detection (cosine similarity)
│   ├── enrich_leakage.py             # Join leakage with bug labels
│   ├── get_popular_repos.py          # Fetch popular repos from GitHub API
│   └── utils.py                      # Shared utilities
├── resources/
│   ├── django 1/                     # Django SDP data
│   │   ├── file_level/               # Bug labels per version
│   │   ├── ast_level/                # AST trees per version
│   │   ├── source_code/              # Source code zips per version
│   │   └── line_level/               # Buggy lines per version
│   └── calcite/                      # Calcite SDP data
│       └── file_level/               # Bug labels per version
├── requirements.txt
└── README.md
```

## Output Files

### Cross-Version Analysis (File-Level)
- `*_{tag}_embeddings.csv` — Per-version file embeddings
- `*_pair{N}_train_duplicates.csv` — Within-training duplicate pairs
- `*_pair{N}_leakage.csv` — Train-test leakage pairs
- `*_pair{N}_leakage_labeled.csv` — Leakage pairs enriched with buggy/clean labels
- `*_leakage_summary.csv` — Per-pair label breakdown
- `*_cross_version_metadata.json` — Full analysis metadata

### Method-Level Analysis
- `*_{ver}_method_embeddings.csv` — Per-version method embeddings
- `*_pair{N}_method_leakage.csv` — Method-level leakage pairs
- `*_pair{N}_method_leakage_labeled.csv` — Enriched with bug labels from AST data
- `*_method_leakage_summary.csv` — Per-pair label breakdown
- `*_method_level_metadata.json` — Full analysis metadata

## Doc2Vec Configuration

- **Vector size**: 200 dimensions
- **Window size**: 5 tokens
- **Min count**: 3 (minimum word frequency)
- **Training epochs**: 20 (base model), 10 (fine-tuning)
- **Algorithm**: PV-DM (Distributed Memory)
- **Tokenizer**: Simple regex extracting identifiers, lowercased

## Requirements

- Python 3.10+
- Dependencies: `gensim`, `pandas`, `tqdm`, `scikit-learn`, `requests`
- Git (for repository cloning)

## Limitations

- **GitHub API**: Max 1000 repos per search query
- **GitHub Actions**: 6-hour timeout per job
- **Memory**: Sub-batch training splits large document sets into chunks of 5000 to avoid OOM
- **Method-level data**: AST + source code zips are too large for git/CI; run `method_level_pipeline.py` locally

## License

MIT License - see LICENSE file for details