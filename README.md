# Doc2Vec Code Embeddings Pipeline

A pipeline for training Doc2Vec models on source code and detecting cross-version duplicate/near-duplicate files and methods. Designed for measuring train/test leakage in Cross-Version Defect Prediction (CVDP) setups.

## How It Works

The pipeline uses a two-stage approach:

1. **Train a base model** on popular open-source repositories to learn general code semantics
2. **Fine-tune cumulatively** on a target codebase's version history (versions 0..i = training set, version i+1 = test set), then measure how many test files/methods have near-duplicates in the training set (leakage)

Embeddings are generated using `infer_vector` (200 inference epochs), which computes each vector from the document's actual tokens. This ensures identical code always produces identical vectors, regardless of training history.

> **Why not `model.dv`?** Cumulative training causes gensim to reassign vector slot indices, making unrelated documents share identical vectors. This produced artificial 92-100% leakage in early experiments. See [analysis/django_model_dv_vs_infer_vector.md](analysis/django_model_dv_vs_infer_vector.md) for the full comparison.

## End-to-End Flow

```
                          GitHub Actions
                    ┌──────────────────────┐
                    │  1. Train Base Model  │──→ base_model.d2v (artifact)
                    └──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                  ▼
┌──────────────────────┐            ┌──────────────────────────┐
│ 2a. Cross-Version    │            │ 2b. Fine-tune + Embed    │
│     Analysis (CI)    │            │     Single version (CI)  │
│     File-level       │            │                          │
└──────────────────────┘            └──────────────────────────┘
              │
              ▼                              Local only
┌──────────────────────────┐     ┌──────────────────────────┐
│ Leakage report +         │     │ 2c. Method-Level         │
│ bug label enrichment     │     │     Analysis (local)     │
└──────────────────────────┘     └──────────────────────────┘
```

## GitHub Actions Workflows

All workflows are triggered manually via **Actions > workflow name > Run workflow**.

### 1. Train Base Model (`train-base-model.yaml`)

Trains a Doc2Vec base model on popular GitHub repositories.

| Input | Default | Description |
|-------|---------|-------------|
| `language` | `python` | Programming language (java, python, go, rust, etc.) |
| `organization` | *(empty)* | GitHub org to fetch repos from (e.g., `apache`). Empty = top repos by stars |
| `repo_count` | `100` | Number of repos to train on (max 1000) |
| `file_extensions` | `.py` | Space-separated extensions to include |
| `vector_size` | `200` | Embedding dimensions |
| `epochs` | `20` | Training epochs |
| `min_stars` | `500` | Minimum stars for repo selection (auto-adjusted for large counts) |

**Output artifact:** `base-model-{language}-{count}repos` (`.d2v` model files + metadata JSON). Note the **artifact name** and the **Run ID** (from the URL) -- both are needed for downstream workflows.

### 2a. Cross-Version Analysis (`cross-version-analysis.yaml`)

File-level cross-version duplicate detection + train/test leakage analysis. Clones the target repo, extracts files per version tag, trains cumulatively, and compares embeddings across consecutive version pairs.

| Input | Default | Description |
|-------|---------|-------------|
| `repo_url` | Django | Target repository URL |
| `tag_regex` | `^[0-9]+\.[0-9]+$` | Regex to match version tags |
| `max_versions` | `0` (all) | Limit number of versions to process |
| `base_model_run_id` | `32718528477` | Run ID from step 1 |
| `base_model_artifact` | `base-model-python-100repos` | Artifact name from step 1 |
| `file_extensions` | `.py` | Space-separated extensions |
| `finetune_epochs` | `10` | Epochs for cumulative fine-tuning |
| `update_vocab` | `true` | Update vocabulary with new words from target repo |
| `duplicate_threshold` | `0.99` | Cosine similarity threshold |
| `source_dir` | `django` | Subdirectory to restrict file search (empty = entire repo) |
| `labels_dir` | `resources/django 1/file_level` | Path to bug label CSVs for enrichment (empty = skip) |

**Job summary** shows a single table per version pair: train/test sizes, buggy/clean counts, leaked buggy/clean counts with percentages, and same_file/collision breakdown per label.

**Output artifact:** `cross-version-{repo_name}` containing:
- `*_{tag}_embeddings.csv` -- per-version file embeddings
- `*_pair{N}_train_duplicates.csv` -- within-training-set duplicates
- `*_pair{N}_leakage.csv` -- train/test leakage pairs
- `*_pair{N}_leakage_labeled.csv` -- leakage enriched with buggy/clean labels
- `*_leakage_summary.csv` -- per-pair label breakdown
- `*_cross_version_metadata.json` -- full analysis metadata

### 2b. Fine-tune Model (`finetune-model.yaml`)

Fine-tune a base model on a **single version** of a repository and run duplicate analysis within that version. Useful for quick sanity checks or single-snapshot studies.

| Input | Default | Description |
|-------|---------|-------------|
| `repo_url` | Django | Target repository URL |
| `repo_version` | *(empty)* | Specific tag/branch/SHA to checkout (empty = latest) |
| `base_model_run_id` | *(required)* | Run ID from step 1 |
| `base_model_artifact` | `base-model-python` | Artifact name from step 1 |
| `file_extensions` | `.py` | Space-separated extensions |
| `finetune_epochs` | `10` | Fine-tuning epochs |
| `duplicate_threshold` | `0.99` | Cosine similarity threshold |
| `source_dir` | `django` | Subdirectory to restrict file search |

**Output artifact:** `finetuned-{repo_name}` containing embeddings CSV, fine-tuned model, duplicate pairs, and metadata.

### 2c. Method-Level Analysis (local only)

Runs locally because the AST + source code data is too large for CI (~1.6 GB for Django).

Requires pre-extracted data in `resources/{project}/ast_level/` and `resources/{project}/source_code/`.

```bash
python src/method_level_pipeline.py \
  --base-model base_model_python.d2v \
  --data-dir "resources/django 1" \
  --output-prefix django_method \
  --threshold 0.99 \
  --epochs 10 \
  --version-prefix django
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--base-model` | *(required)* | Path to base Doc2Vec model |
| `--data-dir` | *(required)* | Directory with `ast_level/` and `source_code/` subdirs |
| `--output-prefix` | *(required)* | Prefix for output files |
| `--threshold` | `0.99` | Cosine similarity threshold |
| `--epochs` | `10` | Fine-tuning epochs |
| `--version-prefix` | `django` | Filename pattern: `{prefix}-{ver}.csv` |
| `--max-versions` | `0` (all) | Limit number of versions |

Method tag format: `{version}/{filepath}::{ClassName.method_name}` (with `#N` suffix for disambiguation). Methods with <= 5 tokens are skipped.

**Output:** `*_{ver}_method_embeddings.csv`, `*_pair{N}_method_leakage.csv`, `*_pair{N}_method_leakage_labeled.csv`, `*_method_leakage_summary.csv`, `*_method_level_metadata.json`

## Duplicate Classification

- **same_file / same_method**: Same filepath (or method signature) appearing in different versions -- genuine unchanged code
- **collision**: Different filepaths/methods with cosine similarity >= threshold -- potentially problematic semantic duplicates

## Project Structure

```
doc2vec-experiments/
├── .github/workflows/
│   ├── train-base-model.yaml         # Train base model on popular repos
│   ├── finetune-model.yaml           # Fine-tune + embed + duplicates (single version)
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
│   └── django 1/                     # Django SDP data
│       ├── file_level/               # Bug labels: django-{ver}.csv (filepath, label)
│       ├── ast_level/                # AST trees: django-{ver}.csv (node_type, line_start, line_end, has_bugs)
│       ├── source_code/              # Source code: django-{ver}.zip
│       └── line_level/               # Buggy lines: django-{ver}.csv
├── analysis/                         # Experiment write-ups and comparisons
├── requirements.txt
└── README.md
```

## Doc2Vec Configuration

| Parameter | Base model | Fine-tuning | Inference |
|-----------|-----------|-------------|-----------|
| Algorithm | PV-DM (dm=1) | PV-DM (dm=1) | -- |
| Vector size | 200 | 200 | 200 |
| Window size | 5 | 5 | -- |
| Min count | 3 | 3 | -- |
| Epochs | 20 | 10 | 200 |
| Tokenizer | Regex (identifiers, lowercased) | Same | Same |

## Requirements

- Python 3.10+
- Dependencies: `gensim`, `pandas`, `tqdm`, `scikit-learn`, `requests`
- Git (for repository cloning in CI and cross-version pipeline)

## Limitations

- **GitHub API**: Max 1000 repos per search query
- **GitHub Actions**: 6-hour timeout per job
- **Memory**: Sub-batch training splits large document sets into chunks of 5000 to avoid OOM
- **Method-level data**: AST + source code zips (~1.6 GB for Django) are too large for git/CI; run `method_level_pipeline.py` locally
- **infer_vector**: ~10x slower than model.dv (64 min vs 6 min for Django 26 versions), but produces reliable results

## License

MIT License - see LICENSE file for details
