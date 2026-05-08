# Doc2Vec Code Embeddings Pipeline

> **Note:** Always update this file when making changes to the project.
> **Commits:** Use `git commit -s` (signoff) and no Co-Authored-By lines.

## Project Overview
A GitHub Actions-powered pipeline for training Doc2Vec models on source code. Uses a two-stage approach: train a base model on popular repositories, then fine-tune on specific codebases.

**Research Context (HRIA):** Part of research comparing duplicate instances in SDP benchmark datasets using different embedding approaches (CodeBERT, T5, Doc2Vec). Doc2Vec has no token limit (unlike CodeBERT's 512 token limit), which may help avoid duplicate embeddings caused by truncation.

## Architecture

### Core Scripts (src/)
- `train_base_model.py` - Train base model on multiple popular GitHub repos
- `finetune_and_embed.py` - Fine-tune pre-trained model and generate embeddings (single version)
- `cross_version_pipeline.py` - Fine-tune on first version, embed all versions, analyze cross-version duplicates
- `get_popular_repos.py` - Fetch popular repos from GitHub API (supports `--org` for organization filtering)
- `analyze_duplicates.py` - Find duplicate/near-duplicate embeddings (single-version and cross-version)
- `utils.py` - Shared utilities (clone_repo, tokenize_code, prepare_documents, get_version_tags)

### Key Patterns
- Doc2Vec with PV-DM algorithm (dm=1)
- Default: 200-dim vectors, window=5, min_count=3, 20 epochs
- Simple regex tokenizer: extracts identifiers, lowercased
- Parallel processing with multiprocessing.Pool for batch repo handling
- Incremental training for large datasets (>10k documents)

## Development Notes

### Fetching Repos from Specific Organization (e.g., Apache)
```bash
# Fetch top 100 Java repos from Apache organization
python src/get_popular_repos.py --org apache --language java --count 100 --output apache_repos.txt

# Train base model on Apache repos
python src/train_base_model.py --repos apache_repos.txt --ext .java --output apache_base_model.d2v
```

### Running Locally (General)
```bash
pip install -r requirements.txt
python src/get_popular_repos.py --language java --count 100 --output popular_repos.txt
python src/train_base_model.py --repos popular_repos.txt --ext .java --output base_model.d2v
python src/finetune_and_embed.py --repo <url> --base-model base_model.d2v --ext .java
```

### GitHub Actions Workflows
- `.github/workflows/train-base-model.yaml` - Train base model (supports `organization` input, job summary on completion)
- `.github/workflows/finetune-model.yaml` - Fine-tune, embed, analyze duplicates (single version)
  - Inputs: `duplicate_threshold` (default 0.99)
  - Automatically runs duplicate analysis after embedding
  - Results displayed on job summary page
- `.github/workflows/cross-version-analysis.yaml` - Cross-version duplicate analysis
  - Inputs: `repo_url`, `tag_pattern` (e.g., `calcite-*`), `max_versions`, `duplicate_threshold`
  - Fine-tunes on first version, embeds all versions, analyzes consecutive pairs + overall
  - Results displayed on job summary page with per-version and per-pair breakdowns

### Constraints
- GitHub API: max 1000 repos per search query
- GitHub Actions: 6-hour workflow timeout
- Memory: Uses sub-batch training - splits large document sets into chunks of `--max-docs-per-batch 5000` to avoid OOM.
- Gensim models: Doc2Vec saves multiple files (.d2v + .npy), must upload all with `base_model_*`
- `*_repos.txt` files are gitignored (generated output, regenerate as needed)

### Analyzing Duplicates
```bash
# Single-version: find duplicate pairs within one embeddings file
python src/analyze_duplicates.py \
  --embeddings embeddings.csv \
  --threshold 0.99 \
  --output duplicates_report

# Cross-version: compare two embeddings files
python src/analyze_duplicates.py \
  --embeddings-a v1_embeddings.csv \
  --embeddings-b v2_embeddings.csv \
  --threshold 0.99 \
  --output v1_vs_v2_report
```

Output: `*_duplicates.csv` (pairs), `*_metadata.json` (stats)

### Cross-Version Analysis
```bash
# Full pipeline: fine-tune on first tag, embed all versions, analyze duplicates
python src/cross_version_pipeline.py \
  --repo https://github.com/apache/calcite.git \
  --base-model base_model.d2v \
  --tag-pattern "calcite-*" \
  --ext .java \
  --threshold 0.99 \
  --max-versions 5
```

Output per version: `*_{tag}_embeddings.csv`
Output per pair: `*_{tagA}_vs_{tagB}_duplicates.csv`
Output overall: `*_all_versions_duplicates.csv`, `*_cross_version_metadata.json`

## Current Task (HRIA)
- Train Doc2Vec on Apache Java repos (X=100)
- Measure total classes and training time
- Calibrate X to stay within 6-hour Actions limit
- Generate embeddings for SDP benchmark datasets
- Analyze duplicates and compare with CodeBERT results

## Dependencies
Python 3.10+: gensim, pandas, tqdm, scikit-learn, requests
