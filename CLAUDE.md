# Doc2Vec Code Embeddings Pipeline

> **Note:** Always update this file when making changes to the project.
> **Commits:** Use `git commit -s` (signoff) and no Co-Authored-By lines.

## Project Overview
A GitHub Actions-powered pipeline for training Doc2Vec models on source code. Uses a two-stage approach: train a base model on popular repositories, then fine-tune on specific codebases.

**Research Context (HRIA):** Part of research comparing duplicate instances in SDP benchmark datasets using different embedding approaches (CodeBERT, T5, Doc2Vec). Doc2Vec has no token limit (unlike CodeBERT's 512 token limit), which may help avoid duplicate embeddings caused by truncation.

## Architecture

### Core Scripts (src/)
- `train_base_model.py` - Train base model on multiple popular GitHub repos
- `finetune_and_embed.py` - Fine-tune pre-trained model and generate embeddings
- `get_popular_repos.py` - Fetch popular repos from GitHub API (supports `--org` for organization filtering)
- `analyze_duplicates.py` - Find duplicate/near-duplicate embeddings using cosine similarity
- `utils.py` - Shared utilities (clone_repo, tokenize_code, prepare_documents)

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
- `.github/workflows/train-base-model.yaml` - Train base model (supports `organization` input for org-specific repos)
- `.github/workflows/finetune-model.yaml` - Fine-tune, embed, analyze duplicates, and post Gist
  - Inputs: `duplicate_threshold` (default 0.99), `post_gist` (default true)
  - Automatically runs duplicate analysis after embedding
  - Posts results to public GitHub Gist (requires `GIST_TOKEN` secret with `gist` scope)

### Constraints
- GitHub API: max 1000 repos per search query
- GitHub Actions: 6-hour workflow timeout
- Memory: Apache repos are huge (e.g., netbeans = 39k files). Swap enabled via `/swapfile2` (runners already have `/swapfile`).
- Gensim models: Doc2Vec saves multiple files (.d2v + .npy), must upload all with `base_model_*`
- `*_repos.txt` files are gitignored (generated output, regenerate as needed)

### Analyzing Duplicates
```bash
# After generating embeddings, find duplicate pairs (similarity >= 0.99)
python src/analyze_duplicates.py \
  --embeddings embeddings.csv \
  --threshold 0.99 \
  --output duplicates_report
```

Output: `duplicates_report_duplicates.csv` (pairs), `duplicates_report_metadata.json` (stats)

## Current Task (HRIA)
- Train Doc2Vec on Apache Java repos (X=100)
- Measure total classes and training time
- Calibrate X to stay within 6-hour Actions limit
- Generate embeddings for SDP benchmark datasets
- Analyze duplicates and compare with CodeBERT results

## Dependencies
Python 3.10+: gensim, pandas, tqdm, scikit-learn, requests
