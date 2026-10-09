# Doc2Vec Code Embeddings Pipeline

> **Note:** Always update this file when making changes to the project.
> **Commits:** Use `git commit -s` (signoff) and no Co-Authored-By lines.

## Project Overview
A pipeline for training Doc2Vec models on source code and detecting cross-version duplicate/near-duplicate files at file and method granularity. Uses a two-stage approach: train a base model on popular repositories, then fine-tune cumulatively on a target codebase's version history.

**Research Context (HRIA):** Part of research comparing duplicate instances in SDP benchmark datasets using different embedding approaches (CodeBERT, T5, Doc2Vec). Doc2Vec has no token limit (unlike CodeBERT's 512 token limit), which may help avoid duplicate embeddings caused by truncation. The goal is to measure train/test leakage in Cross-Version Defect Prediction (CVDP) setups.

## Architecture

### Core Scripts (src/)
- `train_base_model.py` - Train base model on multiple popular GitHub repos
- `finetune_and_embed.py` - Fine-tune pre-trained model and generate embeddings (single version). Supports `--source-dir` for subdirectory filtering.
- `cross_version_pipeline.py` - **File-level** cross-version analysis: clone repo, extract files per version tag, train cumulatively, detect duplicates + train/test leakage. Supports `--source-dir` for subdirectory filtering. Runs in CI.
- `method_level_pipeline.py` - **Method-level** cross-version analysis: extract method bodies from pre-existing AST-level CSVs and source code zips, train cumulatively, detect duplicates + leakage, enrich with bug labels. Runs locally (data too large for CI).
- `get_popular_repos.py` - Fetch popular repos from GitHub API (supports `--org` for organization filtering)
- `analyze_duplicates.py` - Find duplicate/near-duplicate embeddings (single-version and cross-version). Cosine similarity is clamped to [-1, 1] to prevent floating-point overflow false positives.
- `pairwise_pipeline.py` - **Fresh-base-model strategies**: pairwise (fine-tune on each consecutive pair independently) and cumulative-fresh (growing version window, reset to base each time). Outputs embeddings with version + label columns for downstream ML. Runs in CI.
- `enrich_leakage.py` - Join file-level cross-version leakage pairs with bug labels (buggy/clean) from SDP datasets
- `analyze_leakage_summary.py` - Download CI artifacts and compare embedding-based leakage across strategies (pairwise, cumulative-fresh, cumulative-carried) and projects (Django, Calcite). Produces per-pair comparison tables and aggregate summaries.
- `analyze_embedding_distances.py` - Compute cosine similarity distributions for same-path files across consecutive versions (no threshold cutoff). For each strategy and project, matches files by relative path between version pairs and computes per-file cosine similarity. Outputs per-pair stats (mean/median/min/max, pct above 0.99/0.95/0.90) and per-file detail CSVs for distribution plotting.
- `plot_embedding_distances.py` - Generate publication-quality visualizations from the similarity CSVs: KDE distributions, per-pair trend lines, threshold exceedance bars, and aggregate summaries.
- `predict_with_leakage.py` - **Supervised classification experiments**: train RF + LR on Doc2Vec embeddings, evaluate on baseline/cleaned/leaked-only subsets to measure how train/test overlap inflates defect prediction. Supports embedding-based leakage (cosine sim >= threshold), same-code ground truth comparison (from team ZIP archives), and threshold sweep (0.90–1.00). Handles Django float version precision issues (1.10→1.1).
- `utils.py` - Shared utilities (clone_repo, tokenize_code, prepare_documents, get_version_tags)

### Key Patterns
- Doc2Vec with PV-DM algorithm (dm=1)
- Default: 200-dim vectors, window=5, min_count=3, 20 epochs
- Simple regex tokenizer: extracts identifiers, lowercased
- Parallel processing with multiprocessing.Pool for batch repo handling
- Embeddings via `infer_vector` (200 inference epochs) — computes vectors from actual document tokens, so identical code always produces identical vectors
- Cumulative training: versions are trained one at a time in semver order; embeddings are inferred after all training completes
- Fresh-base strategies: pairwise (fresh model per pair) and cumulative-fresh (fresh model, growing window) — both reset to base model each iteration, unlike cumulative-carried. Both train one version at a time per `finetune_model()` call (~700 docs) to avoid gensim SEGV with large corpora. Cumulative-fresh supports `--start-iter`/`--end-iter` for parallel CI execution (5 chunk jobs)
- Phase-split CI execution: `--phase tokenize` (lightweight process, no gensim/numpy/sklearn imports) clones, discovers versions, tokenizes via subprocesses, saves state to `--cache-dir`. `--phase train` (fresh process) loads cached tokens and model, trains, generates embeddings. Prevents heap fragmentation from accumulating across tokenization and model loading in a single process.
- All heavy imports (gensim, pandas, sklearn, numpy) are lazy — imported inside strategy functions, not at module level — so the tokenize phase stays under ~50MB RSS
- Duplicate classification: **same_file/same_method** (same filepath/method across versions) vs **collision** (different files/methods with similar embeddings)

### Data Layout
```
resources/django 1/
  file_level/         # SDP labels: django-{ver}.csv with filepath, label
  ast_level/          # Full AST trees: django-{ver}.csv with node_type, node_info, line_start, line_end, has_bugs
  source_code/        # Source code zips: django-{ver}.zip
  line_level/         # Buggy lines: django-{ver}.csv with filepath, line_number, line_content
```

AST CSVs may contain duplicate rows (~2.5x); always deduplicate on `(filepath, node_id)` before use.

## Development Notes

### Fetching Repos from Specific Organization (e.g., Apache)
```bash
python src/get_popular_repos.py --org apache --language java --count 100 --output apache_repos.txt
python src/train_base_model.py --repos apache_repos.txt --ext .java --output apache_base_model.d2v
```

### Running Locally
```bash
pip install -r requirements.txt

# General file-level pipeline
python src/get_popular_repos.py --language java --count 100 --output popular_repos.txt
python src/train_base_model.py --repos popular_repos.txt --ext .java --output base_model.d2v
python src/finetune_and_embed.py --repo <url> --base-model base_model.d2v --ext .java

# Method-level pipeline (requires pre-extracted AST + source code data)
python src/method_level_pipeline.py \
  --base-model base_model_python.d2v \
  --data-dir "resources/django 1" \
  --output-prefix django_method \
  --threshold 0.99 \
  --epochs 10 \
  --version-prefix django
```

### GitHub Actions Workflows
- `.github/workflows/train-base-model.yaml` - Train base model (supports `organization` input, job summary on completion)
- `.github/workflows/finetune-model.yaml` - Fine-tune, embed, analyze duplicates (single version)
  - Inputs: `duplicate_threshold` (default 0.99), `source_dir` (optional subdirectory filter)
  - Automatically runs duplicate analysis after embedding
  - Results displayed on job summary page
- `.github/workflows/cross-version-analysis.yaml` - File-level cross-version duplicate analysis
  - Defaults configured for Django (repo, tag regex, base model, labels)
  - Inputs: `repo_url`, `tag_regex`, `max_versions`, `duplicate_threshold`, `source_dir`, `base_model_run_id`, `labels_dir`
  - Trains cumulatively, generates embeddings via `infer_vector` (200 epochs)
  - If `labels_dir` is provided, enriches leakage with buggy/clean labels via `enrich_leakage.py`
  - Job summary: single table showing train/test counts, buggy/clean breakdown, leaked counts with same_file/collision split per label
  - Output: `*_embeddings.csv` (per version), `*_train_duplicates.csv`, `*_leakage.csv`, `*_leakage_labeled.csv`, `*_leakage_summary.csv`
- `.github/workflows/embedding-strategies.yaml` - Pairwise and cumulative-fresh embedding strategies
  - Inputs: `strategy` (pairwise or cumulative-fresh), `project` (django or calcite), `max_versions`, etc.
  - Each iteration loads a fresh base model (not carried forward)
  - **Pairwise**: single job (~2.5 h)
  - **Cumulative-fresh**: 5 parallel chunk jobs (iterations 1-5, 6-10, 11-15, 16-20, 21-30) + merge job for combined summary. Each chunk runs in two phases (`--phase tokenize` then `--phase train`) as separate Python processes to avoid heap fragmentation on 7GB runners. Chunks beyond the iteration count exit early with no results.
  - Output embeddings include `version` and `label` columns for downstream ML
  - Output: `*_pair{N}_embeddings.csv` or `*_iter{N}_embeddings.csv`, `*_duplicates.csv`, metadata JSON

### Constraints
- GitHub API: max 1000 repos per search query
- GitHub Actions: 6-hour timeout per job
- Memory: Phase-split execution (`--phase tokenize`/`--phase train`) prevents heap fragmentation; disk-backed corpus streams documents from pickle files; subprocess-per-version tokenization ensures OS reclaims all memory per version.
- Gensim models: Doc2Vec saves multiple files (.d2v + .npy), must upload all with `base_model_*`
- `*_repos.txt` files are gitignored (generated output, regenerate as needed)
- Method-level data (AST + source zips) is too large for git/CI artifacts; run `method_level_pipeline.py` locally

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

Output: `*_duplicates.csv` (pairs with similarity and duplicate_type), `*_metadata.json` (stats)

### Cross-Version Analysis (File-Level)
```bash
# Java example (Apache Calcite)
python src/cross_version_pipeline.py \
  --repo https://github.com/apache/calcite.git \
  --base-model base_model.d2v \
  --tag-regex "calcite-1\.([0-9]|1[0-5])\.0(-incubating)?$" \
  --ext .java \
  --threshold 0.99

# Python example (Django, restricted to django/ subdirectory)
python src/cross_version_pipeline.py \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --tag-regex "^[0-9]+\.[0-9]+$" \
  --ext .py \
  --source-dir django \
  --threshold 0.99
```

Output per version: `*_{tag}_embeddings.csv`
Output per pair: `*_{tagA}_vs_{tagB}_duplicates.csv`, `*_pair{N}_train_duplicates.csv`, `*_pair{N}_leakage.csv`
Metadata: `*_cross_version_metadata.json` (includes leakage stats per pair)

### Method-Level Analysis
```bash
# Requires pre-extracted data in resources/<project>/ast_level/ and source_code/
python src/method_level_pipeline.py \
  --base-model base_model_python.d2v \
  --data-dir "resources/django 1" \
  --output-prefix django_method \
  --threshold 0.99 \
  --epochs 10 \
  --version-prefix django \
  --max-versions 5
```

Method tag format: `{version}/{filepath}::{ClassName.method_name}` (with `#N` disambiguation for collisions). Methods with <= 5 tokens are skipped.

Output: `*_{ver}_method_embeddings.csv`, `*_pair{N}_method_leakage.csv`, `*_pair{N}_method_leakage_labeled.csv`, `*_method_leakage_summary.csv`, `*_method_level_metadata.json`

### Pairwise / Cumulative-Fresh Strategies
```bash
# Pairwise: fresh base model per consecutive pair, embeddings with labels
python src/pairwise_pipeline.py \
  --strategy pairwise \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --tag-regex "^[0-9]+\.[0-9]+$" \
  --ext .py \
  --source-dir django \
  --labels-dir "resources/django 1/file_level" \
  --output django_pairwise

# Cumulative-fresh: fresh base model, growing version window
python src/pairwise_pipeline.py \
  --strategy cumulative-fresh \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --tag-regex "^[0-9]+\.[0-9]+$" \
  --ext .py \
  --source-dir django \
  --labels-dir "resources/django 1/file_level" \
  --output django_cumfresh

# Parallel chunks (used by CI — run iterations 1-5 only)
python src/pairwise_pipeline.py \
  --strategy cumulative-fresh \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --tag-regex "^[0-9]+\.[0-9]+$" \
  --ext .py \
  --source-dir django \
  --labels-dir "resources/django 1/file_level" \
  --output django_cumfresh \
  --start-iter 1 --end-iter 5
```

Output: `*_pair{N}_embeddings.csv` or `*_iter{N}_embeddings.csv` (with version + label columns), `*_pair{N}_duplicates.csv` or `*_iter{N}_leakage.csv`, `*_{strategy}_metadata.json` (or `*_metadata_chunk{start}_{end}.json` when using `--start-iter`/`--end-iter`)

### Enriching Leakage with Bug Labels
```bash
# File-level: join leakage pairs with SDP label CSVs
python src/enrich_leakage.py \
  --metadata django_cross_version_metadata.json \
  --labels-dir "resources/django 1/file_level" \
  --output-prefix django
```

Input: `*_pair{N}_leakage.csv` + `django-{version}.csv` label files
Output: `*_pair{N}_leakage_labeled.csv` (enriched with `label_a`, `label_b`, `same_label`), `*_leakage_summary.csv`

Method-level enrichment is built into `method_level_pipeline.py` (uses `has_bugs` from AST data directly, no separate labels needed).

### Comparing Leakage Across Strategies
```bash
# Download all CI artifacts and produce comparison tables
python src/analyze_leakage_summary.py --download

# Re-run analysis on already-downloaded artifacts
python src/analyze_leakage_summary.py --results-dir results
```

Output: `results/embedding_leakage_comparison_django.csv`, `results/embedding_leakage_comparison_calcite.csv`, `results/embedding_leakage_summary.csv`

### Embedding Similarity Distributions
```bash
# Full analysis: all strategies and projects
python src/analyze_embedding_distances.py --results-dir results

# Single strategy/project
python src/analyze_embedding_distances.py --strategy pairwise --project django
```

For every file that exists in both train and test (same relative path across consecutive versions), computes the cosine similarity between their Doc2Vec embeddings — no threshold cutoff. Extracts version from the file path prefix (not the version column) to handle versions like 1.10 that lose precision as float.

Output: `results/embedding_distances_{project}_{strategy}.csv` (per-file similarities for histogram plotting), `results/embedding_distance_summary.csv` (aggregate stats)

### Plotting Similarity Distributions
```bash
# Generate all plots (PNG, 150 DPI)
python src/plot_embedding_distances.py

# High-DPI PDF for paper
python src/plot_embedding_distances.py --format pdf --dpi 300
```

Reads the per-file similarity CSVs and generates 7 figures: 2 KDE distribution plots, 2 per-pair mean similarity trend lines, 2 threshold exceedance bar charts (% >= 0.99), and 1 aggregate summary.

Output: `results/plots/*.png` (or `*.pdf`)

### Supervised Classification (Leakage Impact)
```bash
# Pairwise with embedding-based and same-code leakage
python src/predict_with_leakage.py --project django --strategy pairwise \
  --same-code-zip "resources/django 1/django-same-code.zip"
python src/predict_with_leakage.py --project calcite --strategy pairwise \
  --same-code-zip "resources/calcite/calcite-same-code.zip"

# Cumulative-fresh (embedding-based only)
python src/predict_with_leakage.py --project django --strategy cumulative-fresh
python src/predict_with_leakage.py --project calcite --strategy cumulative-fresh

# Threshold sweep (0.90-1.00)
python src/predict_with_leakage.py --project django --strategy pairwise --sweep
python src/predict_with_leakage.py --project calcite --strategy pairwise --sweep
```

Trains RF(500 trees) + LR with `class_weight='balanced'` on embedding vectors. Evaluates on three subsets: baseline (full test), cleaned (non-leaked files), leaked-only. Delta(baseline - cleaned) = metric inflation from leakage. Same-code ground truth from team ZIP archives (source-code-based duplicate detection) provides an alternative leaked set for comparison.

Output: `results/prediction/{project}_{strategy}_predictions.csv` (per-pair metrics), `*_summary.csv` (aggregate means), `*_threshold_sweep.csv` (sweep data)

## Current Task (HRIA)
- File-level cross-version analysis done for Django (26 versions) and Calcite (16 versions)
- Method-level cross-version analysis for Django (run locally with `method_level_pipeline.py`)
- Compare file-level vs method-level leakage rates
- Compare Doc2Vec results with CodeBERT findings
- ICAART paper deadline: Oct 22

## Dependencies
Python 3.10+: gensim, pandas, tqdm, scikit-learn, requests
