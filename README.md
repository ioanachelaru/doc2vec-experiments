# Doc2Vec Code Embeddings Pipeline

A pipeline for training Doc2Vec models on source code and detecting cross-version duplicate/near-duplicate files and methods. Built to measure **train/test leakage** in Cross-Version Defect Prediction (CVDP) setups, where the presence of identical or near-identical code across versions can silently inflate classifier performance.

## Table of Contents

- [Research Motivation](#research-motivation)
- [How It Works](#how-it-works)
  - [Two-Stage Training](#two-stage-training)
  - [Tokenization](#tokenization)
  - [Cumulative Training](#cumulative-training)
  - [Embedding Generation](#embedding-generation)
  - [Duplicate Detection](#duplicate-detection)
  - [Leakage Computation](#leakage-computation)
  - [Bug Label Enrichment](#bug-label-enrichment)
- [Architecture](#architecture)
  - [Pipeline Overview](#pipeline-overview)
  - [Scripts and Responsibilities](#scripts-and-responsibilities)
  - [Data Flow](#data-flow)
- [File-Level vs Method-Level Analysis](#file-level-vs-method-level-analysis)
- [Data Formats](#data-formats)
  - [Input Data](#input-data)
  - [Output Data](#output-data)
- [Usage](#usage)
  - [Local Setup](#local-setup)
  - [GitHub Actions Workflows](#github-actions-workflows)
- [Doc2Vec Configuration](#doc2vec-configuration)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

---

## Research Motivation

In Cross-Version Defect Prediction (CVDP), a model is trained on historical versions of a software project and tested on a future version. A known but often-ignored problem is **data leakage**: files or methods that exist unchanged across versions appear in both the training and test sets, inflating prediction accuracy without the model learning anything meaningful.

This pipeline quantifies that leakage using code embeddings. By converting source code files (or methods) into fixed-size vectors and comparing them with cosine similarity, we can identify which test entries have near-duplicates in the training set, and whether those duplicates are buggy or clean.

This work is part of the HRIA study, which compares duplicate detection approaches (CodeBERT, T5, Doc2Vec) across SDP benchmark datasets. Doc2Vec has no token limit (unlike CodeBERT's 512), which may help avoid false duplicates caused by input truncation.

## How It Works

### Two-Stage Training

The pipeline uses a transfer learning approach:

1. **Base model training**: A Doc2Vec model is trained on ~100 popular open-source repositories to learn general code semantics (identifier patterns, common structures). This gives the model a broad vocabulary and understanding of code before it ever sees the target project.

2. **Cumulative fine-tuning**: The base model is then fine-tuned on the target project's version history. Versions are processed in chronological order, and each version's documents are used to update the model. This mirrors the CVDP setup where versions 0..i form the training set and version i+1 is the test set.

### Tokenization

Source code is tokenized with a simple regex-based approach (`utils.py:tokenize_code`):

```
Input:  "def calculate_total(self, items: list[int]) -> int:"
Tokens: ["def", "calculate", "total", "self", "items", "list", "int", "int"]
```

The tokenizer:
- Extracts identifiers matching `[A-Za-z_][A-Za-z_0-9]*`
- Converts everything to lowercase
- Drops single-character tokens (e.g., `x`, `i`)
- Ignores literals, operators, and punctuation

Files (or methods) with <= 5 tokens after tokenization are skipped -- they carry too little semantic information to produce meaningful embeddings.

### Cumulative Training

For cross-version analysis, the model is trained on one version at a time in semver order:

```
Version 1.0 documents --> train model
Version 1.1 documents --> continue training (vocabulary updated)
Version 1.2 documents --> continue training (vocabulary updated)
...
```

At each version, the model calls `build_vocab(documents, update=True)` to incorporate new identifiers, then `train()` to update weights. After all versions are trained, embeddings are generated for every document.

This approach has two important properties:
- The model's vocabulary grows monotonically -- it never forgets words from earlier versions
- The model sees the same code evolve over time, which is exactly the scenario in CVDP

### Embedding Generation

Embeddings are generated using `model.infer_vector(document_tokens, epochs=200)`, which computes a vector from the document's actual tokens by gradient descent against the trained word vectors.

> **Why not `model.dv`?** Gensim's built-in document vectors (`model.dv`) are a lookup table indexed by position. During cumulative training, gensim reassigns slot indices when new documents are added, causing unrelated documents to share identical vectors. This produced artificial 92-100% leakage in early experiments -- every buggy file appeared as "leaked" because it shared a vector with a completely different file. See [analysis/django_model_dv_vs_infer_vector.md](analysis/django_model_dv_vs_infer_vector.md) for the full comparison.

Key properties of `infer_vector`:
- **Content-deterministic**: Identical code always produces identical vectors, regardless of training history
- **High epoch count** (200): Ensures convergence; lower counts produce noisy, unstable vectors
- **Trade-off**: ~10x slower than `model.dv` (64 min vs 6 min for Django 26 versions), but results are reliable

### Duplicate Detection

Two embeddings are considered duplicates if their cosine similarity >= a threshold (default 0.99). The similarity matrix is computed using scikit-learn's `cosine_similarity`, with values clamped to [-1, 1] to prevent floating-point overflow from producing false positives.

Each duplicate pair is classified as one of:

| Type | Meaning | Example |
|------|---------|---------|
| **same_file** | Same relative path in different versions | `1.0/django/utils/text.py` vs `1.1/django/utils/text.py` |
| **collision** | Different paths with nearly identical embeddings | `1.0/django/forms/utils.py` vs `1.1/django/core/validators.py` |

At method level, the same logic applies but uses **same_method** (same `filepath::ClassName.method_name` across versions) and **collision** (different methods with similar embeddings).

Classification works by stripping the version prefix from each tag and comparing the remaining path:
```
"1.0/django/utils/text.py" --> strip "1.0/" --> "django/utils/text.py"
"1.1/django/utils/text.py" --> strip "1.1/" --> "django/utils/text.py"
Paths match --> same_file
```

### Leakage Computation

At each consecutive version boundary (vN, vN+1), the pipeline defines:
- **Training set**: all documents from versions v1 through vN
- **Test set**: all documents from version vN+1

A test entry has **leakage** if it has at least one near-duplicate (cosine similarity >= threshold) in the training set. The pipeline computes:

1. **Within-training duplicates**: pairs of near-duplicates that both appear in the training set (across any combination of training versions)
2. **Train-test leakage**: test entries that have near-duplicates in the training set, broken down by same_file vs collision

The leakage percentage is: `(test files with at least one near-duplicate in train) / (total test files) * 100`

To compute this efficiently, the pipeline first calculates all pairwise cross-version duplicates (every version pair, not just consecutive), then assembles the training and test sets for each boundary from these pre-computed results.

### Bug Label Enrichment

The raw leakage data tells you *which* test files are leaked, but not *whether it matters for defect prediction*. The enrichment step joins leakage pairs with SDP bug labels to answer: are the leaked test files buggy or clean?

**File-level** (`enrich_leakage.py`): reads per-version label CSVs (e.g., `resources/django 1/file_level/1.0.csv`) where each row maps a filepath to "buggy" or "clean". Joins labels to both sides of each leakage pair.

**Method-level** (built into `method_level_pipeline.py`): uses the `has_bugs` column directly from the AST data, since each method node already carries its bug label.

The enrichment produces:
- Per-pair labeled CSVs (`*_leakage_labeled.csv`) with `label_a`, `label_b`, `same_label` columns
- A summary CSV (`*_leakage_summary.csv`) showing the breakdown per pair: how many buggy/clean test entries are leaked, what percentage of buggy vs clean files are affected, and the split by same_file vs collision

## Architecture

### Pipeline Overview

```
                          GitHub Actions (CI)
                    +--------------------------+
                    |  1. Train Base Model     |--> base_model.d2v (artifact)
                    |     get_popular_repos.py |
                    |     train_base_model.py  |
                    +--------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
+---------------------------+     +---------------------------+
| 2a. Cross-Version         |     | 2b. Fine-tune + Embed     |
|     Analysis (CI)         |     |     Single version (CI)   |
|     cross_version_        |     |     finetune_and_embed.py |
|     pipeline.py           |     |     analyze_duplicates.py |
|     analyze_duplicates.py |     +---------------------------+
|     enrich_leakage.py     |
+---------------------------+
              |
              v                            Local only
+---------------------------+     +---------------------------+
| File-level leakage report |     | 2c. Method-Level          |
| with bug label breakdown  |     |     Analysis (local)      |
+---------------------------+     |     method_level_         |
                                  |     pipeline.py           |
                                  +---------------------------+
```

### Scripts and Responsibilities

| Script | Purpose | Used by |
|--------|---------|---------|
| `get_popular_repos.py` | Fetches top GitHub repos by language/stars/org via the Search API. Outputs a text file with one clone URL per line. | Base model training |
| `train_base_model.py` | Clones each repo, tokenizes source files, and trains a single Doc2Vec model on all documents. Saves model + metadata JSON + sample embeddings CSV. | CI workflow |
| `finetune_and_embed.py` | Loads a base model, clones a target repo, fine-tunes, and generates embeddings. Also contains `generate_embeddings_infer()` and `generate_embeddings_from_docvecs()` used by other scripts. | CI workflow, imported by cross-version and method-level pipelines |
| `cross_version_pipeline.py` | Orchestrates file-level cross-version analysis: clone repo, discover version tags, train cumulatively, generate embeddings via `infer_vector`, compute all pairwise duplicates, and compute leakage stats at each boundary. | CI workflow |
| `method_level_pipeline.py` | Same as cross-version but at method granularity. Extracts method bodies from pre-existing AST CSVs and source code zips instead of cloning a repo. Includes built-in bug label enrichment. | Local only |
| `analyze_duplicates.py` | Core duplicate detection: computes cosine similarity matrices and finds pairs above threshold. Used as both a library (imported by pipelines) and a standalone CLI. | All pipelines |
| `enrich_leakage.py` | Joins file-level leakage pairs with SDP bug labels. Reads metadata JSON to find pairs, loads label CSVs, and produces labeled leakage CSVs + summary. | CI workflow (after cross-version pipeline) |
| `utils.py` | Shared utilities: `clone_repo`, `get_source_files`, `tokenize_code`, `prepare_documents`, `get_version_tags`, `checkout_version`. | All scripts |

### Data Flow

```
                       TRAINING PHASE
                       ==============

Popular repos (GitHub API)          Target repo (git clone)
        |                                    |
        v                                    v
  Clone & tokenize                   For each version tag:
  all source files                     checkout --> tokenize
        |                                    |
        v                                    v
  Doc2Vec(all_documents)             build_vocab(update=True)
  = base model                       train(version_docs)
        |                                    |
        v                                    v
  base_model.d2v                     Cumulative model (in memory)


                     EMBEDDING PHASE
                     ===============

  For each version's documents:
    model.infer_vector(doc.words, epochs=200)
        |
        v
    {version}_embeddings.csv
    (file_path, dim_0, dim_1, ..., dim_199)


                     ANALYSIS PHASE
                     ==============

  Version embeddings (all pairs)
        |
        v
  cosine_similarity(vectors_a, vectors_b)
        |
        v
  Filter pairs where similarity >= 0.99
  Classify as same_file or collision
        |
        +--> Within-training duplicates
        +--> Cross-version leakage pairs
        +--> Leakage statistics per boundary
        |
        v (optional)
  Join with bug labels (buggy/clean)
        |
        v
  Summary: leaked buggy %, leaked clean %,
           same_file vs collision breakdown
```

## File-Level vs Method-Level Analysis

The pipeline supports two granularity levels:

| Aspect | File-Level | Method-Level |
|--------|-----------|--------------|
| **Unit of analysis** | Entire source file | Individual function/method body |
| **Data source** | Git clone (checkout each version tag) | Pre-extracted AST CSVs + source code zips |
| **Tag format** | `{version}/{relative_path}` | `{version}/{filepath}::{Class.method}` |
| **Duplicate type labels** | `same_file` / `collision` | `same_file` / `collision` (same_file = same method signature) |
| **Bug labels source** | External SDP label CSVs (filepath + label) | AST data's `has_bugs` column per method node |
| **Runs on** | GitHub Actions (CI) | Local only (data too large for CI) |
| **Method extraction** | N/A | Reads AST CSV for FunctionDef nodes, resolves parent ClassDef for qualified names, extracts body from source zip using line_start/line_end |

Method-level analysis adds additional processing:
- **Class name resolution**: Each `FunctionDef` node in the AST has a `parent_id` pointing to its enclosing `ClassDef` (if any). The pipeline resolves this to produce qualified names like `ModelForm.save` instead of just `save`.
- **Tag disambiguation**: If multiple methods have the same qualified name in the same file (e.g., overloaded or nested), a `#N` suffix is appended (`save#1`, `save#2`).
- **AST deduplication**: AST CSVs can contain ~2.5x duplicate rows; the pipeline deduplicates on `(filepath, node_id)` before processing.

## Data Formats

### Input Data

**Base model training** requires a text file with one repo clone URL per line:
```
https://github.com/django/django.git
https://github.com/pallets/flask.git
...
```

**File-level cross-version analysis** requires only a git repository URL and a tag regex. Source files are extracted directly from the cloned repo.

**Method-level analysis** requires pre-extracted data in `resources/{project}/`:

```
resources/django 1/
  ast_level/          # AST trees per version
  |  django-1.0.csv   # Columns: version, filepath, node_id, edge_id, parent_id,
  |  django-1.1.csv   #   sibling_index, edge_type, node_type, node_info,
  |  ...               #   line_start, line_end, col_offset, end_col_offset, has_bugs
  |
  source_code/        # Source code archives per version
  |  django-1.0.zip   # Contains source files at their original paths
  |  django-1.1.zip
  |  ...
  |
  file_level/         # SDP bug labels per version
  |  1.0.csv           # Columns: version, filepath, label (buggy/clean)
  |  1.1.csv
  |  ...
  |
  line_level/         # Buggy line data per version (not used by this pipeline)
     django-1.0.csv    # Columns: filepath, line_number, line_content
     ...
```

### Output Data

All outputs use a common prefix (e.g., `django` or `django_method`):

| File | Contents |
|------|----------|
| `{prefix}_{version}_embeddings.csv` | Per-version embeddings: `file_path, dim_0, dim_1, ..., dim_199` |
| `{prefix}_{vA}_vs_{vB}_duplicates.csv` | Duplicate pairs between two versions: `file_a, file_b, similarity, duplicate_type` |
| `{prefix}_pair{N}_train_duplicates.csv` | Within-training-set duplicates for boundary N |
| `{prefix}_pair{N}_leakage.csv` | Train-test leakage pairs for boundary N |
| `{prefix}_pair{N}_leakage_labeled.csv` | Leakage pairs enriched with `label_a, label_b, same_label` |
| `{prefix}_leakage_summary.csv` | Per-pair summary: test counts, leaked counts, buggy/clean breakdown, same_file/collision split |
| `{prefix}_cross_version_metadata.json` | Full analysis metadata: config, per-version file counts, per-pair results, timing |
| `{prefix}_finetuned.d2v` | The cumulatively fine-tuned Doc2Vec model |

## Usage

### Local Setup

```bash
pip install -r requirements.txt
```

**File-level cross-version analysis:**
```bash
# 1. Get popular repos for base model training
python src/get_popular_repos.py --language python --count 100 --output popular_repos.txt

# 2. Train base model
python src/train_base_model.py --repos popular_repos.txt --ext .py --output base_model_python.d2v

# 3. Run cross-version analysis (e.g., Django)
python src/cross_version_pipeline.py \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --tag-regex "^[0-9]+\.[0-9]+$" \
  --ext .py \
  --source-dir django \
  --threshold 0.99

# 4. Enrich leakage with bug labels (optional)
python src/enrich_leakage.py \
  --metadata django_cross_version_metadata.json \
  --labels-dir "resources/django 1/file_level" \
  --output-prefix django
```

**Method-level analysis (requires pre-extracted data):**
```bash
python src/method_level_pipeline.py \
  --base-model base_model_python.d2v \
  --data-dir "resources/django 1" \
  --output-prefix django_method \
  --threshold 0.99 \
  --epochs 10 \
  --version-prefix django
```

**Single-version duplicate analysis:**
```bash
# Fine-tune on a single repo version and check for internal duplicates
python src/finetune_and_embed.py \
  --repo https://github.com/django/django.git \
  --base-model base_model_python.d2v \
  --ext .py \
  --source-dir django

# Or analyze an existing embeddings file
python src/analyze_duplicates.py \
  --embeddings django_embeddings.csv \
  --threshold 0.99 \
  --output django_report
```

### GitHub Actions Workflows

All workflows are triggered manually via **Actions > workflow name > Run workflow**.

#### 1. Train Base Model (`train-base-model.yaml`)

Trains a Doc2Vec base model on popular GitHub repositories.

| Input | Default | Description |
|-------|---------|-------------|
| `language` | `python` | Programming language |
| `organization` | *(empty)* | GitHub org to filter (e.g., `apache`). Empty = top repos by stars |
| `repo_count` | `100` | Number of repos (max 1000) |
| `file_extensions` | `.py` | Space-separated extensions |
| `vector_size` | `200` | Embedding dimensions |
| `epochs` | `20` | Training epochs |

**Output artifact:** `base-model-{language}-{count}repos` (`.d2v` model files + metadata JSON). Note the **artifact name** and the **Run ID** (from the URL) -- both are needed for downstream workflows.

#### 2. Cross-Version Analysis (`cross-version-analysis.yaml`)

File-level cross-version duplicate detection + train/test leakage analysis.

| Input | Default | Description |
|-------|---------|-------------|
| `repo_url` | Django | Target repository URL |
| `tag_regex` | `^[0-9]+\.[0-9]+$` | Regex to match version tags |
| `max_versions` | `0` (all) | Limit number of versions |
| `base_model_run_id` | `32718528477` | Run ID from step 1 |
| `base_model_artifact` | `base-model-python-100repos` | Artifact name from step 1 |
| `file_extensions` | `.py` | Space-separated extensions |
| `finetune_epochs` | `10` | Epochs for cumulative fine-tuning |
| `duplicate_threshold` | `0.99` | Cosine similarity threshold |
| `source_dir` | `django` | Subdirectory filter (empty = entire repo) |
| `labels_dir` | `resources/django 1/file_level` | Bug label CSVs for enrichment (empty = skip) |

**Job summary** shows a table per version pair with train/test sizes, buggy/clean counts, leaked counts with percentages, and same_file/collision breakdown per label.

#### 3. Fine-tune Model (`finetune-model.yaml`)

Fine-tune a base model on a single version and run duplicate analysis within that version.

| Input | Default | Description |
|-------|---------|-------------|
| `repo_url` | Django | Target repository URL |
| `repo_version` | *(empty)* | Tag/branch/SHA to checkout (empty = latest) |
| `base_model_run_id` | *(required)* | Run ID from step 1 |
| `base_model_artifact` | `base-model-python` | Artifact name from step 1 |
| `file_extensions` | `.py` | Space-separated extensions |
| `duplicate_threshold` | `0.99` | Cosine similarity threshold |

## Doc2Vec Configuration

| Parameter | Base model | Fine-tuning | Inference |
|-----------|-----------|-------------|-----------|
| Algorithm | PV-DM (dm=1) | PV-DM (dm=1) | -- |
| Vector size | 200 | 200 | 200 |
| Window size | 5 | 5 | -- |
| Min count | 3 | 3 | -- |
| Epochs | 20 | 10 | 200 |
| Tokenizer | Regex (identifiers, lowercased) | Same | Same |

**PV-DM** (Paragraph Vector - Distributed Memory) jointly trains word and document vectors by predicting a target word from its context window plus the document vector. This captures word order and local context, which is relevant for code where identifier co-occurrence patterns are meaningful.

## Project Structure

```
doc2vec-experiments/
+-- .github/workflows/
|   +-- train-base-model.yaml         # CI: train base model on popular repos
|   +-- finetune-model.yaml           # CI: fine-tune + embed + duplicates (single version)
|   +-- cross-version-analysis.yaml   # CI: file-level cross-version analysis
+-- src/
|   +-- get_popular_repos.py          # Fetch popular repos from GitHub API
|   +-- train_base_model.py           # Train base model on multiple repos
|   +-- finetune_and_embed.py         # Fine-tune and generate embeddings
|   +-- cross_version_pipeline.py     # File-level cross-version pipeline
|   +-- method_level_pipeline.py      # Method-level cross-version pipeline (local)
|   +-- analyze_duplicates.py         # Duplicate detection (cosine similarity)
|   +-- enrich_leakage.py             # Join leakage with bug labels
|   +-- utils.py                      # Shared utilities
+-- resources/
|   +-- django 1/                     # Django SDP benchmark data
|   |   +-- file_level/               # Bug labels per version (filepath, label)
|   |   +-- ast_level/                # AST trees per version (node_type, lines, has_bugs)
|   |   +-- source_code/              # Source code zips per version
|   |   +-- line_level/               # Buggy lines per version
|   +-- calcite/
|       +-- file_level/               # Calcite bug labels
+-- analysis/                         # Experiment write-ups and comparisons
+-- requirements.txt
+-- README.md
```

## Limitations

- **GitHub API**: Max 1000 repos per search query
- **GitHub Actions**: 6-hour timeout per job
- **Memory**: Sub-batch training splits large document sets into chunks of 5000 to avoid OOM
- **Method-level data**: AST + source code zips (~1.6 GB for Django) are too large for git/CI; run `method_level_pipeline.py` locally
- **infer_vector performance**: ~10x slower than model.dv (64 min vs 6 min for Django 26 versions), but produces reliable results
- **Duplicate detection complexity**: O(n^2) pairwise cosine similarity computation. For version pairs with ~800 files each, the similarity matrix is 800x800 which is fast, but all-pairs across 26 versions produces 325 matrices
- **Gensim model files**: Doc2Vec saves multiple files (`.d2v` + `.npy`); all must be uploaded/downloaded together

## License

MIT License - see LICENSE file for details
