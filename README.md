# Doc2Vec Code Embeddings Pipeline

A GitHub Actions-powered pipeline for training Doc2Vec models on source code from any GitHub repository. Automatically analyze popular repositories or specify custom ones to generate code embeddings for machine learning applications.

## Features

- **Automated Training**: GitHub Actions workflow for hands-free Doc2Vec training
- **Popular Repos Discovery**: Automatically fetch and analyze the most popular GitHub repositories by language
- **Multi-Repository Support**: Process multiple repositories in a single workflow run
- **Language Agnostic**: Support for any programming language (Java, Python, JavaScript, etc.)
- **Code Tokenization**: Smart tokenization of source code for better embeddings
- **Export Options**: Generate both CSV embeddings and trained Doc2Vec models

## Quick Start

### Via GitHub Actions (Recommended)

1. Fork this repository
2. Go to the **Actions** tab
3. Select **"Train Doc2Vec Model"**
4. Click **"Run workflow"**
5. Choose your options:
   - **Repository source**: `popular` (auto-fetch) or `custom` (your URLs)
   - **Language**: `java`, `python`, `javascript`, etc.
   - **Number of repos**: How many popular repos to analyze (if using popular mode)
   - **File extensions**: `.java`, `.py`, `.js`, etc.

### Local Usage

```bash
# Clone the repo
git clone https://github.com/yourusername/doc2vec-experiments.git
cd doc2vec-experiments

# Install dependencies
pip install gensim pandas tqdm scikit-learn requests

# Train on a single repository
python src/train_doc2vec.py \
    --repo https://github.com/apache/spark.git \
    --ext .java .scala \
    --output spark_embeddings.csv

# Get popular repositories
python src/get_popular_repos.py \
    --language python \
    --count 10 \
    --output popular_python_repos.txt
```

## Workflow Options

### Analyzing Popular Repositories

The workflow can automatically fetch the most popular repositories from GitHub:

- **Languages supported**: java, python, javascript, go, rust, typescript, etc.
- **Ranking criteria**: Star count, recent activity (last 5 years)
- **Default**: Top 10 repositories with 1000+ stars

Example: To analyze the top 10 Python projects:
1. Repository source: `popular`
2. Language: `python`
3. Number of repos: `10`
4. File extensions: `.py`

### Custom Repository List

Provide your own list of repositories to analyze:

1. Repository source: `custom`
2. Enter URLs (one per line):
   ```
   https://github.com/tensorflow/tensorflow.git
   https://github.com/pytorch/pytorch.git
   https://github.com/scikit-learn/scikit-learn.git
   ```

## Output Files

For each repository, the pipeline generates:

- **`{repo_name}_embeddings.csv`**: Document vectors for each source file
  - Column 1: Repository-relative file path (e.g., `src/main/java/MyClass.java`)
  - Columns 2-201: 200-dimensional embedding vectors
- **`{repo_name}_embeddings.model`**: Trained Doc2Vec model for inference

## Doc2Vec Configuration

Default training parameters:
- **Vector size**: 200 dimensions
- **Window size**: 5 tokens
- **Min count**: 3 (minimum word frequency)
- **Training epochs**: 20
- **Model**: DM (Distributed Memory)

## Use Cases

- **Code similarity**: Find similar code files across projects
- **Code search**: Semantic search through codebases
- **Code classification**: Classify code by functionality or quality
- **Technical debt analysis**: Identify problematic code patterns
- **Cross-project analysis**: Compare coding styles across repositories

## Project Structure

```
doc2vec-experiments/
├── .github/
│   └── workflows/
│       └── doc2vec-train.yaml    # GitHub Actions workflow
├── src/
│   ├── train_doc2vec.py          # Main training pipeline
│   └── get_popular_repos.py      # Fetch popular repos from GitHub
└── README.md
```

## Requirements

- Python 3.10+
- Dependencies: `gensim`, `pandas`, `tqdm`, `scikit-learn`, `requests`

## Contributing

Contributions are welcome! Feel free to:
- Add support for more languages
- Improve tokenization strategies
- Add visualization tools
- Enhance the training pipeline

## License

MIT License - see LICENSE file for details