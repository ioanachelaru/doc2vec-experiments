# Doc2Vec Code Embeddings Pipeline

A GitHub Actions-powered pipeline for training Doc2Vec models on source code. Features both base model training on popular repositories and fine-tuning capabilities for specific codebases.

## Features

- **Two-Stage Training**: Train a base model on popular repos, then fine-tune on your specific codebase
- **Base Model Training**: Build robust representations from top GitHub repositories
- **Fine-Tuning Pipeline**: Adapt pre-trained models to your specific repository
- **Popular Repos Discovery**: Automatically fetch the most popular repositories by language
- **Language Agnostic**: Support for any programming language (Java, Python, JavaScript, etc.)
- **Smart Embeddings**: Generate embeddings using models trained on relevant codebases

## Quick Start

### Two-Stage Approach

#### Step 1: Train Base Model on Popular Repositories

1. Go to **Actions** → **"Train Base Doc2Vec Model"**
2. Configure:
   - **Language**: Select the programming language (java, python, etc.)
   - **Repository count**: Number of top repos to use (default: 10)
   - **File extensions**: Extensions to analyze (e.g., `.java`)
   - **Vector size**: Embedding dimensions (default: 200)
3. Run workflow and wait for completion
4. Note the artifact name (e.g., `base-model-java`)

#### Step 2: Fine-tune on Your Repository

1. Go to **Actions** → **"Fine-tune Doc2Vec Model"**
2. Configure:
   - **Repository URL**: Your target repository
   - **Base model artifact**: Name from Step 1 (e.g., `base-model-java`)
   - **File extensions**: Extensions to analyze
   - **Fine-tune epochs**: Training iterations (default: 10)
3. Run workflow to generate embeddings for your repository


### Local Usage

```bash
# Clone the repo
git clone https://github.com/yourusername/doc2vec-experiments.git
cd doc2vec-experiments

# Install dependencies
pip install -r requirements.txt

# Step 1: Get popular repositories
python src/get_popular_repos.py \
    --language java \
    --count 10 \
    --output popular_repos.txt

# Step 2: Train base model on popular repos
python src/train_base_model.py \
    --repos popular_repos.txt \
    --ext .java \
    --output base_model.d2v \
    --vector-size 200 \
    --epochs 20

# Step 3: Fine-tune on your repository
python src/finetune_and_embed.py \
    --repo https://github.com/your-org/your-repo.git \
    --base-model base_model.d2v \
    --ext .java \
    --output your_repo \
    --epochs 10
```

## Workflows Available

### 1. Train Base Model (`train-base-model.yaml`)
- Trains a Doc2Vec model on multiple popular repositories
- Creates a robust base representation for code
- Outputs: base model, metadata, sample embeddings

### 2. Fine-tune Model (`finetune-model.yaml`)
- Takes a pre-trained base model
- Fine-tunes it on your specific repository
- Generates embeddings optimized for your codebase
- Outputs: embeddings CSV, fine-tuned model, metadata

## Output Files

### Base Model Training
- **`base_model_{language}.d2v`**: Trained Doc2Vec model
- **`base_model_{language}.json`**: Training metadata (repos used, parameters)
- **`base_model_{language}.sample.csv`**: Sample embeddings for validation

### Fine-tuning
- **`{repo_name}_embeddings.csv`**: Document vectors for each source file
  - Column 1: Repository-relative file path
  - Columns 2-201: 200-dimensional embedding vectors
- **`{repo_name}_finetuned.d2v`**: Fine-tuned Doc2Vec model
- **`{repo_name}_metadata.json`**: Fine-tuning metadata

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
│       ├── train-base-model.yaml    # Train base model on popular repos
│       └── finetune-model.yaml      # Fine-tune model on specific repo
├── src/
│   ├── train_base_model.py         # Train single model on multiple repos
│   ├── finetune_and_embed.py       # Fine-tune model and generate embeddings
│   ├── get_popular_repos.py        # Fetch popular repos from GitHub API
│   └── train_doc2vec.py            # Original single-repo training script
├── requirements.txt                 # Python dependencies
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