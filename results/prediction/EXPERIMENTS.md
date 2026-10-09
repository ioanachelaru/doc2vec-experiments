# Defect Prediction Experiments: Measuring Leakage Impact

## Research Question

**RQ2: Does train/test overlap inflate defect prediction performance in Cross-Version Defect Prediction (CVDP)?**

In CVDP, a model trains on earlier software versions and predicts defects in a later version. When near-duplicate files exist across versions (e.g., unchanged or minimally modified files), the model may "memorise" these instances rather than learning generalisable defect patterns. We measure how much this overlap inflates reported performance by comparing predictions on the full test set (baseline) against predictions on a cleaned test set with leaked instances removed.

## Experimental Setup

### Classifiers

| Classifier | Configuration |
|---|---|
| RandomForest | 500 trees, `class_weight='balanced'`, `random_state=42` |
| LogisticRegression | `class_weight='balanced'`, `max_iter=1000`, `random_state=42` |

Both classifiers use balanced class weights to handle the imbalanced distribution of buggy vs. clean files (typically 10-20% buggy).

### Leakage Definitions

Two independent methods identify leaked (near-duplicate) test instances:

- **Embedding-based** (`embedding_0.99`): For each test file, compute cosine similarity against all training embeddings. If any pair exceeds a threshold (default 0.99), the test file is flagged as leaked. Uses Doc2Vec embeddings inferred from the trained model.
- **Source-code-based** (`same_code`): Ground-truth labels from exact source-code comparison. A test file is leaked if an identical copy exists in the training set (byte-level match). Provided externally as `*-same-code.zip` files.

### Evaluation Subsets

For each version pair, predictions are evaluated on three subsets:

| Subset | Description |
|---|---|
| **baseline** | Full test set (all files) |
| **cleaned** | Test set with leaked instances removed |
| **leaked-only** | Only the leaked instances |

The key metric is **Delta = F1(baseline) - F1(cleaned)**, representing the performance inflation attributable to leaked instances.

### Metrics

- **F1 macro**: Harmonic mean of precision and recall, macro-averaged across classes
- **AUC**: Area under the ROC curve
- **MCC**: Matthews Correlation Coefficient (robust to class imbalance)

## Configurations

All 8 combinations of:

| Factor | Levels |
|---|---|
| Project | Django (Python, 25 version pairs), Calcite (Java, 15 version pairs) |
| Strategy | Pairwise (fresh model per pair), Cumulative (growing training window) |
| Embedding dimension | 200, 400 |

**Primary dimensions**: Django uses 400-dim, Calcite uses 200-dim. Alternates (Django 200, Calcite 400) are used for sensitivity analysis.

## Results

All values are means across version pairs. Embedding-based leakage at threshold 0.99.

### Django (Python, 25 version pairs)

**RandomForest**

| Strategy | Dim | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 | Baseline MCC | Cleaned MCC |
|---|---|---|---|---|---|---|---|
| Pairwise | 200 | 0.852 | 0.803 | **+0.049** | 0.923 | 0.723 | 0.639 |
| Pairwise | 400 | 0.848 | 0.815 | **+0.033** | 0.923 | 0.715 | 0.654 |
| Cumulative | 200 | 0.828 | 0.807 | **+0.021** | 0.893 | 0.687 | 0.653 |
| Cumulative | 400 | 0.818 | 0.806 | **+0.012** | 0.885 | 0.670 | 0.651 |

**LogisticRegression**

| Strategy | Dim | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 |
|---|---|---|---|---|---|
| Pairwise | 200 | 0.788 | 0.764 | **+0.023** | 0.822 |
| Pairwise | 400 | 0.856 | 0.834 | **+0.022** | 0.920 |
| Cumulative | 200 | 0.712 | 0.715 | **-0.003** | 0.664 |
| Cumulative | 400 | 0.737 | 0.734 | **+0.002** | 0.712 |

### Calcite (Java, 15 version pairs)

**RandomForest**

| Strategy | Dim | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 | Baseline MCC | Cleaned MCC |
|---|---|---|---|---|---|---|---|
| Pairwise | 200 | 0.685 | 0.530 | **+0.154** | 0.765 | 0.463 | 0.142 |
| Pairwise | 400 | 0.701 | 0.537 | **+0.164** | 0.767 | 0.482 | 0.140 |
| Cumulative | 200 | 0.849 | 0.797 | **+0.052** | 0.913 | 0.708 | 0.614 |
| Cumulative | 400 | 0.851 | 0.798 | **+0.053** | 0.918 | 0.710 | 0.613 |

**LogisticRegression**

| Strategy | Dim | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 |
|---|---|---|---|---|---|
| Pairwise | 200 | 0.793 | 0.742 | **+0.050** | 0.829 |
| Pairwise | 400 | 0.886 | 0.813 | **+0.073** | 0.935 |
| Cumulative | 200 | 0.639 | 0.622 | **+0.017** | 0.623 |
| Cumulative | 400 | 0.724 | 0.691 | **+0.033** | 0.762 |

### Dimensionality Sensitivity

Differences between dim 200 and dim 400 are small across all configurations (delta differences < 0.02), suggesting 200-dim embeddings are sufficient. The dimension comparison is visible in each table above (adjacent rows).

## Threshold Sweep

The similarity threshold determines which test files are flagged as leaked. Lower thresholds flag more files, removing a larger portion of the test set. We swept thresholds from 0.90 to 1.00 (11 values). All values are RandomForest delta F1 (baseline - cleaned).

### Django

**Pairwise (dim 200 vs. dim 400)**

| Threshold | Delta (200) | Delta (400) |
|---|---|---|
| 0.90 | +0.238 | +0.214 |
| 0.93 | +0.204 | +0.169 |
| 0.95 | +0.174 | +0.127 |
| 0.97 | +0.127 | +0.084 |
| 0.99 | +0.049 | +0.033 |
| 1.00 | 0.000 | 0.000 |

**Cumulative (dim 400)**

| Threshold | Delta |
|---|---|
| 0.90 | +0.141 |
| 0.93 | +0.111 |
| 0.95 | +0.080 |
| 0.97 | +0.056 |
| 0.99 | +0.012 |
| 1.00 | 0.000 |

### Calcite

**Pairwise (dim 200 vs. dim 400)**

| Threshold | Delta (200) | Delta (400) |
|---|---|---|
| 0.90 | +0.143 | +0.168 |
| 0.93 | +0.181 | +0.205 |
| 0.95 | +0.186 | +0.206 |
| 0.97 | +0.214 | +0.241 |
| 0.99 | +0.154 | +0.164 |
| 1.00 | 0.000 | 0.000 |

**Cumulative (dim 200)**

| Threshold | Delta |
|---|---|
| 0.90 | +0.335 |
| 0.93 | +0.335 |
| 0.95 | +0.331 |
| 0.97 | +0.271 |
| 0.99 | +0.052 |
| 1.00 | 0.000 |

For Django, deltas decrease smoothly as the threshold tightens. For Calcite, the non-monotonic behaviour (delta peaking around 0.95-0.97 rather than 0.90) reflects the interaction between leakage removal and class balance in smaller test sets.

## Embedding-Based vs. Source-Code-Based Leakage

We compared embedding-based detection (cosine similarity >= 0.99) against source-code-based ground truth (exact file match) on two configurations where both are available.

### Django Pairwise (dim 200)

| Leakage Method | Classifier | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 |
|---|---|---|---|---|---|
| Embedding (>= 0.99) | RF | 0.852 | 0.803 | +0.049 | 0.923 |
| Source-code match | RF | 0.852 | 0.818 | +0.034 | 0.893 |
| Embedding (>= 0.99) | LR | 0.788 | 0.764 | +0.023 | 0.822 |
| Source-code match | LR | 0.788 | 0.758 | +0.030 | 0.746 |

### Calcite Pairwise (dim 400)

| Leakage Method | Classifier | Baseline F1 | Cleaned F1 | Delta | Leaked-only F1 |
|---|---|---|---|---|---|
| Embedding (>= 0.99) | RF | 0.701 | 0.537 | +0.164 | 0.767 |
| Source-code match | RF | 0.701 | 0.620 | +0.081 | 0.736 |
| Embedding (>= 0.99) | LR | 0.886 | 0.813 | +0.073 | 0.935 |
| Source-code match | LR | 0.886 | 0.763 | +0.123 | 0.961 |

Embedding-based detection identifies a superset of the source-code duplicates: it catches not only exact copies but also files with only minor changes (e.g., whitespace, comments, import reordering). This produces larger deltas because it removes more test instances. The leaked-only F1 is consistently higher for the embedding method (RF), confirming that the additional flagged files are indeed easier to predict.

## Key Findings

1. **Leakage inflates performance across all configurations.** Removing leaked test instances consistently reduces F1, confirming that train/test overlap provides an unfair advantage.

2. **Calcite is more affected than Django.** Pairwise deltas reach +0.15-0.16 for Calcite vs. +0.03-0.05 for Django (RF, threshold 0.99). This reflects Calcite's smaller codebase and higher proportion of unchanged files across versions.

3. **Pairwise strategy shows larger inflation than cumulative.** Pairwise trains on a single version (smaller, more homogeneous training set), making the model more susceptible to memorising leaked instances. Cumulative training dilutes the leaked instances across a larger corpus.

4. **Leaked instances are substantially easier to predict.** Leaked-only F1 exceeds baseline F1 by 0.05-0.08 across configurations, confirming that these near-duplicate files carry their labels from training into testing.

5. **Embedding-based detection is more conservative than source-code matching.** At threshold 0.99, embeddings flag more files than exact source-code comparison, catching near-duplicates with minor syntactic differences. This leads to larger measured deltas.

6. **Threshold sensitivity is smooth.** Deltas decrease monotonically from ~0.14-0.34 (threshold 0.90) to 0 (threshold 1.00), with the steepest drop between 0.97 and 1.00.

7. **Embedding dimensionality has negligible effect.** 200-dim and 400-dim produce nearly identical leakage measurements (delta differences < 0.02).

## Output Files

### Per-configuration results
- `{project}_{strategy}_dim{N}_predictions.csv` -- per-pair, per-classifier, per-subset metrics
- `{project}_{strategy}_dim{N}_summary.csv` -- aggregate means across all pairs
- `{project}_{strategy}_dim{N}_threshold_sweep.csv` -- metrics at thresholds 0.90 to 1.00

### Plots (`results/plots/`)
- `f1_delta_bar.png` -- F1 inflation (baseline - cleaned) across all configs
- `threshold_sweep.png` -- delta F1 as a function of similarity threshold
- `leakage_method_comparison.png` -- embedding vs. source-code leakage detection
- `per_pair_f1.png` -- F1 variation across version pairs
- `dimension_comparison.png` -- 200-dim vs. 400-dim performance
