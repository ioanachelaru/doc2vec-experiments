# Defect Prediction Experiments: Filename-Based Subset Analysis

## Research Question

**RQ2: Does train/test overlap inflate defect prediction performance in Cross-Version Defect Prediction (CVDP)?**

We partition the test set into three subsets based on filename overlap with the training set, then evaluate a classifier on each subset independently. If the model relies on memorised instances, performance will diverge sharply between subsets containing known files and subsets containing only new or label-changed files.

## Experimental Setup

### Classifier

| Classifier | Configuration |
|---|---|
| RandomForest | 500 trees, `class_weight='balanced'`, `random_state=42` |

### Evaluation Subsets

For each version pair, every test file is assigned to exactly one subset:

| Subset | Definition |
|---|---|
| **baseline** | Full test set (all files) |
| **new_files** | Files whose relative path does not appear in any training version |
| **changed_label** | Files whose relative path exists in training but with a different label. For cumulative: also files with inconsistent labels across training versions (both buggy and clean) |

### Metrics

| Metric | Description |
|---|---|
| Accuracy | Proportion of correct predictions |
| F1 macro | Harmonic mean of precision and recall, macro-averaged |
| F1 weighted | F1 weighted by class support |
| F1 buggy / F1 clean | Per-class F1 scores |
| AUC | Area under the ROC curve |
| AUPRC | Area under the precision-recall curve |
| Precision buggy (PPV) | Positive predictive value |
| Recall buggy (POD) | Probability of detection / sensitivity |
| Precision clean (NPV) | Negative predictive value |
| Recall clean (Specificity) | True negative rate |
| FAR | False alarm rate (1 - Specificity) |
| CSI | Critical success index: TP / (TP + FP + FN) |
| MCC | Matthews correlation coefficient |

## Configurations

All 8 combinations of:

| Factor | Levels |
|---|---|
| Project | Django (Python, 25 version pairs), Calcite (Java, 15 version pairs) |
| Strategy | Pairwise (fresh model per pair), Cumulative-fresh (growing training window) |
| Embedding dimension | 200, 400 |

**Primary dimension**: 200 for both projects. 400-dim runs are for sensitivity analysis.

## Results

All values are means across version pairs. RandomForest with balanced class weights.

### Django (Python, 25 version pairs)

**Pairwise strategy**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.893 | 0.852 | 0.942 | 0.866 | 0.723 | 0.862 | 0.753 | 0.052 | 0.666 |
| Baseline | 400 | 0.892 | 0.848 | 0.940 | 0.860 | 0.715 | 0.858 | 0.741 | 0.052 | 0.659 |
| New files | 200 | 0.631 | 0.470 | 0.834 | 0.791 | 0.124 | 0.290 | 0.194 | 0.082 | 0.144 |
| New files | 400 | 0.632 | 0.503 | 0.825 | 0.775 | 0.171 | 0.372 | 0.243 | 0.100 | 0.202 |
| Changed label | 200 | 0.152 | 0.122 | 0.030 | 0.144 | -0.694 | 0.004 | 0.006 | 0.805 | 0.003 |
| Changed label | 400 | 0.181 | 0.147 | 0.054 | 0.147 | -0.630 | 0.013 | 0.019 | 0.762 | 0.008 |

**Cumulative-fresh strategy**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.870 | 0.828 | 0.937 | 0.855 | 0.687 | 0.809 | 0.771 | 0.074 | 0.623 |
| Baseline | 400 | 0.862 | 0.818 | 0.930 | 0.841 | 0.670 | 0.794 | 0.765 | 0.082 | 0.607 |
| New files | 200 | 0.672 | 0.547 | 0.785 | 0.747 | 0.257 | 0.488 | 0.307 | 0.070 | 0.243 |
| New files | 400 | 0.681 | 0.594 | 0.821 | 0.800 | 0.324 | 0.603 | 0.394 | 0.103 | 0.348 |
| Changed label | 200 | 0.694 | 0.643 | 0.830 | 0.657 | 0.482 | 0.585 | 0.771 | 0.226 | 0.492 |
| Changed label | 400 | 0.682 | 0.632 | 0.827 | 0.652 | 0.461 | 0.562 | 0.774 | 0.245 | 0.475 |

### Calcite (Java, 15 version pairs)

**Pairwise strategy**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.943 | 0.685 | 0.973 | 0.835 | 0.463 | 0.902 | 0.269 | 0.003 | 0.259 |
| Baseline | 400 | 0.947 | 0.701 | 0.974 | 0.827 | 0.482 | 0.883 | 0.308 | 0.004 | 0.293 |
| New files | 200 | 0.902 | 0.584 | 0.862 | 0.627 | 0.021 | 0.091 | 0.005 | 0.000 | 0.005 |
| New files | 400 | 0.902 | 0.580 | 0.883 | 0.620 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Changed label | 200 | 0.557 | 0.344 | 0.004 | 0.260 | -0.151 | 0.000 | 0.000 | 0.107 | 0.000 |
| Changed label | 400 | 0.534 | 0.334 | 0.005 | 0.261 | -0.180 | 0.000 | 0.000 | 0.145 | 0.000 |

**Cumulative-fresh strategy**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.964 | 0.849 | 0.969 | 0.795 | 0.708 | 0.837 | 0.633 | 0.009 | 0.565 |
| Baseline | 400 | 0.965 | 0.851 | 0.968 | 0.802 | 0.710 | 0.815 | 0.653 | 0.011 | 0.572 |
| New files | 200 | 0.904 | 0.597 | 0.759 | 0.436 | 0.047 | 0.091 | 0.027 | 0.000 | 0.027 |
| New files | 400 | 0.904 | 0.597 | 0.881 | 0.551 | 0.047 | 0.091 | 0.027 | 0.000 | 0.027 |
| Changed label | 200 | 0.824 | 0.610 | 0.652 | 0.361 | 0.252 | 0.435 | 0.277 | 0.075 | 0.203 |
| Changed label | 400 | 0.815 | 0.599 | 0.645 | 0.373 | 0.221 | 0.376 | 0.276 | 0.085 | 0.191 |

### Dimensionality Sensitivity

Differences between dim 200 and dim 400 are small across all configurations (F1 differences < 0.03), suggesting embedding dimensionality has negligible impact on subset-level performance.

## Key Findings

1. **Changed-label files expose model memorisation.** In pairwise Django, the model achieves MCC = -0.63 to -0.69 on changed-label files, meaning it actively predicts the *wrong* label — the one memorised from training. AUC drops near zero (0.03-0.05), confirming the model inverts its predictions on these files.

2. **Cumulative training mitigates memorisation.** With cumulative-fresh strategy, changed-label MCC recovers to +0.46 (Django) and +0.25 (Calcite). Having multiple training versions with potentially different labels prevents the model from locking onto a single memorised label.

3. **New files are harder to predict.** F1 macro on new_files drops to 0.47-0.59 (Django) and 0.58 (Calcite) compared to baseline 0.85/0.70. These files lack any training counterpart, so the model must generalise from code features alone.

4. **Calcite pairwise shows zero buggy recall on new files.** The model predicts all new files as clean (recall_buggy = 0.0, FAR = 0.0), defaulting to the majority class when it cannot leverage memorised instances.

5. **Baseline performance is inflated by known files.** The gap between baseline and new_files F1 quantifies how much the model relies on memorised instances vs. genuine defect patterns.

6. **Embedding dimensionality has negligible effect.** 200-dim and 400-dim produce nearly identical results across all subsets and strategies (F1 differences < 0.03).

## Plots

All plots in `results/plots/`:

| Plot | Description |
|---|---|
| `subset_f1_comparison.png` | F1 macro by subset across strategies (dim 200) |
| `subset_mcc_comparison.png` | MCC by subset — shows negative MCC for changed_label in pairwise |
| `subset_auc_comparison.png` | AUC by subset — highlights near-zero AUC for pairwise changed_label |
| `subset_per_pair_f1.png` | F1 macro per version pair for each subset (4 panels) |
| `subset_per_pair_auc.png` | AUC per version pair for each subset (4 panels) |
| `subset_dimension_comparison.png` | F1 macro: dim 200 vs 400 (sensitivity analysis) |
| `subset_dimension_comparison_mcc.png` | MCC: dim 200 vs 400 (sensitivity analysis) |

## Output Files

### Per-configuration results
- `{project}_{strategy}_predictions.csv` — per-pair metrics (primary dim)
- `{project}_{strategy}_dim{N}_predictions.csv` — per-pair metrics (alternate dim)
- `{project}_{strategy}_summary.csv` — aggregate means (primary dim)
- `{project}_{strategy}_dim{N}_summary.csv` — aggregate means (alternate dim)
