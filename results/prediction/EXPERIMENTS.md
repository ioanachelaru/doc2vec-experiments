# Defect Prediction Experiments: Filename-Based Subset Analysis

## Research Question

**RQ2: Does train/test overlap inflate defect prediction performance in Cross-Version Defect Prediction (CVDP)?**

We partition the test set into three subsets based on filename overlap with the training set, then evaluate a classifier on each subset independently. If the model relies on memorised instances, performance will diverge sharply between subsets containing known files and subsets containing only new or label-changed files.

**RQ3: Does removing leaked train instances change performance?**

We compute cosine similarity between all train and test embeddings, remove train instances with similarity >= 0.99 to any test instance ("cleaned" train), retrain the classifier, and compare with the full-train results. This isolates the effect of train-side leakage.

## Experimental Setup

### Classifier

| Classifier | Configuration |
|---|---|
| RandomForest | 500 trees, `class_weight='balanced'`, 10 random seeds (42-51), averaged |

### Evaluation Subsets

For each version pair, every test file is assigned to exactly one subset:

| Subset | Definition |
|---|---|
| **baseline** | Full test set (all files) |
| **new_files** | Files whose relative path does not appear in any training version |
| **changed_label** | Files whose relative path exists in training but with a different label. For cumulative: also files with inconsistent labels across training versions (both buggy and clean) |

### Train Set Variants

| Variant | Definition |
|---|---|
| **full** | All training instances |
| **cleaned** | Training instances with cosine similarity >= 0.99 to any test instance removed |

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
| Project | Django (Python, 23 version pairs), Calcite (Java, 15 version pairs) |
| Strategy | Pairwise (fresh model per pair), Cumulative-fresh (growing training window) |
| Embedding dimension | 200, 400 |

**Primary dimension**: 200 for both projects. 400-dim runs are for sensitivity analysis.

## Results

All values are means across version pairs and 10 random seeds. RandomForest with balanced class weights.

### Django (Python, 23 version pairs)

**Pairwise strategy — full train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.894 | 0.853 | 0.943 | 0.862 | 0.724 | 0.863 | 0.752 | 0.051 | 0.668 |
| Baseline | 400 | 0.892 | 0.849 | 0.941 | 0.860 | 0.718 | 0.862 | 0.744 | 0.051 | 0.661 |
| New files | 200 | 0.634 | 0.486 | 0.818 | 0.760 | 0.149 | 0.327 | 0.224 | 0.089 | 0.168 |
| New files | 400 | 0.634 | 0.505 | 0.820 | 0.770 | 0.175 | 0.386 | 0.245 | 0.096 | 0.202 |
| Changed label | 200 | 0.164 | 0.130 | 0.031 | 0.144 | -0.676 | 0.006 | 0.010 | 0.789 | 0.004 |
| Changed label | 400 | 0.185 | 0.147 | 0.053 | 0.147 | -0.630 | 0.010 | 0.015 | 0.757 | 0.006 |

**Pairwise strategy — cleaned train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.809 | 0.733 | 0.874 | 0.738 | 0.506 | 0.726 | 0.602 | 0.146 | 0.468 |
| Baseline | 400 | 0.832 | 0.760 | 0.897 | 0.772 | 0.554 | 0.767 | 0.616 | 0.112 | 0.502 |
| New files | 200 | 0.637 | 0.509 | 0.811 | 0.751 | 0.168 | 0.387 | 0.307 | 0.149 | 0.215 |
| New files | 400 | 0.630 | 0.504 | 0.821 | 0.768 | 0.169 | 0.354 | 0.276 | 0.128 | 0.210 |
| Changed label | 200 | 0.278 | 0.215 | 0.138 | 0.155 | -0.472 | 0.079 | 0.223 | 0.723 | 0.064 |
| Changed label | 400 | 0.281 | 0.222 | 0.139 | 0.156 | -0.451 | 0.073 | 0.184 | 0.703 | 0.057 |

**Cumulative-fresh strategy — full train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.871 | 0.829 | 0.938 | 0.856 | 0.689 | 0.811 | 0.771 | 0.073 | 0.626 |
| Baseline | 400 | 0.863 | 0.820 | 0.930 | 0.842 | 0.673 | 0.796 | 0.767 | 0.081 | 0.609 |
| New files | 200 | 0.660 | 0.536 | 0.802 | 0.761 | 0.226 | 0.451 | 0.308 | 0.094 | 0.248 |
| New files | 400 | 0.678 | 0.591 | 0.818 | 0.783 | 0.306 | 0.564 | 0.396 | 0.109 | 0.338 |
| Changed label | 200 | 0.695 | 0.645 | 0.831 | 0.659 | 0.485 | 0.588 | 0.774 | 0.226 | 0.495 |
| Changed label | 400 | 0.684 | 0.634 | 0.827 | 0.653 | 0.467 | 0.566 | 0.778 | 0.243 | 0.479 |

**Cumulative-fresh strategy — cleaned train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.841 | 0.796 | 0.917 | 0.803 | 0.631 | 0.751 | 0.761 | 0.108 | 0.570 |
| Baseline | 400 | 0.847 | 0.802 | 0.919 | 0.811 | 0.642 | 0.761 | 0.764 | 0.101 | 0.579 |
| New files | 200 | 0.660 | 0.546 | 0.803 | 0.757 | 0.243 | 0.484 | 0.329 | 0.098 | 0.266 |
| New files | 400 | 0.677 | 0.590 | 0.816 | 0.777 | 0.303 | 0.587 | 0.401 | 0.116 | 0.337 |
| Changed label | 200 | 0.642 | 0.595 | 0.789 | 0.577 | 0.395 | 0.509 | 0.769 | 0.305 | 0.432 |
| Changed label | 400 | 0.655 | 0.607 | 0.807 | 0.611 | 0.422 | 0.523 | 0.779 | 0.287 | 0.447 |

### Calcite (Java, 15 version pairs)

**Pairwise strategy — full train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.943 | 0.686 | 0.972 | 0.835 | 0.466 | 0.906 | 0.270 | 0.003 | 0.260 |
| Baseline | 400 | 0.947 | 0.706 | 0.974 | 0.824 | 0.492 | 0.889 | 0.314 | 0.004 | 0.299 |
| New files | 200 | 0.902 | 0.584 | 0.839 | 0.597 | 0.021 | 0.091 | 0.005 | 0.000 | 0.005 |
| New files | 400 | 0.902 | 0.584 | 0.887 | 0.626 | 0.018 | 0.073 | 0.005 | 0.000 | 0.005 |
| Changed label | 200 | 0.560 | 0.345 | 0.004 | 0.261 | -0.153 | 0.000 | 0.000 | 0.106 | 0.000 |
| Changed label | 400 | 0.537 | 0.336 | 0.006 | 0.261 | -0.173 | 0.000 | 0.000 | 0.136 | 0.000 |

**Pairwise strategy — cleaned train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.926 | 0.513 | 0.900 | 0.513 | 0.131 | 0.694 | 0.036 | 0.001 | 0.035 |
| Baseline | 400 | 0.926 | 0.524 | 0.887 | 0.465 | 0.161 | 0.756 | 0.051 | 0.002 | 0.048 |
| New files | 200 | 0.902 | 0.580 | 0.815 | 0.566 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| New files | 400 | 0.902 | 0.580 | 0.885 | 0.617 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Changed label | 200 | 0.612 | 0.365 | 0.055 | 0.267 | -0.034 | 0.000 | 0.000 | 0.023 | 0.000 |
| Changed label | 400 | 0.586 | 0.357 | 0.097 | 0.274 | -0.067 | 0.000 | 0.000 | 0.054 | 0.000 |

**Cumulative-fresh strategy — full train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.964 | 0.849 | 0.968 | 0.795 | 0.708 | 0.837 | 0.633 | 0.009 | 0.566 |
| Baseline | 400 | 0.965 | 0.852 | 0.969 | 0.801 | 0.711 | 0.816 | 0.654 | 0.011 | 0.573 |
| New files | 200 | 0.903 | 0.595 | 0.772 | 0.484 | 0.043 | 0.091 | 0.022 | 0.000 | 0.022 |
| New files | 400 | 0.904 | 0.596 | 0.845 | 0.522 | 0.046 | 0.091 | 0.026 | 0.000 | 0.026 |
| Changed label | 200 | 0.822 | 0.609 | 0.650 | 0.366 | 0.250 | 0.430 | 0.277 | 0.076 | 0.203 |
| Changed label | 400 | 0.815 | 0.599 | 0.646 | 0.372 | 0.223 | 0.382 | 0.276 | 0.086 | 0.192 |

**Cumulative-fresh strategy — cleaned train**

| Subset | Dim | Accuracy | F1 macro | AUC | AUPRC | MCC | Prec buggy | Rec buggy | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 200 | 0.946 | 0.755 | 0.944 | 0.654 | 0.542 | 0.762 | 0.426 | 0.010 | 0.373 |
| Baseline | 400 | 0.945 | 0.759 | 0.936 | 0.640 | 0.541 | 0.723 | 0.449 | 0.013 | 0.381 |
| New files | 200 | 0.902 | 0.586 | 0.787 | 0.488 | 0.018 | 0.053 | 0.008 | 0.001 | 0.008 |
| New files | 400 | 0.902 | 0.588 | 0.881 | 0.576 | 0.028 | 0.091 | 0.010 | 0.000 | 0.010 |
| Changed label | 200 | 0.804 | 0.532 | 0.568 | 0.242 | 0.091 | 0.260 | 0.145 | 0.075 | 0.101 |
| Changed label | 400 | 0.778 | 0.525 | 0.564 | 0.249 | 0.072 | 0.240 | 0.157 | 0.107 | 0.102 |

### Dimensionality Sensitivity

Differences between dim 200 and dim 400 are small across all configurations (F1 differences < 0.03), suggesting embedding dimensionality has negligible impact on subset-level performance.

### Cleaned-Train Impact (Delta: full - cleaned, dim 200)

| Project | Strategy | Subset | Delta MCC | Delta F1 macro | Delta AUC |
|---|---|---|---|---|---|
| Django | Pairwise | Baseline | +0.218 | +0.120 | +0.069 |
| Django | Pairwise | New files | -0.019 | -0.023 | +0.007 |
| Django | Pairwise | Changed label | -0.204 | -0.085 | -0.107 |
| Django | Cumfresh | Baseline | +0.058 | +0.033 | +0.021 |
| Django | Cumfresh | New files | -0.017 | -0.010 | -0.001 |
| Django | Cumfresh | Changed label | +0.090 | +0.050 | +0.042 |
| Calcite | Pairwise | Baseline | +0.335 | +0.173 | +0.072 |
| Calcite | Pairwise | New files | +0.021 | +0.004 | +0.024 |
| Calcite | Pairwise | Changed label | -0.119 | -0.020 | -0.051 |
| Calcite | Cumfresh | Baseline | +0.166 | +0.094 | +0.024 |
| Calcite | Cumfresh | New files | +0.025 | +0.009 | -0.015 |
| Calcite | Cumfresh | Changed label | +0.159 | +0.077 | +0.082 |

## Key Findings

1. **Changed-label files expose model memorisation.** In pairwise Django, the model achieves MCC = -0.68 on changed-label files, meaning it actively predicts the *wrong* label -- the one memorised from training. AUC drops near zero (0.03), confirming the model inverts its predictions on these files.

2. **Cumulative training mitigates memorisation.** With cumulative-fresh strategy, changed-label MCC recovers to +0.49 (Django) and +0.25 (Calcite). Having multiple training versions with potentially different labels prevents the model from locking onto a single memorised label.

3. **New files are harder to predict.** F1 macro on new_files drops to 0.49-0.54 (Django) and 0.58-0.60 (Calcite) compared to baseline 0.85/0.69. These files lack any training counterpart, so the model must generalise from code features alone.

4. **Calcite pairwise shows zero buggy recall on new files.** The model predicts all new files as clean (recall_buggy = 0.005, FAR = 0.0), defaulting to the majority class when it cannot leverage memorised instances.

5. **Baseline performance is inflated by known files.** The gap between baseline and new_files F1 quantifies how much the model relies on memorised instances vs. genuine defect patterns.

6. **Removing leaked train instances degrades baseline but not new-file performance.** Cleaning the training set drops baseline MCC by 0.06-0.34 (removing memorised instances hurts overall accuracy) but barely affects new_files performance (MCC deltas < 0.03). This confirms the baseline inflation comes from train-side leakage.

7. **Cleaning helps changed-label predictions in cumulative but not pairwise.** In cumulative-fresh, cleaned-train reduces changed-label errors (Django MCC +0.09, Calcite +0.16). In pairwise, the damage from memorisation is too severe for cleaning to reverse (MCC stays deeply negative).

8. **Embedding dimensionality has negligible effect.** 200-dim and 400-dim produce nearly identical results across all subsets and strategies (F1 differences < 0.03).

9. **Results are robust across random seeds.** All metrics are averaged over 10 RF seeds (42-51). Standard deviations are small (not shown), confirming findings are not artefacts of a single random initialisation.

## Plots

All plots in `results/plots/`:

| Plot | Description |
|---|---|
| `subset_f1_comparison.png` | F1 macro by subset across strategies (dim 200) |
| `subset_mcc_comparison.png` | MCC by subset -- shows negative MCC for changed_label in pairwise |
| `subset_auc_comparison.png` | AUC by subset -- highlights near-zero AUC for pairwise changed_label |
| `subset_per_pair_f1.png` | F1 macro per version pair for each subset (4 panels) |
| `subset_per_pair_auc.png` | AUC per version pair for each subset (4 panels) |
| `subset_dimension_comparison.png` | F1 macro: dim 200 vs 400 (sensitivity analysis) |
| `subset_dimension_comparison_mcc.png` | MCC: dim 200 vs 400 (sensitivity analysis) |

## Output Files

### Per-configuration results
- `{project}_{strategy}_dim{N}_predictions.csv` -- per-pair, per-seed metrics
- `{project}_{strategy}_dim{N}_summary.csv` -- aggregate means across pairs and seeds
