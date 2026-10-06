# Effect of Embedding Dimensionality (200 vs 400) on Leakage Detection

## Data Sources

| Run ID | Project | Dim | Strategy | Artifact |
|--------|---------|-----|----------|----------|
| 32718528477 (base) | Django | 400 | pairwise + cumul-fresh | `pairwise-django`, `cumulative-fresh-django` |
| 37197438933 (base) | Django | 200 | pairwise + cumul-fresh | `pairwise-django-dim200`, `cumulative-fresh-django-dim200` |
| 32937494910 (base) | Calcite | 200 | pairwise + cumul-fresh | `pairwise-calcite`, `cumulative-fresh-calcite` |
| 37197450624 (base) | Calcite | 400 | pairwise + cumul-fresh | `pairwise-calcite-dim400`, `cumulative-fresh-calcite-dim400` |

All runs use the same pipeline (`pairwise_pipeline.py`), same threshold (0.99), same training epochs (10), same inference epochs (200). The only variable is the base model's `vector_size`.


## Aggregate Results

| Project | Strategy | Dim | Avg Leakage % | Total Same-file Pairs | Total Collision Pairs |
|---------|----------|-----|---------------|----------------------|----------------------|
| Django | Pairwise | 400 | 26.60 | 4798 | 10 |
| Django | Pairwise | 200 | 38.28 | 6892 | 20 |
| Django | Cumul-fresh | 400 | 12.55 | 4148 | 23 |
| Django | Cumul-fresh | 200 | 23.15 | 8832 | 33 |
| Calcite | Pairwise | 200 | 35.66 | 8034 | 57 |
| Calcite | Pairwise | 400 | 45.96 | 10321 | 64 |
| Calcite | Cumul-fresh | 200 | 21.79 | 13391 | 276 |
| Calcite | Cumul-fresh | 400 | 24.22 | 14001 | 252 |


## Per-Pair Comparison (Pairwise Strategy)

### Django

| Pair | Train | Test | dim400 Leak% | dim200 Leak% | Delta (pp) |
|------|-------|------|-------------|-------------|------------|
| 1 | 1.0 | 1.1 | 19.82 | 32.66 | +12.84 |
| 2 | 1.1 | 1.2 | 17.24 | 26.89 | +9.65 |
| 3 | 1.2 | 1.3 | 21.02 | 30.91 | +9.89 |
| 4 | 1.3 | 1.4 | 18.26 | 27.95 | +9.69 |
| 5 | 1.4 | 1.5 | 16.11 | 26.64 | +10.53 |
| 6 | 1.5 | 1.6 | 19.46 | 30.68 | +11.22 |
| 7 | 1.6 | 1.7 | 15.36 | 23.22 | +7.86 |
| 8 | 1.7 | 1.8 | 12.66 | 18.63 | +5.97 |
| 9 | 1.8 | 1.9 | 16.85 | 28.09 | +11.24 |
| 10 | 1.9 | 1.10 | 21.13 | 31.41 | +10.28 |
| 11 | 1.10 | 1.11 | 25.14 | 38.48 | +13.34 |
| 12 | 1.11 | 2.0 | 4.55 | 8.95 | +4.40 |
| 13 | 2.0 | 2.1 | 29.79 | 40.85 | +11.06 |
| 14 | 2.1 | 2.2 | 31.70 | 43.44 | +11.74 |
| 15 | 2.2 | 3.0 | 34.08 | 47.63 | +13.55 |
| 16 | 3.0 | 3.1 | 30.33 | 44.46 | +14.13 |
| 17 | 3.1 | 3.2 | 33.38 | 47.35 | +13.97 |
| 18 | 3.2 | 4.0 | 33.43 | 46.38 | +12.95 |
| 19 | 4.0 | 4.1 | 36.67 | 47.92 | +11.25 |
| 20 | 4.1 | 4.2 | 38.10 | 53.51 | +15.41 |
| 21 | 4.2 | 5.0 | 35.93 | 51.09 | +15.16 |
| 22 | 5.0 | 5.1 | 38.30 | 53.49 | +15.19 |
| 23 | 5.1 | 5.2 | 40.82 | 55.65 | +14.83 |
| 24 | 5.2 | 6.0 | 38.24 | 51.20 | +12.96 |
| 25 | 6.0 | 6.1 | 36.69 | 49.54 | +12.85 |

Delta direction: dim200 always higher. Range: +4.40 to +15.41 pp.

### Calcite

| Pair | Train | Test | dim200 Leak% | dim400 Leak% | Delta (pp) |
|------|-------|------|-------------|-------------|------------|
| 1 | 1.0.0 | 1.1.0 | 31.85 | 44.59 | +12.74 |
| 2 | 1.1.0 | 1.2.0 | 36.28 | 49.16 | +12.88 |
| 3 | 1.2.0 | 1.3.0 | 35.61 | 48.03 | +12.42 |
| 4 | 1.3.0 | 1.4.0 | 34.98 | 48.65 | +13.67 |
| 5 | 1.4.0 | 1.5.0 | 31.31 | 41.63 | +10.32 |
| 6 | 1.5.0 | 1.6.0 | 31.48 | 43.02 | +11.54 |
| 7 | 1.6.0 | 1.7.0 | 34.01 | 44.41 | +10.40 |
| 8 | 1.7.0 | 1.8.0 | 34.82 | 44.86 | +10.04 |
| 9 | 1.8.0 | 1.9.0 | 37.65 | 46.36 | +8.71 |
| 10 | 1.9.0 | 1.10.0 | 45.01 | 54.07 | +9.06 |
| 11 | 1.10.0 | 1.11.0 | 37.56 | 46.48 | +8.92 |
| 12 | 1.11.0 | 1.12.0 | 34.19 | 40.89 | +6.70 |
| 13 | 1.12.0 | 1.13.0 | 37.69 | 45.92 | +8.23 |
| 14 | 1.13.0 | 1.14.0 | 35.45 | 45.62 | +10.17 |
| 15 | 1.14.0 | 1.15.0 | 36.96 | 45.69 | +8.73 |

Delta direction: dim400 always higher. Range: +6.70 to +13.67 pp.


## Key Finding

The effect of embedding dimensionality on leakage is **project-dependent**:

- **Django**: dim200 produces more leakage than dim400 (avg +11.7pp)
- **Calcite**: dim400 produces more leakage than dim200 (avg +10.3pp)

The direction is consistent within each project (no pair reverses).

### Interpretation

The opposite direction likely relates to the proportion of genuinely unchanged files between versions:

**Django (Python, ~42% source code identity between consecutive versions):** Most files have real edits. In a compressed 200-dim space, the model lacks capacity to encode fine-grained differences, so edited files still land close to their previous version, exceeding the 0.99 threshold. With 400 dims, the model can represent subtle differences, and edited files drop below 0.99.

**Calcite (Java, ~81% source code identity between consecutive versions):** Most files are genuinely unchanged. Higher dimensionality gives the model more capacity to faithfully represent truly identical code as near-identical vectors. The extra dimensions don't spread identical files apart — they make the representation more precise, pushing genuinely unchanged files closer to 1.0 and above 0.99.

In short: for codebases with high source identity (Calcite), more dimensions = better at detecting real duplicates. For codebases with lower identity (Django), more dimensions = better at distinguishing edited files from unchanged ones.

### Calcite cumulative-fresh: direction reverses mid-way

An interesting detail in Calcite cumulative-fresh: the dimension effect reverses around pair 8. Early pairs (1-7) show dim400 > dim200 (+0.9 to +13.1pp), but later pairs (8-15) show dim200 > dim400 (-0.1 to -3.3pp). As the cumulative training window grows, the 400-dim model accumulates more vocabulary and weight updates, and its advantage in detecting unchanged files erodes. The overall average (+2.43pp) is small because the two effects nearly cancel out.

This pattern is unique to cumulative-fresh — pairwise shows a consistent dim400 > dim200 direction across all 15 Calcite pairs.

### Collision pairs

Collision counts (near-duplicates between files with different paths) remain low and stable across all configurations, confirming that collisions are not a dimensionality artifact.

### Implications

1. A fixed threshold of 0.99 means different things depending on dimensionality and codebase characteristics.
2. The "right" embedding dimension is not universal — it depends on the codebase's code churn rate.
3. Threshold sensitivity analysis across dimensions is needed to establish comparable results.
4. The cumulative-fresh direction reversal in Calcite suggests that the training window size interacts with dimensionality — another reason to prefer pairwise as the most controlled experimental setup.
