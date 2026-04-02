Set up and run the threshold-based unknown species detection experiment.

## What this experiment does

Tests formula variants (using s0..s4 neighbour scores) to find a threshold that separates
**known** test-strain species (7 Penicillium species held out from Qdrant) from **unknown**
species (all other images in diverse_data).

- **Retrieval**: EfficientNetB1_finetuned + E1 (same environment) + k=11 + weighted aggregation
- **Test set**: 861 images — 210 known (7 species × 30 images), 651 unknown
- **Known species**: DTO 217-D9 (neoechinulatum), DTO 470-I9 (tricolor), DTO 158-D1 (melanoconidium),
  DTO 148-D1 (polonicum), DTO 469-I5 (aurantiogriseum), DTO 469-I4 (freii), DTO 163-I2 (viridicatum)
- **Algorithms**: f1_grid, roc_opt, otsu (3 per formula — NO fpr-based algorithms)

## Prerequisites

```bash
# 1. Qdrant must be running with the finetuned collection populated
docker compose up -d

# 2. Test strain images must be in Qdrant-excluded set
#    (run prepare_test_strains.py if not already done)
```

## Step 1 — Copy test strain images + build retrieval list (one-time)

```bash
uv run python -m src.experiments.threshold.prepare_test_strains
# Outputs:
#   Dataset/diverse_data/images/{species}/{env}/  (copied test strain segments)
#   results/threshold/test_strain_retrieval_list.json
```

## Step 2 — Retrieve scores (~10–20 min)

```bash
uv run python -m src.experiments.threshold.retrieve_with_train_filter
# Outputs: results/threshold/diverse_retrieval_results.csv

# Resume if interrupted:
uv run python -m src.experiments.threshold.retrieve_with_train_filter --resume
```

## Step 3 — Run threshold analysis + visualize

```bash
# 162 formulas × 3 algorithms = 486 experiments
uv run python -m src.experiments.threshold.threshold_analysis

# Outputs:
#   results/threshold/threshold_analysis.csv   (best F1 per strategy × algorithm)
#   results/threshold/log/all_experiments.csv   (ALL individual experiments)
#   results/threshold/threshold_curves.png     (F1 vs t curves per formula)
#   results/threshold/roc_curves.png           (ROC curves per algorithm)
#   results/threshold/confusion_matrices.png   (confusion matrices at optimal t)
```

To generate 800+ formula variants (extended search):
```bash
uv run python -m src.experiments.threshold.expanded_threshold_analysis
```

## Step 4 — Record as autoresearch attempt

```bash
uv run python src/run.py --experiment threshold --description "describe what changed"
# Returns best F1 across all strategies. New best → kept checkpoint.
```

## Branch naming

```bash
git checkout -b threshold/{N}-{summary}
# e.g. threshold/2-gap-strategy, threshold/3-otsu-tuning
```

Merge winning branches back to `threshold/` (no attempt suffix).

## Threshold algorithms

| Algorithm | Description |
|-----------|-------------|
| `f1_grid` | Sweep 500 threshold candidates, pick argmax F1 |
| `roc_opt` | Maximise Youden's J (sensitivity + specificity − 1) |
| `otsu` | Minimise intra-class variance |

## Formula naming conventions

| Prefix | Meaning |
|--------|---------|
| `gap_{i}_{j}` | s{i} − s{j} |
| `gnorm_{i}_{j}` | (s{i} − s{j}) / (s{i} + s{j}) |
| `ratio_{i}_{j}` | s{i} / s{j} |
| `avg_top{k}` | Mean of top-k scores |
| `gm_top{k}` | Geometric mean of top-k |
| `ne_top{k}` | Normalised entropy of top-k |

## Key files

- `src/experiments/threshold/program.md` — full design doc
- `src/experiments/threshold/retrieve_with_train_filter.py` — Qdrant retrieval with test strain exclusion
- `src/experiments/threshold/prepare_test_strains.py` — copy test strain images + build retrieval list
- `src/experiments/threshold/threshold_analysis.py` — 3-algorithm threshold strategies + plots
- `src/experiments/threshold/expanded_threshold_analysis.py` — 100+ formula variants
- `src/run.py` — autoresearch entry point (returns F1, plots staircase)
