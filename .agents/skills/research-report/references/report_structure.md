# Report Section Templates

## Section 1: Overview

```markdown
## 1. Overview

### 1.1 Problem Introduction

Fungal species classification from colony images is challenging due to intra-species variation across growth environments and inter-species visual similarity. This system uses a **retrieval-based approach**: features are extracted from colony images and stored in a vector database (Qdrant); a new strain is identified by finding its nearest neighbours and aggregating their species labels.

**Why retrieval-based?**
- New strains can be added to the database without retraining
- Predictions are interpretable (you can inspect which reference images matched)
- Robust to environmental variation by controlling the neighbour pool

### 1.2 Glossary

| Term | Definition |
|------|-----------|
| Strain | A specific fungal isolate (e.g. DTO 148-D1) |
| Species | Taxonomic label to predict |
| Environment | Growth medium (MEA, CYA, YES, CYA25, DG18, OAT, IDF) |
| E1 (same env) | KNN restricted to same medium as query |
| E2 (all env) | KNN from all media |
| weighted | Score-weighted vote aggregation |
| uni | Uniform vote aggregation |
| K | Number of nearest neighbours |
| Fold | One CV split: one test strain per species |

### 1.3 Purpose of Cross-Validation

Cross-validation is used to **select the best hyperparameter configuration** (K, aggregation strategy, environment strategy) without overfitting to a single held-out set. By rotating which strain is the test strain across 5 folds, we get a robust estimate of how well each configuration generalises.
```

---

## Section 2: Methodology

```markdown
## 2. Methodology

### 2.1 Fold Design

The dataset contains **5 species with 4–7 strains each**. We use **strain-level 5-fold cross-validation**: in each fold, one strain per species is held out as the test set; the remaining strains form the reference database.

| Fold | Test Strains (one per species) |
|------|-------------------------------|
| 0 | ... |
| 1 | ... |
| 2 | ... |
| 3 | ... |
| 4 | ... |

### 2.2 Why Strain-Level Split?

The classification goal is to identify **novel strains** not seen during training. A random image-level split would let the model memorise images of test strains during training, making accuracy artificially high and failing to measure generalisation.

Strain-level splitting ensures:
- Training images and test images come from **different biological isolates**
- Performance reflects ability to generalise to unseen strains — the real-world use case

### 2.3 Configuration Space

| Hyperparameter | Values Tested |
|---------------|---------------|
| K | 3, 5, 7, 9, 11 |
| Aggregation strategy | weighted (score), uni (uniform) |
| Environment strategy | E1 (same), E2 (all) |
| **Total combinations** | 5 × 2 × 2 = **20 per fold** |

### 2.4 Method Diagram

![Pipeline Overview](../final_gr2/mermaid/query_flow.png)
```

---

## Section 3: Implementation

```markdown
## 3. Implementation

### 3.1 Vector Database Retrieval Pipeline

Each segmented colony image passes through a feature extractor (EfficientNetB1 finetuned) to produce a 1280-dimensional embedding, stored as a named vector in Qdrant. At query time, KNN retrieval returns the top-K most similar reference images, and their species labels are aggregated by vote.

![Preprocessing Flow](../final_gr2/mermaid/preprocessed_diagram_flow.png)

### 3.2 Cross-Validation Script Architecture

The cross-validation script (`src/scripts/cross_validation.py`) automates the full evaluation loop:

```mermaid
flowchart TD
    A[Load fold mapping CSVs\nDataset/strain_to_specy_fold*.csv] --> B
    B[For each fold 0–4] --> C
    C[Load fold-specific weights\nfold{idx}_EfficientNetB1_finetuned.pth] --> D
    D[Extract features for all images\nextract_finetuned_features] --> E
    E[Upload to fold Qdrant collection\nmyco_fungi_*_fold{idx}] --> F
    F[For each config\nK × strategy × env_strategy] --> G
    G[Run evaluate_species\nget per-species accuracy] --> H
    H[Append row to results CSV\nresults/cross_validation/results.csv] --> I
    I{All configs done?}
    I -- No --> F
    I -- Yes for this fold --> B
    B -- All folds done --> J
    J[Aggregate results across folds\nmean ± std per config] --> K
    K[Visualize\ncv_visualize.py → report/week_1_2/]
```

**Key functions:**

| Function | File | Purpose |
|----------|------|---------|
| `run_fold_evaluation` | `cross_validation.py` | Orchestrates one fold: extract → upload → evaluate all configs |
| `load_fold_mapping` | `cross_validation.py` | Reads `strain_to_specy_fold{idx}.csv` and sets Test flags |
| `evaluate_species` | `evaluate_species.py` | Runs KNN query and returns per-species accuracy for one config |
| `append_results` | `cross_validation.py` | Appends one result row to CSV (safe to interrupt) |
| `visualize_results` | `cv_visualize.py` | Generates comparison charts from aggregated CSV |
```

---

## Section 4: Results

```markdown
## 4. Results

### 4.1 Full Configuration Comparison

![All Configurations Accuracy](images/all_configs_accuracy.png)

| K | Strategy | Env | Mean Accuracy | Std |
|---|----------|-----|--------------|-----|
| 3 | weighted | E1 | X.XX% | ±X.X% |
| ... | ... | ... | ... | ... |

### 4.2 Best Configuration

**Best**: K=?, strategy=?, env=? → **X.X% mean accuracy**

![Best Config Analysis](images/best_config_detail.png)

Analysis: ...

### 4.3 Hyperparameter Sensitivity

**Effect of K**: ...

**Effect of strategy (weighted vs uni)**: ...

**Effect of environment strategy (E1 vs E2)**: ...
```

---

## Section 5: Conclusion

```markdown
## 5. Conclusion

**Key outcomes:**
- Best configuration: K=?, strategy=?, env_strategy=?
- Accuracy achieved: X.X% (mean across 5 folds)
- ...

**Practical recommendation:** Use `--k ? --strategy ? --environment ?` for production inference.

**Limitations and future work:**
- ...
```
