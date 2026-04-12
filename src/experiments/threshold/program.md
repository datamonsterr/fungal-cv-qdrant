# Threshold — Unknown Species Detection via Score Formulas

## Objective

Determine whether a **formula** applied to the top-k neighbour similarity scores (s0, s1, ..., s4)
can reliably separate **known** test-strain species from **unknown** species.

The approach: for each query image, the KNN retrieval returns ranked neighbours with scores.
A formula maps these scores to a single "confidence" value. A threshold applied to this
value classifies the image as **known** (≥ t) or **unknown** (< t).

---

## Test Set

**Known** = 7 Penicillium species held out from Qdrant during retrieval (210 images):
DTO 217-D9 (neoechinulatum), DTO 470-I9 (tricolor), DTO 158-D1 (melanoconidium),
DTO 148-D1 (polonicum), DTO 469-I5 (aurantiogriseum), DTO 469-I4 (freii), DTO 163-I2 (viridicatum)

**Unknown** = all other diverse_data images (651 images, 44 species)

---

## Retrieval Configuration

- **Collection:** `myco_fungi_features_full_finetuned`
- **Extractor:** `EfficientNetB1_finetuned`
- **K:** 11 neighbours
- **Environment strategy:** E1 (same growth medium)
- **Aggregation:** weighted (score-weighted)

---

## Running the Experiment

### One-time setup (test strain images)

```bash
uv run python -m src.experiments.threshold.prepare_test_strains
```

### Retrieve scores (~10–20 min)

```bash
uv run python -m src.experiments.threshold.retrieve_with_train_filter
# Outputs: results/threshold/diverse_retrieval_results.csv
```

### Run threshold analysis + record

```bash
uv run python -m src.experiments.threshold.expanded_threshold_analysis
uv run python src/run.py --experiment threshold --description "describe the new formula type"
```

The `expanded_threshold_analysis` script generates ~50 new formula variants × 3 algorithms
and appends them to `log/all_experiments.csv`. The staircase chart (`results/autoresearch/threshold.png`)
shows all experiments as dots — green for new bests, gray for discarded — with a
horizontal staircase connecting running bests.

---

## Formula Design

A **formula** is a function `f(s0, s1, s2, s3, s4) → R` that maps neighbour scores to a scalar.

Each formula should be **qualitatively different** from existing ones — a new equation type,
not just a parameter tweak of a previous formula.

### Formula naming conventions

| Prefix | Meaning |
|--------|---------|
| `gap_{i}_{j}` | s{i} − s{j} |
| `gnorm_{i}_{j}` | (s{i} − s{j}) / (s{i} + s{j}) |
| `ratio_{i}_{j}` | s{i} / s{j} |
| `prod_{i}_{j}` | s{i} × s{j} |
| `avg_top{k}` | mean of top-k scores |
| `gm_top{k}` | geometric mean of top-k |
| `hm_top{k}` | harmonic mean of top-k |
| `ne_top{k}` | normalised entropy of top-k |
| `w_{name}` | custom weight vector |
| `exp_decay{N}` | exponential decay with base N |
| `hybrid_*` | blend of two formula types |
| `agree_top{k}` | 1 / (std of top-k + 0.01) |

### Adding a new formula

Edit `expanded_threshold_analysis.py` → `generate_formulas()`. Add a new entry:
```python
formulas["my_new_formula"] = <numpy expression using s[:, 0..4]>
```

Run `expanded_threshold_analysis` to evaluate and append to the log.

---

## Threshold-finding Algorithms

Only these 3 (no FPR-based algorithms):

| Algorithm | Description |
|-----------|-------------|
| `f1_grid` | Sweep 500 thresholds, pick argmax F1 |
| `roc_opt` | Maximise Youden's J (sensitivity + specificity − 1) |
| `otsu` | Minimise intra-class variance |

---

## Log Structure

```
results/threshold/log/
  all_experiments.csv    ← every formula × algorithm → f1, precision, recall, ...
  best_strategy.json     ← current overall best
  experiments.log       ← one line per attempt
  attempt_005.json       ← full detail for attempt N
```

### Staircase chart

`results/autoresearch/threshold.png`:
- **Gray dots**: experiments below the running best staircase
- **Green dots**: experiments that set a new running best (staircase step-up)
- **Green staircase**: horizontal segments connecting green dots
- Labels: `{formula}_{algorithm}` on each green dot

---

## Accuracy Metric

- **Primary:** F1 score for the "known" class (TP known accepted / all predicted known)
- **Secondary:** AUROC (threshold-independent)

Formula: F1 = 2 × precision × recall / (precision + recall), where:
- TP = known images with formula_score ≥ t
- FP = unknown images with formula_score ≥ t
- FN = known images with formula_score < t

---

## Experiment Strategy

**Do NOT** enumerate hundreds of trivial parameter variations (e.g., `s0 * 1.01`, `s0 * 1.02`).
Instead, each attempt should introduce **qualitatively new formula types**.

Good: "consecutive ratios", "inverse-variance agreement", "power-law normalised gap"
Bad: "s0 * 1.5", "s0 * 2.0", "s0 * 2.5" (same formula, different constant)

Each attempt should test ~50 diverse formulas × 3 algorithms = 150 experiments.
All experiments accumulate in `log/all_experiments.csv`.
The staircase shows which formula types consistently produce high F1.

---

## Branch Naming

```
threshold/{N}-{formula-type-summary}
```

Examples:
- `threshold/1-initial-50-formulas`
- `threshold/2-consecutive-ratios`
- `threshold/3-agreement-metrics`
