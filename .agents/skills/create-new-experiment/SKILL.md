---
name: create-new-experiment
description: "Bootstrap a new autoresearch experiment folder. Use when user asks to create or initialize a new experiment."
version: 2.0.0
author: project
---

# Create New Experiment

## Use When

- User asks to create a new experiment
- User asks to add a new research track or use case
- User asks for a reproducible experiment scaffold

## Before Creating — Ask These Questions

Ask the user **before** scaffolding. Do NOT create files until you have answers:

1. **Experiment name?** (e.g., `segmentation`, `feature-extractor`, `embedding-lr`)
2. **What is the independent variable?** (What are you changing between attempts?)
3. **What is the accuracy metric?** (How will you measure improvement? Same-environment retrieval accuracy? F1? Segmentation IoU?)
4. **Which existing components does this experiment use?**
   - Same-environment KNN retrieval (E1 strategy, EfficientNetB1_finetuned, k=11)?
   - Feature extraction from scratch?
   - Preprocessing/segmentation changes?
   - Something else?
5. **Branch name?** (auto-generated from experiment name if not specified)
6. **Is this experiment ready to start, or is it exploratory?**
7. **What is the environment strategy?** (`E1` = same medium, `E2` = all media — for retrieval experiments, this determines which environments are included in the candidate pool)
8. **What is the aggregation strategy?** (`weighted` = score-weighted, `uni` = uniform count — for retrieval experiments, this determines how neighbour scores are combined into a species prediction)
9. **What is the K value for KNN retrieval?** (Common values: 3, 5, 7, 9, 11 — defaults to 11 if not specified)
10. **Should this use the shared cross-validation library** (`src.lib.cross_validation.run_cross_validation`) **or is it custom?** (Most retrieval-based experiments should use the shared CV library for consistency)

## Branch Naming

If the user hasn't specified a branch, create it:
```bash
git checkout -b "autoresearch/{experiment-name}/1-initial-baseline"
```

Format: `autoresearch/{experiment-name}/{N}-{summary}`
- `{N}` starts at 1 and increments per attempt
- Merge best result back to `autoresearch/{experiment-name}` (no attempt suffix)

## What to Create

1. `src/experiments/<experiment_name>/`
2. `src/experiments/<experiment_name>/program.md`
3. `src/experiments/<experiment_name>/run_accuracy.py` (or integrate with `src/run.py`)

## program.md Template

```markdown
# {Experiment Name}

## Objective

What problem does this experiment solve? What are you measuring?

## Accuracy Metric

How is accuracy defined for this experiment?
- Metric: ...
- Baseline value (before change): ...
- Target: ...

## Entry Point

Run the experiment and record accuracy:
```bash
uv run python src/run.py --experiment {experiment_name} --description "your change"
```

Prepare with checks:
```bash
uv run python src/prepare.py --experiment {experiment_name}
```

## Experiment Configuration

- **Collection:** `myco_fungi_features_full_finetuned` (Qdrant)
- **Extractor:** `efficientnetb1_finetuned`
- **K:** 11
- **Strategy:** E1 (same environment), weighted by score
- **CV Folds:** 5

## Outputs

- Accuracy number (single float 0.0–1.0)
- Results saved to: `results/autoresearch/{experiment_name}.csv`
- Chart saved to: `results/autoresearch/{experiment_name}.png`

## Dependencies

What does this experiment depend on? (Dataset, Qdrant collection, segmentation method, etc.)
```

## run_accuracy Function

For retrieval-based experiments, implement `run_accuracy()` in the experiment module:

```python
"""
{Experiment Name} — run_accuracy()
Returns a single accuracy number (0.0–1.0).
"""

from src.lib.cross_validation import compute_mean_accuracy, run_cross_validation


def run_accuracy(
    collection: str = "myco_fungi_features_full_finetuned",
    extractor: str = "efficientnetb1_finetuned",
    k: int = 11,
    strategy: str = "weighted",
    environment: str = None,  # None = E1 (same environment)
    n_folds: int = 5,
) -> float:
    """
    Run the experiment and return a single accuracy number.

    This function is called by src/run.py when running this experiment.
    """
    results = run_cross_validation(
        collection_name=collection,
        extractor_key=extractor,
        k=k,
        environment=environment,
        strategy=strategy,
        n_folds=n_folds,
    )
    return compute_mean_accuracy(results)


if __name__ == "__main__":
    # Allow direct execution
    acc = run_accuracy()
    print(f"Accuracy: {acc:.4f}")
```

## Experiment Registry

After creating the experiment module, register it in `src/run.py` in the `EXPERIMENT_REGISTRY` dict:

```python
EXPERIMENT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ... existing entries ...
    "{experiment_name}": {
        "module": "src.experiments.{experiment_name}",
        "description": "One-line description",
        "default_params": {
            "collection": "myco_fungi_features_full_finetuned",
            "extractor": "efficientnetb1_finetuned",
            "k": 11,
            "strategy": "weighted",
            "environment": None,
            "n_folds": 5,
        },
    },
}
```

## Prepare + Check

For prerequisite validation before running:
```bash
uv run python src/prepare.py --experiment {experiment_name}
```

This runs:
1. Dataset root check
2. Qdrant connection check
3. Metadata existence check
4. Experiment-specific checks
5. Then runs `src/run.py`

## Plot — Staircase Chart Conventions

The autoresearch plot (`results/autoresearch/{experiment}.png`) follows these rules:

**Single-float accuracy per attempt** (most experiments):
- X-axis: experiment attempt number
- Y-axis: accuracy (0.0–1.0)
- Gray dots: discarded results (worse than running best)
- Green circles: new best at that attempt
- Staircase green line: horizontal segments at each new best level

**Multi-experiment experiments** (e.g. threshold — many individual formula × algorithm pairs):
- When `accuracy` is a JSON dict (multiple strategy-algorithm F1s per attempt), ALL individual
  experiments are plotted as dots, NOT colored lines.
- X-axis: experiment index (each formula × algorithm = one dot)
- Gray dots: f1 ≤ running best (discarded)
- Green dots: f1 > running best (new staircase step-up)
- Green staircase: horizontal segments connecting green dots in chronological order
- Labels on green dots: `{formula}_{algorithm}` (truncated to 25 chars)
- The threshold experiment reads from `results/{experiment}/log/all_experiments.csv`

**Rules for `ExperimentHistory.add()`**:
- Compare with `round(value, 6)` to avoid floating-point noise
- `kept="1"` only when the new accuracy beats the previous best (strict `>`)
- When accuracy is a dict, use `max(dict.values())` for the "best accuracy" comparison
- `accuracy` field in CSV stores: plain float string (e.g. `"0.089965"`) OR JSON dict string (e.g. `'{"geom_mean_top3_roc_opt": 0.164}'`)

**Mixed historical rows**: If some rows are plain float and others are dicts, the plotting code handles both — dict rows contribute their per-key values, plain float rows are skipped for keys they don't have.

## File Checklist

- [ ] `src/experiments/{name}/program.md`
- [ ] `src/experiments/{name}/run_accuracy.py` (if custom logic needed)
- [ ] Entry in `EXPERIMENT_REGISTRY` in `src/run.py`
- [ ] Git branch created: `autoresearch/{name}/1-initial-baseline`
