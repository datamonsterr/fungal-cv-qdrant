# Rule: Experiment Active Learning — Log, Read, Improve

## Core principle

Do NOT write a large procedural script that tries everything at once. Instead:
- Write SMALL, focused experiment files (< 10 lines each for strategy/result)
- Log every run to a shared structured log
- After each run, READ the log to understand what has been tried
- Use the log to actively decide what to try next (not re-run what already failed)

## Log structure

```
results/{experiment}/log/
  experiments.log         ← one line per attempt: "[N] ★ NEW BEST | best_f1=0.xxxx (strategy) | top3: ..."
  attempt_001.json          ← full dict of all strategy F1s for that attempt
  best_strategy.json       ← always current best: {"strategy": "...", "f1": 0.xxx, "threshold": ...}
  all_experiments.csv       ← every individual experiment: formula,algorithm,f1,precision,recall,...
  setup.json                ← shared config: dataset, k, extractor, env_strategy, n_known, n_unknown
  strategies/              ← individual strategy files tried
    geom_mean_top3_roc_opt.json
    gap_s0s2_f1_grid.json
```

## Common setup file

```json
{
  "dataset": "diverse_retrieval_results.csv",
  "retrieval": {"k": 11, "extractor": "EfficientNetB1_finetuned", "env_strategy": "E1"},
  "test_set": {"total": 861, "known": 210, "unknown": 651},
  "current_best": {"strategy": "rat01_p_rat12", "algorithm": "roc_opt", "f1": 0.4536, "attempt": 5}
}
```

## Each strategy file (< 10 lines JSON)

```json
{
  "formula": "(s0 * s1 * s2)^(1/3)",
  "algorithm": "f1_grid",
  "threshold": 0.302755,
  "f1": 0.163934,
  "precision": 0.125,
  "recall": 0.238,
  "tp": 5, "fp": 35, "tn": 595, "fn": 16,
  "attempt": 2
}
```

## Workflow per experiment run

1. **Before running**: Read `best_strategy.json` and `experiments.log`
   → Know current best, what has been tried, what failed
2. **Agent specifies experiment name**: a meaningful description, not auto-generated
   → e.g. "gm3 ceiling confirmed, try agreement + score hybrid"
3. **Run**: execute the experiment
4. **After running**: update `best_strategy.json` if new best; append to `experiments.log`
5. **Read the log before next run** to avoid repeating failed strategies

## Threshold experiment specifics

### Algorithms (only these 3 — no FPR-based)

| Algorithm | Description |
|-----------|-------------|
| `f1_grid` | Sweep 500 thresholds, pick argmax F1 |
| `roc_opt` | Maximise Youden's J (sensitivity + specificity − 1) |
| `otsu` | Minimise intra-class variance |

### Formula naming conventions

| Prefix | Meaning |
|--------|---------|
| `s{i}` | Raw score of i-th neighbour |
| `gap_{i}_{j}` | s{i} − s{j} |
| `gnorm_{i}_{j}` | (s{i} − s{j}) / (s{i} + s{j}) |
| `ratio_{i}_{j}` | s{i} / s{j} |
| `prod_{i}_{j}` | s{i} × s{j} |
| `avg_top{k}` | Mean of top-k scores |
| `gm_top{k}` | Geometric mean of top-k |
| `ne_top{k}` | Normalised entropy of top-k |
| `w_{name}` | Custom weight vector |
| `exp_decay{N}` | Exponential decay with base N |

## Do NOT

- Do NOT run 1000 experiments in a single script without logging each one
- Do NOT overwrite previous results without reading the log first
- Do NOT auto-generate experiment names — agent specifies them
- Do NOT create a new strategy file longer than 10 lines
- Do NOT use fpr_5pct or fpr_10pct algorithms — they perform poorly and are excluded
