Run a new threshold experiment attempt and record the result.

Assumes Step 2 retrieval CSV already exists at `results/threshold/diverse_retrieval_results.csv`.
If not, run `/threshold-setup` first.

## Workflow for each new attempt

1. **Read previous log** to know what was already tried:
   ```bash
   cat results/threshold/log/experiments.log
   cat results/threshold/log/best_strategy.json
   cat results/autoresearch/threshold.csv
   ```

2. **Create a new branch** for the attempt:
   ```bash
   git checkout -b threshold/{N}-{summary}
   ```

3. **Make the change** in:
   - `src/experiments/threshold/threshold_analysis.py` — new formulas or algorithms
   - `src/experiments/threshold/expanded_threshold_analysis.py` — additional formula variants
   - `src/experiments/threshold/retrieve_with_train_filter.py` — retrieval config (k, extractor, etc.)

4. **Re-run analysis** and record result:
   ```bash
   uv run python -m src.experiments.threshold.expanded_threshold_analysis
   uv run python src/run.py --experiment threshold --description "what changed"
   ```

5. **If new best F1** → merge to `threshold/`:
   ```bash
   git checkout threshold
   git merge threshold/{N}-{summary}
   ```
   If not → leave the branch as historical record.

## Run analysis only (no autoresearch tracking)

```bash
uv run python -m src.experiments.threshold.threshold_analysis
# or extended:
uv run python -m src.experiments.threshold.expanded_threshold_analysis
```

## Re-retrieve (if retrieval config changed)

```bash
uv run python -m src.experiments.threshold.retrieve_with_train_filter
# resume:
uv run python -m src.experiments.threshold.retrieve_with_train_filter --resume
```

## Check current best result

```bash
cat results/autoresearch/threshold.csv
cat results/threshold/log/best_strategy.json
```

## Metric

Primary: **F1 score** (balances TP known-species acceptance vs FP unknown-species acceptance).
Secondary: AUROC (in `results/threshold/log/all_experiments.csv`).

## Experiments logging

Each run appends to `results/threshold/log/`. The staircase chart plots every individual
experiment (formula × algorithm pair) as a dot:
- **Gray dots**: discarded (F1 ≤ running best)
- **Green dots**: new running best at that point (staircase step-up)
- **Green staircase line**: horizontal segments connecting green dots in chronological order

## Run up to 4 parallel subagents

When exploring multiple strategies simultaneously, dispatch up to 4 subagents,
each on their own git worktree, each testing a different set of formulas.
Merge the best result back to `threshold/`.
