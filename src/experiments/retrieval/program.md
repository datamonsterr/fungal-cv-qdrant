# Retrieval Experiment Program

## Objective

Consolidate retrieval evaluation, comprehensive benchmark runs, and ensemble-model analysis under a single experiment package.

## Entrypoint

```bash
uv run python -m src.experiments.retrieval.run <command> [options]
```

## Commands

### 1) Comprehensive Retrieval Benchmark

Runs combinations of feature extractors, environment strategies, and aggregation strategies.

```bash
uv run python -m src.experiments.retrieval.run comprehensive \
  --identifier run_001 \
  --extractors resnet50 mobilenetv2 efficientnetv2 hog gabor colorhistogram \
  --env_strategies E1 E2 \
  --agg_strategies weighted uni \
  --k 5
```

### 2) Ensemble Analysis

Consumes comprehensive outputs and builds ensemble performance reports.

```bash
uv run python -m src.experiments.retrieval.run ensemble-analysis
```

### 3) Ensemble Report Rendering

Creates detailed charts for one strategy.

```bash
uv run python -m src.experiments.retrieval.run ensemble-report --strategy weighted
```

### 4) Strategy Comparison

Compares weighted and simple-average ensemble predictions.

```bash
uv run python -m src.experiments.retrieval.run ensemble-compare
```

## Outputs

- Retrieval metrics and per-strategy artifacts under `results/<identifier>/...`
- Ensemble artifacts under `results/ensemble_analysis/`
- CSV/ranked predictions and confusion matrices per run
