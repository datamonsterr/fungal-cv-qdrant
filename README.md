# Myco Fungi Multi-Research Workflow

This repository now follows a multi-experiment autoresearch-style layout.
Core code remains under src/, with per-experiment checks colocated under each experiment folder.

## High-Level Workflow

1. Put raw data in Dataset/original/
2. Run prepare bootstrap
3. Run one or more experiment programs
4. Run immutable checks from each experiment package
5. Generate analysis visualizations
6. Produce per-experiment LaTeX reports

## Structure

- src/prepare: initialization pipeline and data/qdrant checks
- src/experiments: experiment implementations (preprocessing, feature extraction, finetune_dl, cross_validation, etc.)
- src/utils: reusable helper modules and unified uploader
- src/experiments/*/check.py: concise immutable targets colocated with each experiment
- src/analysis: visualization and analysis scripts
- report: archived markdown reports; new experiments should generate LaTeX reports

## Canonical Commands

### 1) Full Preparation (Dataset/original -> Qdrant)

```bash
uv run python -m src.prepare.init --collection myco_fungi_features_full
```

### 2) Generate Mapping Only

```bash
uv run python -m src.utils.generate_strain_mapping
```

### 3) Extract Features Only

```bash
uv run python -m src.experiments.feature_extraction.generate_features
```

### 4) Unified Upload (single JSON)

```bash
uv run python -m src.utils.upload_qdrant \
  --features-json Dataset/segmented_features.json \
  --metadata-json Dataset/segmented_image_metadata.json \
  --collection myco_fungi_features_full
```

### 5) Cross Validation

```bash
uv run python -m src.experiments.cross_validation.run --collection myco_fungi_features_full_finetuned
uv run python -m src.experiments.cross_validation.visualize
```

### 6) Finetune DL

```bash
uv run python -m src.experiments.finetune_dl.train_models
```

### 7) Lint / Type Check

```bash
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src
```

## Colab

Use src/colab_config.py to print a one-cell setup snippet:

```bash
uv run python -m src.colab_config
```

Colab assets are under src/experiments/finetune_dl/colab/.

## Notes

- src/main.py was removed by design.
- Qdrant named vectors must still match extractor names.
- New experiments should include program.md and a colocated check.py target.
