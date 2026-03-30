# CLAUDE.md

## Project Overview

Myco fungi species classification from colony images with multiple extractors and Qdrant KNN retrieval.
The repository now uses a multi-research architecture inspired by autoresearch workflows.

## Core Layout

- src/prepare
  - bootstrap from Dataset/original to metadata/features/upload
  - qdrant/dataset checks
- src/experiments
  - preprocessing
  - feature_extraction
  - finetune_dl
  - cross_validation
  - each experiment should own a local check.py contract
  - additional experiment folders, each with program.md
- src/utils
  - reusable scripts and unified upload_qdrant
- src/analysis
  - visualization and analysis scripts

## Commands

```bash
# Prepare all (Dataset/original -> qdrant)
uv run python -m src.prepare.init --collection myco_fungi_features_full

# Mapping
uv run python -m src.utils.generate_strain_mapping

# Feature extraction
uv run python -m src.experiments.feature_extraction.generate_features

# Upload from one JSON file
uv run python -m src.utils.upload_qdrant \
  --features-json Dataset/segmented_features.json \
  --metadata-json Dataset/segmented_image_metadata.json \
  --collection myco_fungi_features_full

# Finetune
uv run python -m src.experiments.finetune_dl.train_models

# Cross-validation
uv run python -m src.experiments.cross_validation.run --collection myco_fungi_features_full_finetuned
uv run python -m src.experiments.cross_validation.visualize
```

## Qdrant Conventions

- Collection naming remains configurable via src/config.py
- Named vector key must match extractor.name contract
- Unified uploader path: src/utils/upload_qdrant.py

## Reporting

- report/ markdown artifacts are archived history
- new experiments should produce LaTeX reports and keep content.md context files

## Lint / Format

```bash
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src
```
