# feature_extraction

## Objective
Generate named-vector features for all enabled extractors.

## Entry Point
- Run: `uv run python -m src.experiments.feature_extraction.generate_features`

## Inputs
- `Dataset/segmented_image/`
- `Dataset/segmented_image_metadata.json`

## Outputs
- `Dataset/segmented_features.json`

## Check Target
Use `src/experiments/feature_extraction/check.py` for vector-shape and coverage checks.
