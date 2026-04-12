# cross_validation

## Objective
Run strain-level cross-validation experiments and track resumable metrics.

## Entry Points
- Run: `uv run python -m src.experiments.cross_validation.run`
- Visualize: `uv run python -m src.experiments.cross_validation.visualize`

## Inputs
- Dataset metadata: `Dataset/segmented_image_metadata.json`
- Mapping: `Dataset/strain_to_specy.csv`
- Qdrant collection: configurable via CLI args in run script

## Outputs
- CSV metrics in `results/` and report artifacts under `report/`.

## Check Target
Use `src/experiments/cross_validation/check.py` to assert target metrics.
