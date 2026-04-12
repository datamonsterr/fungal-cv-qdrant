# kmeans_segmentation

## Objective
Run K-means segmentation experiments and inspect contour quality.

## Entry Point
- Run: `uv run python -m src.experiments.kmeans_segmentation.run`

## Inputs
- Source images from `Dataset/original/`

## Outputs
- Segmentation debug plots and intermediate artifacts in local output folders.

## Check Target
Use `src/experiments/kmeans_segmentation/check.py` once created to lock expected quality thresholds.
