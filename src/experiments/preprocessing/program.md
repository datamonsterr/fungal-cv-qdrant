# preprocessing

## Objective
Preprocess dish images and produce segmented colony candidates.

## Entry Points
- `src.experiments.preprocessing.preprocess`
- `src.experiments.preprocessing.kmeans`

## Inputs
- Raw dataset from `Dataset/original/`

## Outputs
- Processed images and segment bounding boxes for downstream extraction.

## Check Target
Use `src/check/preprocessing_check.py` once created to enforce metadata/image quality checks.
