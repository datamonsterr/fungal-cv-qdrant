# finetune_dl

## Objective
Fine-tune DL backbones and export model weights for downstream feature extraction.

## Entry Point
- Run: `uv run python -m src.experiments.finetune_dl.train_models`

## Inputs
- Preprocessed dataset and train/test mapping CSV.

## Outputs
- Weights in `weights/`
- Training artifacts in `results/` or experiment-specific folders.

## Check Target
Use `src/experiments/finetune_dl/check.py` once created to enforce accuracy/loss targets.
