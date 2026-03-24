# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Myco fungi species classification from colony images. Uses multiple feature extractors (hand-crafted + deep learning) with Qdrant as the vector database for k-nearest-neighbour retrieval to identify 5 *Penicillium* species across 7 growth media.

## Environment Setup

```bash
uv sync                          # install dependencies
source .venv/bin/activate        # activate venv
# or with Nix:
nix-shell -r "zsh" && source .venv/bin/activate
```

When adding dependencies, update `requirements.txt` (in addition to `pyproject.toml`).

## Commands

All pipeline stages run through `src/main.py`. Qdrant must be running for `extract`, `upload-finetuned`, `evaluate`, `cross-validate`, and `predict`.

```bash
# Qdrant (required for DB operations)
docker compose up -d
docker compose down

# Check CLI
uv run python src/main.py --help

# Pipeline (run in order for fresh setup)
uv run python src/main.py reformat
uv run python src/main.py generate-mapping
uv run python src/main.py extract
uv run python src/main.py extract-finetuned    # requires weights/*.pth files
uv run python src/main.py upload-finetuned
uv run python src/main.py train

# Evaluate
uv run python src/main.py evaluate \
  --extractor efficientnetb1_finetuned --k 7 \
  --strategy weighted --environment all \
  --collection myco_fungi_features_full_finetuned
uv run python src/main.py evaluate-all --k 7 --collection myco_fungi_features_full_finetuned

# Cross-validation (100 runs, resumable)
uv run python src/main.py cross-validate --collection myco_fungi_features_full_finetuned
uv run python src/main.py cross-validate-visualize

# Predict
uv run python src/main.py predict --strain "DTO 123-A1" --extractor resnet50 --k 5
uv run python src/main.py predict-new --path /path/to/new_strain --extractor efficientnetb1_finetuned --k 7

# Lint / format
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src

# Tests (none yet)
uv run pytest
```

### Slash Commands (`.claude/commands/`)

| Command | Purpose |
|---------|---------|
| `/lint` | Run black + isort + flake8 + mypy |
| `/evaluate` | Standard evaluation with finetuned collection |
| `/db` | Start / stop Qdrant |
| `/cross-validate` | Run or resume cross-validation |

## Architecture

### Data Flow

```
Dataset/original/
  → reformat          → Dataset/full_image/ + Dataset/segmented_image/ + *_metadata.json
  → generate-mapping  → Dataset/strain_to_specy.csv
  → extract           → Dataset/segmented_features.json → Qdrant collection
  → extract-finetuned → Dataset/finetuned_dl_features.json
  → upload-finetuned  → Qdrant collection (finetuned vectors added)
  → evaluate          → results/run_<timestamp>_k<K>/
```

### Key Modules

| File | Role |
|------|------|
| `src/config.py` | Single source of truth: paths, `QDRANT_URL`, collection names, image size (256×256) |
| `src/main.py` | argparse CLI wiring all subcommands |
| `src/feature_extraction/feature_extractors.py` | All extractor classes; `extractor.name` is the Qdrant named-vector key |
| `src/database/upload_qdrant.py` | Recreates Qdrant collection and uploads points with named vectors + payload |
| `src/database/query_utils.py` | Similarity search helpers |
| `src/classification/prediction.py` | Core KNN prediction; aggregates neighbour votes to species |
| `src/classification/evaluate_species.py` | Evaluation loop, CSV output, confusion matrix |
| `src/preprocessing/preprocess.py` | Petri dish crop |
| `src/preprocessing/kmeans.py` | K-means colony segmentation |
| `src/scripts/` | One-off scripts: finetuned features, cross-validation, fold mapping |
| `src/experiments/` | Ensemble analysis, comprehensive reports, complementary case visualization |
| `colab/` | Google Colab training scripts (ResNet50 / MobileNetV2 / EfficientNetB1, triplet loss, CellViT) |

### Qdrant Collections

| Collection | Contents |
|---|---|
| `myco_fungi_features_full` | Standard 7 feature types (HOG, Gabor, color histograms, ResNet50, MobileNetV2, EfficientNetB1) |
| `myco_fungi_features_full_finetuned` | All standard + finetuned DL features |

Data persists in `./qdrant_storage/` across container restarts.

### Feature Extractors

| Type | Extractors | Dims |
|------|-----------|------|
| Hand-crafted | `hog`, `gabor`, `colorhistogram`, `colorhistogramhs` | ~2880, 32, 96, 64 |
| DL (ImageNet) | `resnet50`, `mobilenetv2`, `efficientnetb1` | 2048, 1280, 1280 |
| DL (finetuned) | `ResNet50_finetuned`, `MobileNetV2_finetuned`, `EfficientNetB1_finetuned`, `EfficientNetB1_triplet` | same as above |
| ViT | `vit256_dino`, `cellvit_x20/x40`, `sam_vit_b/l/h` | 768 |

Finetuned weights: `weights/*.pth`. ViT weights: `pretrained/`.

## Key Conventions

- **Metadata schema**: base records have `id` + nested `data` (`strain`, `environment`, `angle`, `specy`). Segmented records add `parent_id`, `segment_index`, `bbox`.
- **Qdrant named vector keys** must be consistent between feature generation, upload, and query (`using=...`). Keys are `extractor.name` (or lowercased variants).
- **Train/test split**: one strain per species held out (second alphabetical strain if ≥2 exist), marked `Test=True` in `Dataset/strain_to_specy.csv`.
- **Cross-validation** results append to CSV after each combination — safe to interrupt and resume.
- `QDRANT_URL` can be overridden via environment variable (default `http://localhost:6333`).
- Use **Pydantic models** for structured data contracts (see `src/experiments/ensemble_analysis.py`).
- `species_weights.json` — per-species manual ensemble weights.

## Evaluation Terminology

- **Strategy**: `weighted`/`score`/`avg` = score-weighted; `uni` = uniform-count
- **E1** (same env): neighbours from same growth medium only
- **E2** (all env): neighbours from all growth media
- **E3\_`<ENV>`**: query images from one specific medium
- **E4\_`<ENV>`**: exclude one medium from the candidate pool

See [`docs/ENSEMBLE_STRATEGY.md`](docs/ENSEMBLE_STRATEGY.md) for weight math and manual tuning.

## Skills (`.claude/skills/`)

Project skills are symlinked from `.agents/skills/` and managed via `npx skills`:

| Skill | When to Use |
|-------|-------------|
| `qdrant-vector-search` | Qdrant queries, collections, hybrid search |
| `deep-learning-pytorch` | PyTorch model development, training loops |
| `pytorch-lightning` | Distributed training, callbacks, LR scheduling |
| `scikit-learn` | Evaluation metrics, cross-validation, preprocessing |
| `image-processing` | Pillow-based image ops |
| `mermaid` / `mermaid-diagrams` | Diagrams in docs |
| `docker-expert` / `multi-stage-dockerfile` | Docker / Compose setup |
| `find-skills` | Discover and install new skills via `npx skills` |

## Documentation (`docs/`)

| File | Topic |
|------|-------|
| [`FINETUNED_FEATURES.md`](docs/FINETUNED_FEATURES.md) | Finetuned model workflow (extract → upload → evaluate) |
| [`VIT_FEATURE_EXTRACTION.md`](docs/VIT_FEATURE_EXTRACTION.md) | ViT model setup and usage |
| [`ENSEMBLE_STRATEGY.md`](docs/ENSEMBLE_STRATEGY.md) | Aggregation strategies and per-species weight tuning |
