# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**Task:** Myco fungi species classification from colony images.
**Target:** Classify 5 *Penicillium* species (*P. citreonigrum*, *P. commune*, *P. crustosum*, *P. expansum*, *P. chrysogenum*) grown across multiple environments (growth media: MEA, CYA, DG18, etc.).

**System I/O:**
- **Input:** Plate images showing fungal colonies (1–3 colonies per plate)
- **Output:** A ranked list of species with similarity scores

---

## Methodology

### Pipeline

```
Dataset/original/
  1. Reformat           → Dataset/full_image/ + Dataset/segmented_image/ + *_metadata.json
  2. Generate mapping   → Dataset/strain_to_specy.csv
  3. Extract features   → Dataset/segmented_features.json
  4. Upload to Qdrant  → Qdrant collection (named vectors per extractor)

At query time:
  → Preprocess (crop petri dish)
  → Segment colonies (KMeans / Contour → 1–3 segments per image)
  → Extract features (same extractor as uploaded)
  → Qdrant KNN retrieval (k neighbours, E1/E2/E3/E4 strategy)
  → Filter siblings (exclude same-plate neighbours)
  → Aggregate scores (weighted / uni → species ranking)
```

### Key Concepts

| Term | Definition |
|------|-----------|
| **Strain** | A specific fungal isolate (e.g., `DTO 123-A1`). Unit of train/test split. |
| **Species** | Taxonomic target (5 *Penicillium* species). One species may have multiple strains. |
| **Colony** | A single fungal growth on a plate. One plate has 1–3 colonies. |
| **Environment** | Growth medium (MEA, CYA, DG18). Each strain is cultured on multiple media. |

### Environment Strategies

| Strategy | Description |
|----------|-------------|
| **E1** (same env) | Query images are matched only against neighbours from the **same growth medium**. This is the default and recommended strategy for controlled lab retrieval. |
| **E2** (all env) | Query images are matched against neighbours from **all available growth media** — maximum recall when medium is unknown or mixed. |
| **E3\_\<ENV\>** | Only a specific medium is used (e.g., `E3_MEA`). For evaluating single-medium performance. |
| **E4\_\<ENV\>** | All media **except** one are used (e.g., `E4_CYA`). For testing cross-medium generalization. |

### Aggregation Strategies

| Strategy | Description |
|----------|-------------|
| **weighted** (score-weighted) | Each neighbour votes for a species with a weight proportional to its cosine similarity. The species with the highest total weighted score wins. **First-ranked species is the prediction.** |
| **uni** (uniform) | Each neighbour contributes an equal vote (1/k) to its species regardless of similarity score. Useful when scores are unreliable. |

See [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md) for full definitions.

---

## Architecture

### Directory Layout

```
src/
├── config.py              # Paths, QDRANT_URL, collection names, image size (256×256)
├── main.py                # (legacy CLI — prefer direct module invocation)
├── prepare/               # Bootstrap: raw Dataset → Qdrant-ready
│   ├── init.py            # Main entry: runs full prepare pipeline
│   ├── checks.py          # check_dataset_root, check_metadata_exists, check_qdrant
│   └── init_yolo.py       # YOLO-based initialization
├── experiments/           # Each experiment has run.py + program.md
│   ├── cross_validation/  # 5-fold CV runner + visualize
│   ├── finetune_dl/       # Colab notebooks for training (train_models.py)
│   ├── kmeans_segmentation/
│   └── retrieval/         # KNN retrieval + ensemble strategies
├── feature_extraction/    # Feature extractor implementations + generate_features.py (prepare pipeline)
├── utils/                 # Shared: upload_qdrant, qdrant_query, reformat_dataset, etc.
├── analysis/              # Post-experiment analysis and visualizations
│   ├── retrieval/
│   └── visualization/
└── lib/                   # Shared cross-experiment code (cross-validation, metrics)
```

### Qdrant Collections

| Collection | Contents |
|---|---|
| `myco_fungi_features_full` | Standard 7 feature types (HOG, Gabor, color histograms, ResNet50, MobileNetV2, EfficientNetB1) |
| `myco_fungi_features_full_finetuned` | All standard + finetuned DL features |

Named vector keys must match `extractor.name`. Data persists in `./qdrant_storage/`.

### Feature Extractors

| Type | Extractors | Dims |
|------|-----------|------|
| Hand-crafted | `hog`, `gabor`, `colorhistogram`, `colorhistogramhs` | ~2880, 32, 96, 64 |
| DL (ImageNet) | `resnet50`, `mobilenetv2`, `efficientnetb1` | 2048, 1280, 1280 |
| DL (finetuned) | `ResNet50_finetuned`, `MobileNetV2_finetuned`, `EfficientNetB1_finetuned`, `EfficientNetB1_triplet` | same as above |
| ViT | `vit256_dino`, `cellvit_x20/x40`, `sam_vit_b/l/h` | 768 |

Finetuned weights: `weights/*.pth`. ViT weights: `pretrained/`.

---

## Environment Setup

```bash
uv sync                          # install dependencies
source .venv/bin/activate        # activate venv (use activate.fish for fish shell)
# or with Nix:
nix-shell -r "zsh" && source .venv/bin/activate
```

When adding dependencies, update `requirements.txt` (in addition to `pyproject.toml`).

---

## Commands

```bash
# Qdrant (required for DB operations)
docker compose up -d
docker compose down

# Prepare pipeline
uv run python -m src.prepare.init --collection myco_fungi_features_full
uv run python -m src.prepare.init_yolo

# Feature extraction
uv run python -m src.experiments.feature_extraction.generate_features

# Upload to Qdrant
uv run python -m src.utils.upload_qdrant \
  --features-json Dataset/segmented_features.json \
  --metadata-json Dataset/segmented_image_metadata.json \
  --collection myco_fungi_features_full

# Evaluate
uv run python -m src.experiments.retrieval.run comprehensive \
  --extractors efficientnetb1_finetuned \
  --env_strategies E1 E2 \
  --agg_strategies weighted uni \
  --k 7

# Cross-validation
uv run python -m src.experiments.cross_validation.run \
  --collection myco_fungi_features_full_finetuned \
  --extractor efficientnetb1_finetuned

uv run python -m src.experiments.cross_validation.visualize

# Experiment workflow (autoresearch pattern)
uv run python src/run.py --experiment <name>   # run.py: run one experiment, return accuracy
uv run python src/prepare.py --experiment <name>  # prepare.py: prepare + check in one step

# Lint / format
uv run black src && uv run isort src && uv run flake8 src && uv run mypy src
```

---

## Cross-Validation

- **Fixed extractor:** EfficientNetB1_finetuned
- **Collection:** `myco_fungi_features_full_finetuned`
- **K values:** 3, 5, 7, 9, 11
- **Env strategies:** E1 (same env), E2 (all env)
- **Agg strategies:** weighted (score-weighted), uni (uniform count)
- **Split:** one strain per species held out (round-robin across 5 folds)
- **Accuracy:** first-ranked species == ground truth species

Results are appended to `report/week_1_2/cv_results.csv` (safe to interrupt/resume).
Shared cross-validation logic is in `src/lib/cross_validation.py`.

---

## Experiment Workflow (autoresearch pattern)

1. **Create experiment branch:** `git checkout -b autoresearch/{experiment-name}/1-initial-description`
2. **Implement change** in `src/experiments/{experiment-name}/`
3. **Run:** `uv run python src/run.py --experiment {experiment-name}` → returns accuracy number
4. **Visualize:** `src/run.py` auto-plots to `results/autoresearch/{experiment-name}.png`
   - Green circle = new best (kept checkpoint)
   - Gray dot = discarded (worse than latest best)
   - Staircase green line = running best trajectory
5. **Merge best:** when a run beats the previous best, merge to `autoresearch/{experiment-name}` (no attempt suffix)
6. **Branch naming:** `autoresearch/{experiment-name}/{N}-{summary}` for each attempt

---

## Key Conventions

- **Metadata schema:** base records have `id` + nested `data` (`strain`, `environment`, `angle`, `specy`). Segmented records add `parent_id`, `segment_index`, `bbox`.
- **Qdrant named vector keys** must be consistent between feature generation, upload, and query (`using=...`). Keys are `extractor.name`.
- **Train/test split:** one strain per species held out (second alphabetical strain if ≥2 exist), marked `Test=True` in `Dataset/strain_to_specy.csv`.
- `species_weights.json` — per-species manual ensemble weights.
- Use **Pydantic models** for structured data contracts (see `src/analysis/retrieval/ensemble_analysis.py`).

---

## Documentation

| File | Topic |
|------|-------|
| [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md) | E1/E2/E3/E4 strategies, aggregation, domain terms |
| [`docs/FINETUNED_FEATURES.md`](docs/FINETUNED_FEATURES.md) | Finetuned model workflow |
| [`docs/VIT_FEATURE_EXTRACTION.md`](docs/VIT_FEATURE_EXTRACTION.md) | ViT model setup |
| [`docs/ENSEMBLE_STRATEGY.md`](docs/ENSEMBLE_STRATEGY.md) | Aggregation strategies and tuning |
| [`colab/TRAINING.md`](colab/TRAINING.md) | Colab training (ResNet50, MobileNetV2, EfficientNetB1) |
| [`colab/VIT_NOTES.md`](colab/VIT_NOTES.md) | ViT analysis, augmentation, TPU setup |

---

## Skills (`.claude/skills/`)

Skills are symlinked from `.agents/skills/` and managed via `npx skills`:

| Skill | When to Use |
|-------|-------------|
| `qdrant-vector-search` | Qdrant queries, collections, hybrid search |
| `deep-learning-pytorch` | PyTorch model development, training loops |
| `pytorch-lightning` | Distributed training, callbacks, LR scheduling |
| `scikit-learn` | Evaluation metrics, cross-validation, preprocessing |
| `image-processing` | Pillow-based image ops |
| `mermaid` / `mermaid-diagrams` | Diagrams in docs |
| `docker-expert` / `multi-stage-dockerfile` | Docker / Compose setup |
| `find-skills` | Discover and install new skills |
| `create-new-experiment` | Scaffold new experiment in `src/experiments/` |
