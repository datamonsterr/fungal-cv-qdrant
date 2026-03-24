# Myco Fungi Feature Extraction and Classification

This project experiments with multiple feature extractor methods for classifying **Penicillium** fungi species from colony images using Computer Vision, Deep Learning (PyTorch), and Qdrant as the vector database. Species identification uses k-nearest-neighbour retrieval over named Qdrant vector collections with configurable aggregation and environment strategies.

## Species Covered

5 *Penicillium* species, each with 4–7 strains grown across 7 growth media (environments):

| Species | Example strain |
|---|---|
| *P. aurantiogriseum* | DTO 001-A1 |
| *P. cyclopium* | DTO 002-A1 |
| *P. freii* | DTO 003-A1 |
| *P. melanoconidium* | DTO 004-A1 |
| *P. viridicatum* | DTO 005-A1 |

Growth media (environments): `MEA`, `DG18`, `CREA`, `CYA`, `CYA30`, `CYAS`, `YES`

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose (for Qdrant vector database)
- Nix (optional, for reproducible shell environment)

## 1. Environment Setup

```bash
# Clone the repo, then install dependencies
uv sync

# Activate virtual environment (uv run works without this,
# but activate if running scripts directly)
source .venv/bin/activate
```

Using Nix:
```bash
nix-shell -r "zsh"   # or bash
source .venv/bin/activate
```

## 2. Dataset Download

Download the original dataset from Google Drive and place it at `Dataset/original/`:

**Dataset**: [Google Drive – fungal images](https://drive.google.com/drive/folders/1onJ4404wG65BbdrM9PBvESjfx68N2fs1?usp=sharing)

Expected structure after download:
```
Dataset/original/
├── DTO 001-A1 Penicillium aurantiogriseum/
│   ├── MEA/
│   │   ├── DTO001A1-MEA-ob.jpg
│   │   └── DTO001A1-MEA-rev.jpg
│   └── ...
└── DTO 002-A1 Penicillium cyclopium/
    └── ...
```

## 3. Setup Qdrant Vector Database

```bash
# Start Qdrant (required before extract / evaluate / predict)
docker compose up -d

# Verify it is running
docker compose ps

# Qdrant dashboard: http://localhost:6333/dashboard
# Stop when done
docker compose down
```

## 4. Full Pipeline

All steps are orchestrated via `src/main.py`.  Run in order for a fresh setup.

### Step 1 — Reformat Dataset

Preprocesses raw images (Petri dish crop), runs K-means segmentation to isolate colonies, and generates metadata JSON files.

```bash
uv run python src/main.py reformat
```

Output:
```
Dataset/full_image/                      ← copied & preprocessed originals
Dataset/segmented_image/                 ← cropped colony segments (256×256)
Dataset/full_image_metadata.json         ← id, strain, environment, angle, species
Dataset/segmented_image_metadata.json    ← + parent_id, segment_index, bbox
```

### Step 2 — Generate Strain Mapping

Scans the original dataset folders to build `strain_to_specy.csv`. Marks **one strain per species** as the hold-out test strain (second alphabetical strain if ≥2 exist).

```bash
uv run python src/main.py generate-mapping
```

Output: `Dataset/strain_to_specy.csv` — columns: Strain, Species, Test (31 strains, 7 test)

### Step 3 — Feature Extraction (Standard)

Extracts 7 feature types per segment and uploads them to Qdrant.

```bash
uv run python src/main.py extract
```

Output:
```
Dataset/segmented_features.json     ← feature vectors for all segments
Qdrant collection: myco_fungi_features_full
```

Feature types and vector dimensions:

| Extractor key | Type | Dimension |
|---|---|---|
| `hog` | Hand-crafted | ~2880 |
| `gabor` | Hand-crafted | 32 |
| `colorhistogram` | Hand-crafted | 96 |
| `colorhistogramhs` | Hand-crafted | 64 |
| `resnet50` | DL (ImageNet) | 2048 |
| `mobilenetv2` | DL (ImageNet) | 1280 |
| `efficientnetb1` | DL (ImageNet) | 1280 |

### Step 4 — Fine-tuned Model Weights (Optional)

Download the fine-tuned model weights from Google Drive:

**Weights**: [Google Drive – finetuned weights](https://drive.google.com/drive/folders/1QvmnfLz40vpr20eJoejeXw5Sc-aM47kG?usp=sharing)

Place `.pth` files into the `weights/` directory:
```
weights/
├── EfficientNetB1_finetuned.pth
├── MobileNetV2_finetuned.pth
├── ResNet50_finetuned.pth
└── EfficientNetB1_triplet.pth
```

These models were trained on the fungi dataset using `colab/train_models.py`.

**4a. Extract fine-tuned features:**
```bash
uv run python src/main.py extract-finetuned
```
Output: `Dataset/finetuned_dl_features.json`

**4b. Upload combined features (standard + fine-tuned) to Qdrant:**
```bash
uv run python src/main.py upload-finetuned
```
Output: Qdrant collection `myco_fungi_features_full_finetuned`

Fine-tuned extractors:

| Extractor key | Weights file | Dimension |
|---|---|---|
| `ResNet50_finetuned` | `ResNet50_finetuned.pth` | 2048 |
| `MobileNetV2_finetuned` | `MobileNetV2_finetuned.pth` | 1280 |
| `EfficientNetB1_finetuned` | `EfficientNetB1_finetuned.pth` | 1280 |
| `EfficientNetB1_triplet` | `EfficientNetB1_triplet.pth` | 1280 |

### Step 5 — Train Models

Fine-tunes ResNet50, MobileNetV2, and EfficientNetB1 on the training split (locally, no GPU required but recommended). For GPU/TPU cloud training see [`colab/TRAINING.md`](colab/TRAINING.md).

```bash
uv run python src/main.py train
```

Output:
```
weights/ResNet50_finetuned.pth
weights/MobileNetV2_finetuned.pth
weights/EfficientNetB1_finetuned.pth
weights/*_history.json     ← training curves
```

### Step 6 — Evaluation

Evaluates on the test strains (held-out one strain per species).

**Terminology**

| Term | Meaning |
|---|---|
| Strategy | How neighbor votes are aggregated: `weighted` (score-weighted) or `uni` (uniform count) |
| E1 (same env) | Query images use only neighbors from the **same** growth medium |
| E2 (all env) | Query images use neighbors from **all** growth media |
| E3\_`<env>` | Use only query images from a specific medium |
| E4\_`<env>` | Exclude one specific medium from the candidate pool |

**Single extractor, specific strategy:**
```bash
uv run python src/main.py evaluate \
    --extractor efficientnetb1_finetuned \
    --k 5 \
    --strategy weighted \
    --environment all \
    --collection myco_fungi_features_full_finetuned
```

**All extractors sweep:**
```bash
uv run python src/main.py evaluate --extractor all --k 7
```

**Full brute-force sweep (all extractors × all envs × all strategies):**
```bash
uv run python src/main.py evaluate-all --k 7 \
    --collection myco_fungi_features_full_finetuned
```

Available `--extractor` values:
`resnet50`, `resnet50_finetuned`, `mobilenetv2`, `mobilenetv2_finetuned`, `efficientnetb1`, `efficientnetb1_finetuned`, `efficientnetb1_triplet`, `hog`, `gabor`, `colorhistogram`, `colorhistogramhs`, `all`, `all-finetuned`

Output per run (saved to `results/run_<timestamp>_k<K>/`):
- `prediction_report_*.txt`
- `evaluation_results.json`
- `confusion_matrix.png`
- `<extractor>_<strategy>_<env>.csv`

### Step 7 — 5-Fold Cross-Validation

Runs strain-level cross-validation: all strains rotate as test across 5 folds. Sweeps E1/E2 × weighted/uni × K∈{3,5,7,9,11} — **100 runs total**. Results append to CSV so the job is resumable.

```bash
uv run python src/main.py cross-validate \
    --collection myco_fungi_features_full_finetuned
```

Output:
```
report/week_1_2/cv_results.csv          ← per-prediction rows (fold, env, strategy, K)
report/week_1_2/cv_summary_table.csv    ← mean/std accuracy per (env, strategy, K)
```

**Generate visualizations after the run:**
```bash
uv run python src/main.py cross-validate-visualize
```

Output images (`report/week_1_2/images/`):
- `accuracy_vs_k.png` — accuracy vs K for all 4 setting combinations
- `fold_variance.png` — box plot of fold variance at each K
- `heatmap_env_strategy_k.png` — heatmap (env×strategy) vs K
- `env_comparison.png` — E1 vs E2 bar chart per strategy

### Step 8 — Prediction

**Known strain (already in Qdrant):**
```bash
uv run python src/main.py predict --strain "DTO 123-A1" --extractor resnet50 --k 5
```

**New unseen strain (raw images):**
```bash
# Directory: new_strain/MEA/*.jpg, new_strain/CYA/*.jpg ...
uv run python src/main.py predict-new \
    --path /path/to/new_strain \
    --extractor efficientnetb1_finetuned \
    --k 7 \
    --strategy avg
```

### Step 9 — Comprehensive Report

```bash
uv run python src/main.py report \
    --identifier run_experiment_1 \
    --extractors resnet50,mobilenetv2 \
    --strategies all,ob,rev \
    --agg-strategies avg,uni \
    --k 5
```

## Project Structure

```
.
├── pyproject.toml                         # Project metadata & dependencies
├── requirements.txt                       # Pip-compatible requirements
├── docker-compose.yml                     # Qdrant service
├── species_weights.json                   # Per-species manual ensemble weights
├── colab/                                 # Google Colab training scripts
│   ├── train_models.py                    # Fine-tune ResNet50/MobileNetV2/EfficientNetB1
│   ├── train_models_triplet_loss.py       # Triplet loss training for EfficientNetB1
│   ├── train_models_selfsupervised.py     # Self-supervised (SimCLR) pre-training
│   ├── train_models_cellvit.py            # CellViT-based ViT training
│   ├── TRAINING.md                        # Training approaches guide
│   └── VIT_NOTES.md                       # ViT analysis, augmentation, TPU setup
├── src/
│   ├── main.py                            # CLI entry point (all subcommands)
│   ├── config.py                          # Centralized paths & constants
│   ├── classification/
│   │   ├── evaluate_species.py            # Evaluation loop & CSV output
│   │   ├── prediction.py                  # Core KNN prediction logic
│   │   └── visualization/
│   ├── database/
│   │   ├── upload_qdrant.py               # Feature upload to Qdrant
│   │   └── query_utils.py                 # Similarity search helpers
│   ├── feature_extraction/
│   │   ├── feature_extractors.py          # All extractor classes
│   │   └── generate_features.py           # Batch feature extraction
│   ├── preprocessing/
│   │   ├── preprocess.py                  # Petri dish crop
│   │   └── kmeans.py                      # K-means colony segmentation
│   ├── training/
│   │   └── train_models.py                # Local PyTorch training loop
│   ├── utils/
│   │   └── dataset_utils.py               # Strain/species mapping helpers
│   └── scripts/
│       ├── reformat_dataset.py            # Step 1: dataset reformatting
│       ├── generate_strain_mapping.py     # Step 2: CSV mapping generation
│       ├── extract_finetuned_features.py  # Feature extraction (fine-tuned)
│       ├── upload_finetuned_features.py   # Upload to finetuned collection
│       ├── cross_validation.py            # 5-fold CV runner
│       └── cv_visualize.py                # CV result visualizations
├── report/
│   ├── week_1_2/                          # Phase 3 & 4 outputs
│   │   ├── cv_results.csv
│   │   ├── cv_summary_table.csv
│   │   ├── images/                        # Visualization exports
│   │   └── REPORT.md
│   └── final_gr2/                         # Final group report
├── Dataset/                               # Data directory (git-ignored)
│   ├── original/                          ← place downloaded dataset here
│   ├── full_image/
│   ├── segmented_image/
│   └── strain_to_specy.csv
└── weights/                               # Model weight files (git-ignored)
    ├── EfficientNetB1_finetuned.pth
    ├── MobileNetV2_finetuned.pth
    ├── ResNet50_finetuned.pth
    └── EfficientNetB1_triplet.pth
```

## Quick Command Reference

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Qdrant database |
| `uv run python src/main.py reformat` | Preprocess and segment images |
| `uv run python src/main.py generate-mapping` | Create train/test split CSV |
| `uv run python src/main.py extract` | Extract standard features → Qdrant |
| `uv run python src/main.py extract-finetuned` | Extract finetuned DL features |
| `uv run python src/main.py upload-finetuned` | Upload finetuned collection to Qdrant |
| `uv run python src/main.py train` | Fine-tune DL models locally |
| `uv run python src/main.py evaluate --extractor efficientnetb1_finetuned --k 5` | Single-setting evaluation |
| `uv run python src/main.py evaluate-all --k 7` | Full brute-force sweep |
| `uv run python src/main.py cross-validate` | 5-fold CV (100 runs) |
| `uv run python src/main.py cross-validate-visualize` | Generate CV visualizations |
| `uv run python src/main.py predict --strain "X"` | Predict known strain |
| `uv run python src/main.py predict-new --path /path` | Predict new unseen strain |

## Notes

- All paths and constants are defined in `src/config.py`.
- Qdrant must be running before `extract`, `upload-finetuned`, `evaluate`, `predict`, or `cross-validate`.
- Qdrant data persists in `./qdrant_storage/` across container restarts.
- Cross-validation results append to CSV after each combination, so interrupted runs can be resumed safely.
- The `--collection` flag on evaluate/evaluate-all/cross-validate selects between the base collection (`myco_fungi_features_full`) and the finetuned collection (`myco_fungi_features_full_finetuned`).
