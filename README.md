# Myco Fungi Feature Extraction and Classification

This project experiments with different feature extractor methods for classifying myco fungi species from images using Computer Vision, Deep Learning (PyTorch), and Qdrant Vector Database.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Docker & Docker Compose (for Qdrant vector database)
- Nix (optional, for reproducible environment)

## 1. Environment Setup

This project uses `uv` for fast package management.

```bash
# Install dependencies
uv sync

# Activate virtual environment (optional, uv run handles this automatically)
source .venv/bin/activate
```

## 2. Setup Qdrant Vector Database

### Option A: Using Docker Compose (Recommended)

```bash
# Start Qdrant service
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f qdrant

# Stop service
docker compose down
```

### Option B: Using Docker directly

```bash
# Pull and run Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`.

## 3. Usage Pipeline

All pipeline steps are orchestrated via `src/main.py`. Use `uv run` to execute commands.

### Step 1: Reformat Dataset

Preprocesses raw images, segments fungi colonies using K-means clustering, and generates metadata.

```bash
uv run python src/main.py reformat
```

**Output:**
- `Dataset/full_image/`
- `Dataset/segmented_image/`
- `Dataset/segmented_image_metadata.json`

### Step 1.1: Reformat Hierarchical (Optional)

Reformats the dataset into a hierarchical structure `{species}/{strain}/{environment}/` for easier manual inspection or standard image classification loaders.

```bash
uv run python src/main.py reformat-hierarchical
```

**Output:**
- `Dataset/hierarchical/`

### Step 2: Generate Strain Mapping

Generates a clean `strain_to_specy.csv` based on available data and creates a train/test split (one strain per species reserved for testing).

```bash
uv run python src/main.py generate-mapping
```

**Output:**
- `Dataset/strain_to_specy.csv`

### Step 3: Feature Extraction & Upload

Extracts features (ResNet50, MobileNetV2, EfficientNetB1, HOG, Gabor, Color Histograms) and uploads them to Qdrant.

```bash
uv run python src/main.py extract
```

**Output:**
- `Dataset/segmented_features.json`
- Qdrant Collection: `myco_fungi_features_full`

### Step 4: Train Models (Optional)

Fine-tunes PyTorch models (ResNet50, MobileNetV2, EfficientNetB1) on the training set.

```bash
uv run python src/main.py train
```

**Output:**
- `weights/*.pth` (Trained models)
- `weights/*_history.json` (Training logs)

### Step 5: Evaluation

Evaluates the models on the test set (unseen strains).

**Evaluate with a single extractor:**
```bash
uv run python src/main.py evaluate --extractor resnet50 --k 5
```

**Evaluate with ALL extractors:**
```bash
uv run python src/main.py evaluate --extractor all --k 5
```

Available extractors: `resnet50`, `mobilenetv2`, `efficientnetb1`, `hog`, `gabor`, `colorhistogram`, `colorhistogramhs`, `all`

**Output:**
- `results/prediction_report_*.txt`
- `results/confusion_matrix.png`
- `results/eval_{extractor}/` (when using `--extractor all`)

### Step 6: Prediction

#### 6.1 Predict for Known Strain

Predict species for a specific strain using vector similarity search.

```bash
uv run python src/main.py predict --strain "DTO 123-A1" --extractor resnet50 --k 5
```

#### 6.2 Predict for New Unseen Strain

Predict species for a completely new strain from raw images (not in the database).

**Directory structure required:**
```
new_strain_folder/
├── environment1/
│   ├── image1.jpg
│   └── image2.jpg
└── environment2/
    ├── image1.jpg
    └── image2.jpg
```

**Command:**
```bash
uv run python src/main.py predict-new \
    --path /path/to/strain_folder \
    --extractor resnet50 \
    --k 7 \
    --strategy avg
```

**Options:**
- `--path`: Path to strain folder (required)
- `--extractor`: Feature extractor (default: `resnet50`)
  - Options: `resnet50`, `mobilenetv2`, `efficientnetb1`, `hog`, `gabor`, `colorhistogram`, `colorhistogramhs`
- `--k`: Number of nearest neighbors (default: 7)
- `--strategy`: Aggregation strategy (default: `avg`)
  - `avg`: Weighted by similarity scores
  - `uni`: Uniform weight (count-based)

**Output:**
- `results/predict_new_{strain_name}_{timestamp}/`
  - `prediction_results.json` - Full prediction results with rankings
  - `prediction_visualization.jpg` - Visual prediction display

**What it does:**
1. Reads raw images from `strain_folder/environment_name/*.jpg`
2. Applies Petri dish preprocessing and K-means segmentation
3. Extracts features from each segment using selected extractor
4. Searches Qdrant for nearest neighbors
5. Aggregates predictions across all segments
6. Outputs species ranking and visualization

### Step 7: Generate Comprehensive Report

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
├── pyproject.toml                     # Project dependencies
├── docker-compose.yml                 # Qdrant service configuration
├── src/
│   ├── main.py                        # CLI Entry point
│   ├── config.py                      # Centralized configuration
│   ├── classification/                # Prediction & Evaluation logic
│   │   └── visualization/             # Visualization utilities
│   ├── database/                      # Qdrant interaction
│   ├── feature_extraction/            # Feature extractors (PyTorch & Traditional)
│   ├── preprocessing/                 # Image processing & Segmentation
│   │   ├── preprocess.py              # Petri dish preprocessing
│   │   └── kmeans.py                  # K-means segmentation
│   ├── training/                      # PyTorch training loop
│   ├── utils/                         # Helper utilities
│   └── scripts/                       # Standalone scripts
├── Dataset/                           # Data directory (ignored by git)
├── qdrant_storage/                    # Qdrant persistent storage
└── README.md                          # Documentation
```

## Quick Command Reference

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Qdrant database |
| `docker compose down` | Stop Qdrant database |
| `uv run python src/main.py reformat` | Preprocess and segment images |
| `uv run python src/main.py generate-mapping` | Create train/test split |
| `uv run python src/main.py extract` | Extract features and upload to Qdrant |
| `uv run python src/main.py train` | Fine-tune deep learning models |
| `uv run python src/main.py evaluate --extractor all` | Evaluate all extractors |
| `uv run python src/main.py predict --strain "X"` | Predict known strain |
| `uv run python src/main.py predict-new --path /path` | Predict new unseen strain |

## Notes

- **Configuration**: Paths and constants are defined in `src/config.py`.
- **Dependencies**: Managed via `pyproject.toml`. Use `uv add <package>` to add new ones.
- **Qdrant**: Ensure Qdrant is running before uploading or querying.
- **Data Persistence**: Qdrant data is stored in `./qdrant_storage/` and persists across container restarts.
- **New Strain Prediction**: Automatically applies preprocessing and segmentation to raw images before feature extraction.
