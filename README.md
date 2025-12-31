# Myco Fungi Feature Extraction and Classification

This project experiments with different feature extractor methods for classifying myco fungi species from images using Computer Vision, Deep Learning (PyTorch), and Qdrant Vector Database.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- Docker (for Qdrant vector database)
- Nix (optional, for reproducible environment)

## 1. Environment Setup

This project uses `uv` for fast package management.

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

## 2. Setup Qdrant Vector Database

Run Qdrant using Docker:

```bash
# Pull and run Qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

Qdrant will be available at `http://localhost:6333`.

## 3. Usage Pipeline

All pipeline steps are orchestrated via `src/main.py`.

### Step 1: Reformat Dataset

Preprocesses raw images, segments fungi colonies using K-means clustering, and generates metadata.

```bash
python src/main.py reformat
```

**Output:**
- `Dataset/full_image/`
- `Dataset/segmented_image/`
- `Dataset/segmented_image_metadata.json`

### Step 1.1: Reformat Hierarchical (Optional)

Reformats the dataset into a hierarchical structure `{species}/{strain}/{environment}/` for easier manual inspection or standard image classification loaders.

```bash
python src/main.py reformat-hierarchical
```

**Output:**
- `Dataset/hierarchical/`

### Step 2: Generate Strain Mapping

Generates a clean `strain_to_specy.csv` based on available data and creates a train/test split (one strain per species reserved for testing).

```bash
python src/main.py generate-mapping
```

**Output:**
- `Dataset/strain_to_specy.csv`

### Step 3: Feature Extraction & Upload

Extracts features (ResNet50, MobileNetV2, EfficientNetV2, HOG, Gabor, Color Histograms) and uploads them to Qdrant.

```bash
python src/main.py extract
```

**Output:**
- `Dataset/segmented_features.json`
- Qdrant Collection: `myco_fungi_features_full`

### Step 4: Train Models

Fine-tunes PyTorch models (ResNet50, MobileNetV2, EfficientNetV2) on the training set.

```bash
python src/main.py train
```

**Output:**
- `weights/*.pth` (Trained models)
- `weights/*_history.json` (Training logs)

### Step 5: Evaluation

Evaluates the models on the test set (unseen strains).

```bash
python src/main.py evaluate --extractor resnet50 --k 5
```

### Step 6: Prediction

Predict species for a specific strain using vector similarity search.

```bash
python src/main.py predict --strain "DTO 123-A1" --extractor resnet50
```

## Project Structure

```
.
├── pyproject.toml                     # Project dependencies
├── src/
│   ├── main.py                        # CLI Entry point
│   ├── config.py                      # Centralized configuration
│   ├── classification/                # Prediction & Evaluation logic
│   ├── database/                      # Qdrant interaction
│   ├── feature_extraction/            # Feature extractors (PyTorch & Traditional)
│   ├── preprocessing/                 # Image processing & Segmentation
│   ├── training/                      # PyTorch training loop
│   ├── utils/                         # Helper utilities
│   └── scripts/                       # Standalone scripts
├── Dataset/                           # Data directory (ignored by git)
└── README.md                          # Documentation
```

## Notes

- **Configuration**: Paths and constants are defined in `src/config.py`.
- **Dependencies**: Managed via `pyproject.toml`. Use `uv add <package>` to add new ones.
- Ensure Qdrant is running before uploading or querying.
