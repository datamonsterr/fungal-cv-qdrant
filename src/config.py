import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Paths
ORIGINAL_DATASET_PATH = DATASET_ROOT / "original"
FULL_IMAGE_PATH = DATASET_ROOT / "full_image"
SEGMENTED_IMAGE_DIR = DATASET_ROOT / "segmented_image"

# Metadata Paths
FULL_IMAGE_METADATA_PATH = DATASET_ROOT / "full_image_metadata.json"
SEGMENTED_METADATA_PATH = DATASET_ROOT / "segmented_image_metadata.json"
STRAIN_SPECIES_MAPPING_PATH = DATASET_ROOT / "strain_to_specy.csv"

# Feature Paths
FEATURES_JSON_PATH = DATASET_ROOT / "segmented_features.json"

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "myco_fungi_features_full"

# Image Processing
HEIGHT = 256
WIDTH = 256
TARGET_SIZE = (HEIGHT, WIDTH)

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
