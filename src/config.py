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
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://dcb3eb29-ce49-4e3c-adb4-c980e48488b3.eu-central-1-0.aws.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.por1vDH3JOHaLYwrL_Eu_21ZGuF5mXca7pxGSBUvMDI",
)
COLLECTION_NAME = "myco_fungi_features_full"

# Image Processing
HEIGHT = 256
WIDTH = 256
TARGET_SIZE = (HEIGHT, WIDTH)

# Ensure directories exist
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def get_qdrant_client():
    """Return a QdrantClient connected to the configured Qdrant instance."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
