import os
from pathlib import Path


def _default_workspace_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    parent = project_root.parent
    monorepo_markers = (
        "Dataset",
        "results",
        "weights",
        "mise.toml",
        ".agents",
        ".claude",
        ".opencode",
        "AGENTS.md",
        "CLAUDE.md",
    )

    if project_root.name == "fungal-cv-qdrant" and any(
        (parent / marker).exists() for marker in monorepo_markers
    ):
        return parent

    return project_root


# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(
    os.getenv("MYCOAI_ROOT", str(_default_workspace_root()))
).resolve()
DATASET_ROOT = Path(
    os.getenv("DATASET_ROOT", str(WORKSPACE_ROOT / "Dataset"))
).resolve()
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", str(WORKSPACE_ROOT / "weights"))).resolve()
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(WORKSPACE_ROOT / "results"))).resolve()
SPECIES_WEIGHTS_PATH = Path(
    os.getenv("SPECIES_WEIGHTS_PATH", str(WORKSPACE_ROOT / "species_weights.json"))
).resolve()

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
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def relative_to_workspace(path: Path) -> str:
    """Return a path string relative to the monorepo/workspace root."""
    return str(Path(path).resolve().relative_to(WORKSPACE_ROOT))


def get_qdrant_client():
    """Return a QdrantClient connected to the configured Qdrant instance."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
