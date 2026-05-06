import os
import re
from pathlib import Path


def _default_workspace_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
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

    search_roots = [project_root.parent, project_root.parent.parent]
    if project_root.name == "fungal-cv-qdrant":
        for candidate in search_roots:
            if any((candidate / marker).exists() for marker in monorepo_markers):
                return candidate

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
CANONICAL_CURATED_SOURCE_DATASET_PATH = DATASET_ROOT / "curated_primary"
CANONICAL_INCOMING_SOURCE_DATASET_PATH = DATASET_ROOT / "incoming_low_quality"
LEGACY_CURATED_SOURCE_DATASET_PATH = DATASET_ROOT / "original"
LEGACY_INCOMING_SOURCE_DATASET_PATH = DATASET_ROOT / "new_data"

CURATED_SOURCE_DATASET_PATH = (
    CANONICAL_CURATED_SOURCE_DATASET_PATH
    if CANONICAL_CURATED_SOURCE_DATASET_PATH.exists()
    else LEGACY_CURATED_SOURCE_DATASET_PATH
)
INCOMING_SOURCE_DATASET_PATH = (
    CANONICAL_INCOMING_SOURCE_DATASET_PATH
    if CANONICAL_INCOMING_SOURCE_DATASET_PATH.exists()
    else LEGACY_INCOMING_SOURCE_DATASET_PATH
)
SOURCE_COLLECTIONS = {
    "curated": {
        "display_name": "curated_primary",
        "quality_tier": "curated",
        "path": CURATED_SOURCE_DATASET_PATH,
    },
    "incoming": {
        "display_name": "incoming_low_quality",
        "quality_tier": "incoming",
        "path": INCOMING_SOURCE_DATASET_PATH,
    },
}
ORIGINAL_DATASET_PATH = CURATED_SOURCE_DATASET_PATH
PREPARED_DATASET_DIR = DATASET_ROOT / "prepared"
FULL_IMAGE_PATH = DATASET_ROOT / "full_image"
SEGMENTED_IMAGE_DIR = DATASET_ROOT / "segmented_image"
FILE_EXTENSION = ".jpg"

# Rename control: set MYCOAI_PERFORM_SOURCE_RENAME=1 to atomically rename source dirs
_PERFORM_RENAME = os.getenv("MYCOAI_PERFORM_SOURCE_RENAME", "") == "1"

# Incoming source uses letter-range grouping folders that must be skipped
LETTER_RANGE_PATTERN = re.compile(r"^[A-Z]\s*[-–]\s*[A-Z]$")

# Incoming source filename: "Txxx ENV ob.jpg" or "Txxx ENV rev.jpg"
INCOMING_FILENAME_PATTERN = re.compile(
    r"(?P<strain>[A-Z0-9][A-Z0-9\s]*?[0-9]+)\s+(?P<environment>[A-Z0-9]+)\s+(?P<angle>ob|rev)",
    re.IGNORECASE,
)

# Curated source filename: "DTO 148-D1 MEAob.jpg" or "DTO 148-D1 MEAob_edited.jpg"
CURATED_FILENAME_PATTERN = re.compile(
    r"(?P<strain>DTO\s[0-9]+-[A-Z0-9]+)\s+(?P<environment>[A-Z0-9]+)(?P<angle>ob|rev)",
    re.IGNORECASE,
)

# Metadata Paths — consolidated per-collection JSON arrays
CURATED_METADATA_PATH = DATASET_ROOT / "curated_primary_metadata.json"
INCOMING_METADATA_PATH = DATASET_ROOT / "incoming_low_quality_metadata.json"
COLLECTION_METADATA_PATHS = {
    "curated": CURATED_METADATA_PATH,
    "incoming": INCOMING_METADATA_PATH,
}

PREPARED_ITEMS_METADATA_PATH = DATASET_ROOT / "prepared_items_metadata.json"
PREPARED_SEGMENTS_METADATA_PATH = DATASET_ROOT / "prepared_segments_metadata.json"
FULL_IMAGE_METADATA_PATH = DATASET_ROOT / "full_image_metadata.json"
SEGMENTED_METADATA_PATH = PREPARED_SEGMENTS_METADATA_PATH
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
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(resolved)


def get_qdrant_client():
    """Return a QdrantClient connected to the configured Qdrant instance."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def perform_source_rename() -> None:
    """Atomically rename legacy source directories to canonical names.

    original/ -> curated_primary/
    new_data/  -> incoming_low_quality/
    """
    rename_map = [
        (LEGACY_CURATED_SOURCE_DATASET_PATH, CANONICAL_CURATED_SOURCE_DATASET_PATH),
        (LEGACY_INCOMING_SOURCE_DATASET_PATH, CANONICAL_INCOMING_SOURCE_DATASET_PATH),
    ]
    for legacy, canonical in rename_map:
        if legacy.exists() and not canonical.exists():
            legacy.rename(canonical)
