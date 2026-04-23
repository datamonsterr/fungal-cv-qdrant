import os
import subprocess
from pathlib import Path


def _has_any_marker(path: Path, markers: tuple[str, ...]) -> bool:
    return any((path / marker).exists() for marker in markers)


def _discover_root_from_worktree(project_root: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    worktree_root = Path(result.stdout.strip()).resolve()
    if not str(worktree_root):
        return None
    monorepo_root = (
        worktree_root.parent.parent
        if worktree_root.parent.name == "worktrees"
        else worktree_root
    )
    return monorepo_root


def _default_workspace_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    strong_markers = (
        "Dataset",
        "results",
        "weights",
        "species_weights.json",
    )
    fallback_markers = (
        "mise.toml",
        ".agents",
        ".claude",
        ".opencode",
        "AGENTS.md",
        "CLAUDE.md",
    )

    if project_root.name != "fungal-cv-qdrant":
        return project_root

    worktree_root = _discover_root_from_worktree(project_root)
    if worktree_root is not None and _has_any_marker(worktree_root, strong_markers):
        return worktree_root

    fallback_root: Path | None = None
    for candidate in project_root.parents:
        if _has_any_marker(candidate, strong_markers):
            return candidate
        if fallback_root is None and _has_any_marker(candidate, fallback_markers):
            fallback_root = candidate

    if fallback_root is not None:
        return fallback_root

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
