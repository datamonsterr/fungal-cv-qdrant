from pathlib import Path
from typing import Tuple

from qdrant_client import QdrantClient

from src.config import ORIGINAL_DATASET_PATH, QDRANT_API_KEY, QDRANT_URL, SEGMENTED_METADATA_PATH


def check_dataset_root(path: Path = ORIGINAL_DATASET_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Dataset root does not exist: {path}"
    if not any(path.iterdir()):
        return False, f"Dataset root is empty: {path}"
    return True, f"Dataset root is ready: {path}"


def check_metadata_exists(path: Path = SEGMENTED_METADATA_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"Segmented metadata file missing: {path}"
    if path.stat().st_size == 0:
        return False, f"Segmented metadata file is empty: {path}"
    return True, f"Segmented metadata looks present: {path}"


def check_qdrant() -> Tuple[bool, str]:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        return True, f"Qdrant reachable at {QDRANT_URL}"
    except Exception as exc:
        return False, f"Qdrant connection failed: {exc}"
