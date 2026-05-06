from pathlib import Path
from typing import Tuple

from qdrant_client import QdrantClient

from src.config import (
    PREPARED_SEGMENTS_METADATA_PATH,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from src.prepare.dataset import required_source_roots


def check_dataset_root(path: Path | None = None) -> Tuple[bool, str]:
    candidate_paths = [path] if path is not None else required_source_roots()
    missing_paths = [candidate for candidate in candidate_paths if not candidate.exists()]
    if missing_paths:
        missing = ", ".join(str(candidate) for candidate in missing_paths)
        return False, f"Dataset source roots do not exist: {missing}"
    empty_paths = [candidate for candidate in candidate_paths if not any(candidate.iterdir())]
    if empty_paths:
        empty = ", ".join(str(candidate) for candidate in empty_paths)
        return False, f"Dataset source roots are empty: {empty}"
    ready = ", ".join(str(candidate) for candidate in candidate_paths)
    return True, f"Dataset source roots are ready: {ready}"


def check_metadata_exists(path: Path = PREPARED_SEGMENTS_METADATA_PATH) -> Tuple[bool, str]:
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
