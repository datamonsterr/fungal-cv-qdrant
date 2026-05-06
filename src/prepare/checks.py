from pathlib import Path
from typing import Tuple

from qdrant_client import QdrantClient

from src.config import (
    COLLECTION_METADATA_PATHS,
    QDRANT_API_KEY,
    QDRANT_URL,
)
from src.prepare.dataset import required_source_roots


def check_dataset_root(paths: list[Path] | None = None) -> Tuple[bool, str]:
    candidate_paths = paths if paths is not None else required_source_roots()
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


def check_metadata_exists(
    path: Path | None = None,
    collection_keys: list[str] | None = None,
) -> Tuple[bool, str]:
    if path is not None:
        metadata_paths = [path]
    elif collection_keys is not None:
        metadata_paths = [
            COLLECTION_METADATA_PATHS[k]
            for k in collection_keys
            if k in COLLECTION_METADATA_PATHS
        ]
    else:
        metadata_paths = list(COLLECTION_METADATA_PATHS.values())

    present: list[str] = []
    missing: list[str] = []
    for mp in metadata_paths:
        if mp.exists() and mp.stat().st_size > 0:
            present.append(str(mp))
        else:
            missing.append(str(mp))
    if missing:
        return False, f"Metadata files missing or empty: {', '.join(missing)}"
    return True, f"Metadata files present: {', '.join(present)}"


def check_qdrant() -> Tuple[bool, str]:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        return True, f"Qdrant reachable at {QDRANT_URL}"
    except Exception as exc:
        return False, f"Qdrant connection failed: {exc}"
