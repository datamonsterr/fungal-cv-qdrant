import argparse
import sys

from src.config import (
    COLLECTION_NAME,
    FEATURES_JSON_PATH,
    PREPARED_SEGMENTS_METADATA_PATH,
    SOURCE_COLLECTIONS,
    _PERFORM_RENAME,
    perform_source_rename,
)
from src.experiments.feature_extraction.generate_features import generate_features
from src.prepare.checks import check_dataset_root, check_metadata_exists, check_qdrant
from src.prepare.dataset import (
    prepare_dataset,
    resolve_source_collection_names,
    run_segmentation,
)
from src.utils.generate_strain_mapping import generate_strain_mapping
from src.utils.upload_qdrant import upload_features_to_qdrant


def run_prepare_init(
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
    source_collections: list[str] | None = None,
    limit: int | None = None,
) -> None:
    resolved_collections = resolve_source_collection_names(source_collections)
    request_paths = [
        SOURCE_COLLECTIONS[key]["path"] for key in resolved_collections
    ]
    ok, msg = check_dataset_root(request_paths)
    print(msg)
    if not ok:
        raise RuntimeError(msg)

    print("Generating strain mapping...")
    generate_strain_mapping()

    if _PERFORM_RENAME:
        print("Renaming source collections to canonical names...")
        perform_source_rename()

    print("Preparing canonical dataset hierarchy...")
    item_records = prepare_dataset(
        source_collections=resolve_source_collection_names(source_collections),
        limit=limit,
    )
    print(f"  Prepared {len(item_records)} items")

    print("Running segmentation...")
    run_segmentation(item_records, limit=limit)
    print(f"  Segmented {len(item_records)} items")

    ok, msg = check_metadata_exists(collection_keys=resolved_collections)
    print(msg)
    if not ok:
        raise RuntimeError(msg)

    print("Generating base features...")
    generate_features()

    ok, msg = check_qdrant()
    print(msg)
    if not ok:
        raise RuntimeError(msg)

    print("Uploading features to Qdrant...")
    from qdrant_client import QdrantClient

    from src.config import QDRANT_API_KEY, QDRANT_URL

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    upload_features_to_qdrant(
        client=client,
        collection_name=collection_name,
        features_json_path=str(FEATURES_JSON_PATH),
        metadata_json_path=str(PREPARED_SEGMENTS_METADATA_PATH),
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare canonical dataset hierarchy and Qdrant inputs"
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Target Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Qdrant upload batch size",
    )
    parser.add_argument(
        "--source-collection",
        action="append",
        default=[],
        dest="source_collections",
        help="Source collection key to prepare (repeatable: curated, incoming)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max image count for smoke runs",
    )

    args = parser.parse_args()

    try:
        run_prepare_init(
            collection_name=args.collection,
            batch_size=args.batch_size,
            source_collections=args.source_collections or None,
            limit=args.limit,
        )
        print("Prepare init completed successfully.")
    except Exception as exc:
        print(f"Prepare init failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
