import argparse
import sys

from src.config import COLLECTION_NAME, FEATURES_JSON_PATH, SEGMENTED_METADATA_PATH
from src.experiments.feature_extraction.generate_features import generate_features
from src.prepare.checks import check_dataset_root, check_metadata_exists, check_qdrant
from src.utils.generate_strain_mapping import generate_strain_mapping
from src.utils.reformat_dataset import reformat_dataset
from src.utils.upload_qdrant import upload_features_to_qdrant


def run_prepare_init(collection_name: str = COLLECTION_NAME, batch_size: int = 100) -> None:
    ok, msg = check_dataset_root()
    print(msg)
    if not ok:
        raise RuntimeError(msg)

    print("Generating strain mapping...")
    generate_strain_mapping()

    print("Running dataset reformat (flat + hierarchical)...")
    reformat_dataset(create_hierarchical=True)

    ok, msg = check_metadata_exists()
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
        metadata_json_path=str(SEGMENTED_METADATA_PATH),
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare pipeline: Dataset/original to Qdrant collection"
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

    args = parser.parse_args()

    try:
        run_prepare_init(collection_name=args.collection, batch_size=args.batch_size)
        print("Prepare init completed successfully.")
    except Exception as exc:
        print(f"Prepare init failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
