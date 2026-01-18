"""
Upload ViT features to a new Qdrant collection.

This script uploads ViT features extracted by extract_vit_features.py
to a new Qdrant collection named 'myco_fungi_features_vit'.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient

from src.config import QDRANT_URL, SEGMENTED_METADATA_PATH
from src.database.upload_qdrant import upload_features_to_qdrant


def upload_vit_features(
    features_json_path: str = "Dataset/vit_features.json",
    collection_name: str = "myco_fungi_features_vit",
) -> None:
    """
    Upload ViT features to Qdrant.

    Args:
        features_json_path: Path to the ViT features JSON file
        collection_name: Name of the Qdrant collection to create/update
    """
    features_path = Path(features_json_path)

    if not features_path.exists():
        print(f"Error: Features file not found: {features_path}")
        print("Run extract_vit_features.py first to generate features.")
        sys.exit(1)

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        print("Successfully connected to Qdrant!")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running at http://localhost:6333")
        sys.exit(1)

    try:
        upload_features_to_qdrant(
            client=client,
            collection_name=collection_name,
            features_json_path=str(features_path),
            metadata_json_path=str(SEGMENTED_METADATA_PATH),
            batch_size=100,
        )
        print(f"\n✅ Successfully uploaded ViT features to collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)


def main():
    """Main function with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Upload ViT features to Qdrant")
    parser.add_argument(
        "--features",
        type=str,
        default="Dataset/vit_features.json",
        help="Path to ViT features JSON file",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="myco_fungi_features_vit",
        help="Name of the Qdrant collection",
    )

    args = parser.parse_args()

    upload_vit_features(
        features_json_path=args.features, collection_name=args.collection
    )


if __name__ == "__main__":
    main()
