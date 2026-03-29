import argparse
import json
import sys
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL, SEGMENTED_METADATA_PATH


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_configs: Dict[str, int],
    distance: Distance = Distance.COSINE,
) -> None:
    """Create a Qdrant collection with named vectors, replacing existing collection."""
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name=collection_name)

    vectors_config = {
        name: VectorParams(size=dim, distance=distance)
        for name, dim in vector_configs.items()
    }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
    )
    print(
        f"Created collection '{collection_name}' with vectors: {list(vector_configs.keys())}"
    )


def upload_features_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    features_json_path: str,
    metadata_json_path: str,
    batch_size: int = 100,
) -> None:
    """Upload feature JSON records to Qdrant with metadata payloads."""
    with open(features_json_path, "r") as f:
        features_data = json.load(f)

    print(f"Loaded {len(features_data)} feature records from {features_json_path}")

    with open(metadata_json_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Loaded {len(metadata_list)} metadata records from {metadata_json_path}")

    metadata_by_id = {item["id"]: item for item in metadata_list}
    print(f"Created metadata lookup for {len(metadata_by_id)} records")

    if not features_data:
        print("No data to upload!")
        return

    vector_configs = {
        feat_name: feat_data["dimension"]
        for feat_name, feat_data in features_data[0]["features"].items()
    }
    print(f"Detected feature types: {list(vector_configs.keys())}")
    print(f"Feature dimensions: {vector_configs}")

    create_collection(client, collection_name, vector_configs)

    points: List[PointStruct] = []
    skipped_count = 0

    for idx, record in enumerate(features_data):
        image_id = record["id"]

        vectors = {
            feat_name: feat_data["vector"]
            for feat_name, feat_data in record["features"].items()
        }

        metadata = metadata_by_id.get(image_id)
        if metadata is None:
            print(f"Warning: No metadata found for image_id {image_id}, skipping...")
            skipped_count += 1
            continue

        payload = {
            "image_id": image_id,
            "feature_types": list(vectors.keys()),
            "parent_id": metadata.get("parent_id", ""),
            "segment_index": metadata.get("segment_index", -1),
            "bbox": metadata.get("bbox", {}),
            "strain": metadata.get("data", {}).get("strain", "unknown"),
            "environment": metadata.get("data", {}).get("environment", "unknown"),
            "angle": metadata.get("data", {}).get("angle", "unknown"),
            "specy": metadata.get("data", {}).get("specy", "unknown"),
        }

        point = PointStruct(id=idx, vector=vectors, payload=payload)
        points.append(point)

        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            print(
                f"Uploaded batch of {len(points)} points (total: {idx + 1}/{len(features_data)})"
            )
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Uploaded final batch of {len(points)} points")

    collection_info = client.get_collection(collection_name=collection_name)
    print("\nUpload complete!")
    print(f"Total points in collection: {collection_info.points_count}")
    print(f"Vectors per point: {list(vector_configs.keys())}")
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} records due to missing metadata")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload feature JSON data to Qdrant")
    parser.add_argument(
        "--features-json",
        required=True,
        help="Path to features JSON file",
    )
    parser.add_argument(
        "--metadata-json",
        default=str(SEGMENTED_METADATA_PATH),
        help="Path to metadata JSON file",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of points to upload per batch",
    )

    args = parser.parse_args()

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        print("Successfully connected to Qdrant!")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running at http://localhost:6333")
        sys.exit(1)

    try:
        upload_features_to_qdrant(
            client=client,
            collection_name=args.collection,
            features_json_path=args.features_json,
            metadata_json_path=args.metadata_json,
            batch_size=args.batch_size,
        )
        print(f"\nSuccessfully uploaded features to collection '{args.collection}'")
    except Exception as e:
        print(f"Error during upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
