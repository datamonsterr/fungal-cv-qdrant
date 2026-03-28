"""
Upload EfficientNet B1 Triplet features to a new Qdrant collection.
This script creates a new collection with only the triplet loss features.
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    QDRANT_API_KEY,
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
)


def upload_triplet_features(
    client: QdrantClient,
    collection_name: str,
    features_path: Path,
    metadata_path: Path,
    batch_size: int = 100,
) -> None:
    """
    Upload triplet features to a new collection.

    Args:
        client: Qdrant client instance
        collection_name: Name of the new collection to create
        features_path: Path to JSON file with triplet features
        metadata_path: Path to metadata JSON file
        batch_size: Number of points to process in each batch
    """
    # Load features
    with open(features_path, "r") as f:
        features_data = json.load(f)
    print(f"Loaded {len(features_data)} feature records")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} metadata records")

    # Create metadata lookup by image_id
    metadata_by_id = {item["id"]: item for item in metadata_list}

    if not features_data:
        print("Error: Missing feature data!")
        return

    # Get vector configurations from the first item
    vector_configs = {
        feat_name: feat_data["dimension"]
        for feat_name, feat_data in features_data[0]["features"].items()
    }

    print(f"\nFeature types: {list(vector_configs.keys())}")
    for name, dim in vector_configs.items():
        print(f"  {name}: {dim}")

    # Create new collection
    print(f"\nCreating new collection '{collection_name}'...")

    # Delete if already exists
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"  Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)

    vectors_config = {
        name: VectorParams(size=dim, distance=Distance.COSINE)
        for name, dim in vector_configs.items()
    }

    client.create_collection(
        collection_name=collection_name, vectors_config=vectors_config
    )
    print(f"✓ Collection created with {len(vector_configs)} vector fields")

    # Prepare points
    print("\nPreparing and uploading points...")
    uploaded_count = 0
    skipped_count = 0
    points_batch = []

    for item in features_data:
        image_id = item["id"]

        # Get metadata for this image
        metadata = metadata_by_id.get(image_id)
        if metadata is None:
            print(f"  Warning: No metadata found for image {image_id}, skipping")
            skipped_count += 1
            continue

        # Get vectors
        vectors = {}
        for feat_name, feat_data in item["features"].items():
            vectors[feat_name] = feat_data["vector"]

        # Create payload with proper metadata structure
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

        # Create point
        point = PointStruct(
            id=uploaded_count,
            vector=vectors,
            payload=payload,
        )
        points_batch.append(point)
        uploaded_count += 1

        # Upload in batches
        if len(points_batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=points_batch)
            print(f"  Uploaded {uploaded_count} points...")
            points_batch = []

    # Upload remaining points
    if points_batch:
        client.upsert(collection_name=collection_name, points=points_batch)

    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"{'='*60}")
    print(f"Uploaded points: {uploaded_count}")
    print(f"Skipped points: {skipped_count}")

    # Verify final state
    final_info = client.get_collection(collection_name=collection_name)
    print("\nFinal collection info:")
    print(f"  Total points: {final_info.points_count}")
    print(f"  Vector fields: {list(final_info.config.params.vectors.keys())}")


def main():
    """Main function to upload triplet features."""
    # Paths
    features_path = SEGMENTED_IMAGE_DIR.parent / "triplet_features.json"
    metadata_path = Path(SEGMENTED_METADATA_PATH)
    
    # Collection name with suffix
    new_collection_name = f"{COLLECTION_NAME}_triplet"

    if not features_path.exists():
        print(f"Error: Features not found at {features_path}")
        print("\nPlease run the triplet feature extraction first:")
        print("  uv run python -m src.scripts.extract_triplet_features")
        sys.exit(1)

    print("=" * 60)
    print("Upload Triplet Features to Qdrant")
    print("=" * 60)
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {new_collection_name}")
    print(f"Features: {features_path}")
    print("=" * 60 + "\n")

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        print("✓ Successfully connected to Qdrant\n")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running at http://localhost:6333")
        sys.exit(1)

    # Upload features
    try:
        upload_triplet_features(
            client=client,
            collection_name=new_collection_name,
            features_path=features_path,
            metadata_path=metadata_path,
            batch_size=100,
        )
        print(f"\n✅ Successfully created collection '{new_collection_name}'")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
