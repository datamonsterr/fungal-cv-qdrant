"""
Upload fine-tuned deep learning features to a new Qdrant collection.
This script creates a new collection with all vectors (old + new fine-tuned).
"""

import json
import sys
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import COLLECTION_NAME, FEATURES_JSON_PATH, QDRANT_API_KEY, QDRANT_URL, SEGMENTED_METADATA_PATH


def upload_combined_features_to_new_collection(  # noqa: C901
    client: QdrantClient,
    new_collection_name: str,
    old_features_path: Path,
    finetuned_features_path: Path,
    batch_size: int = 100,
) -> None:
    """
    Upload combined features (old + fine-tuned) to a new collection.

    Args:
        client: Qdrant client instance
        new_collection_name: Name of the new collection to create
        old_features_path: Path to JSON file with original features
        finetuned_features_path: Path to JSON file with fine-tuned features
        batch_size: Number of points to process in each batch
    """
    # Load old features
    with open(old_features_path, "r") as f:
        old_features_data = json.load(f)
    print(f"Loaded {len(old_features_data)} old feature records")

    # Load fine-tuned features
    with open(finetuned_features_path, "r") as f:
        finetuned_data = json.load(f)
    print(f"Loaded {len(finetuned_data)} fine-tuned feature records")
    
    # Load metadata
    metadata_path = Path(SEGMENTED_METADATA_PATH)
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} metadata records")
    
    # Create metadata lookup by image_id
    metadata_by_id = {item["id"]: item for item in metadata_list}

    if not old_features_data or not finetuned_data:
        print("Error: Missing feature data!")
        return

    # Get old vector configurations
    old_vector_configs = {
        feat_name: feat_data["dimension"]
        for feat_name, feat_data in old_features_data[0]["features"].items()
    }

    # Get new vector configurations
    new_vector_configs = {
        feat_name: feat_data["dimension"]
        for feat_name, feat_data in finetuned_data[0]["features"].items()
    }

    print(f"\nOld feature types: {list(old_vector_configs.keys())}")
    print(f"New feature types: {list(new_vector_configs.keys())}")

    # Create mapping of image_id to fine-tuned features
    finetuned_by_id = {item["id"]: item["features"] for item in finetuned_data}

    # Combine old and new vector configurations
    all_vector_configs = {**old_vector_configs, **new_vector_configs}
    print("\nCombined vector configuration:")
    for name, dim in all_vector_configs.items():
        print(f"  {name}: {dim}")

    # Create new collection with all vectors
    print(f"\nCreating new collection '{new_collection_name}' with all vectors...")

    # Delete if already exists
    collections = client.get_collections().collections
    if any(col.name == new_collection_name for col in collections):
        print(f"  Deleting existing collection '{new_collection_name}'...")
        client.delete_collection(collection_name=new_collection_name)

    vectors_config = {
        name: VectorParams(size=dim, distance=Distance.COSINE)
        for name, dim in all_vector_configs.items()
    }

    client.create_collection(
        collection_name=new_collection_name, vectors_config=vectors_config
    )
    print(f"✓ Collection created with {len(all_vector_configs)} vector fields")

    # Get metadata from old features (has all metadata)
    print("\nPreparing points with combined vectors...")
    uploaded_count = 0
    skipped_count = 0
    points_batch = []

    for old_item in old_features_data:
        image_id = old_item["id"]

        if image_id not in finetuned_by_id:
            print(f"  Warning: No fine-tuned features for image {image_id}, skipping")
            skipped_count += 1
            continue

        # Combine old and new vectors
        combined_vectors = {}
        for feat_name, feat_data in old_item["features"].items():
            combined_vectors[feat_name] = feat_data["vector"]

        for feat_name, feat_data in finetuned_by_id[image_id].items():
            combined_vectors[feat_name] = feat_data["vector"]

        # Combine feature types
        all_feature_types = list(old_item["features"].keys()) + list(
            finetuned_by_id[image_id].keys()
        )

        # Get metadata for this image
        metadata = metadata_by_id.get(image_id)
        if metadata is None:
            print(f"  Warning: No metadata found for image {image_id}, skipping")
            skipped_count += 1
            continue

        # Create payload with proper metadata structure
        payload = {
            "image_id": image_id,
            "feature_types": all_feature_types,
            "parent_id": metadata.get("parent_id", ""),
            "segment_index": metadata.get("segment_index", -1),
            "bbox": metadata.get("bbox", {}),
            "strain": metadata.get("data", {}).get("strain", "unknown"),
            "environment": metadata.get("data", {}).get("environment", "unknown"),
            "angle": metadata.get("data", {}).get("angle", "unknown"),
            "specy": metadata.get("data", {}).get("specy", "unknown"),
        }

        # Create point with metadata
        point = PointStruct(
            id=uploaded_count,
            vector=combined_vectors,
            payload=payload,
        )
        points_batch.append(point)
        uploaded_count += 1

        # Upload in batches
        if len(points_batch) >= batch_size:
            client.upsert(collection_name=new_collection_name, points=points_batch)
            print(f"  Uploaded {uploaded_count} points...")
            points_batch = []

    # Upload remaining points
    if points_batch:
        client.upsert(collection_name=new_collection_name, points=points_batch)

    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"{'='*60}")
    print(f"Uploaded points: {uploaded_count}")
    print(f"Skipped points (no matching features): {skipped_count}")

    # Verify final state
    final_info = client.get_collection(collection_name=new_collection_name)
    print("\nFinal collection info:")
    print(f"  Total points: {final_info.points_count}")
    print(f"  Vector fields: {list(final_info.config.params.vectors.keys())}")  # type: ignore
    print(f"{'='*60}")


def main(fold_index: Optional[int] = None):
    """Main function to upload fine-tuned features."""
    # Paths
    old_features_path = Path(FEATURES_JSON_PATH)
    if fold_index is None:
        finetuned_path = Path("Dataset/finetuned_dl_features.json")
        new_collection_name = f"{COLLECTION_NAME}_finetuned"
    else:
        finetuned_path = Path(f"Dataset/finetuned_dl_features_fold{fold_index}.json")
        new_collection_name = f"{COLLECTION_NAME}_finetuned_fold{fold_index}"

    if not old_features_path.exists():
        print(f"Error: Old features not found at {old_features_path}")
        print("\nPlease run the feature extraction first:")
        print("  uv run python -m src.main extract")
        sys.exit(1)

    if not finetuned_path.exists():
        print(f"Error: Fine-tuned features not found at {finetuned_path}")
        print("\nPlease run the feature extraction script first:")
        print("  uv run python -m src.main extract-finetuned")
        sys.exit(1)

    print("=" * 60)
    print("Upload Combined Features to New Qdrant Collection")
    print("=" * 60)
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"New Collection: {new_collection_name}")
    print(f"Fold index: {fold_index if fold_index is not None else 'default'}")
    print(f"Old features: {old_features_path}")
    print(f"Fine-tuned features: {finetuned_path}")
    print("=" * 60 + "\n")

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        client.get_collections()
        print("✓ Successfully connected to Qdrant\n")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running at http://localhost:6333")
        sys.exit(1)

    # Upload features
    try:
        upload_combined_features_to_new_collection(
            client=client,
            new_collection_name=new_collection_name,
            old_features_path=old_features_path,
            finetuned_features_path=finetuned_path,
            batch_size=10,
        )
        print(f"\n✅ Successfully created collection '{new_collection_name}'")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
