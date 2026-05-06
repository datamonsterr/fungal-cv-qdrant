import argparse
import json
import sys
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import (
    COLLECTION_METADATA_PATHS,
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
)


def _load_all_items() -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    for path in COLLECTION_METADATA_PATHS.values():
        if not path.exists():
            continue
        with open(path, "r") as f:
            all_items.extend(json.load(f))
    return all_items


def _build_path_lookup(
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        for idx, seg_path in enumerate(item.get("paths", {}).get("segments", [])):
            lookup[seg_path] = {
                "instance_info": item.get("instance_info", {}),
                "segmentation": item.get("segmentation", {}),
                "index": idx,
            }
    return lookup


def _build_id_lookup(
    items: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item.get("item_id", "")
        for idx, seg_path in enumerate(item.get("paths", {}).get("segments", [])):
            seg_id = f"{item_id}_seg{idx}"
            lookup[seg_id] = {
                "instance_info": item.get("instance_info", {}),
                "segmentation": item.get("segmentation", {}),
                "index": idx,
                "segment_path": seg_path,
            }
    return lookup


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_configs: dict[str, int],
    distance: Distance = Distance.COSINE,
) -> None:
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
    metadata_json_path: str | None = None,
    batch_size: int = 100,
) -> None:
    with open(features_json_path, "r") as f:
        features_data = json.load(f)

    print(f"Loaded {len(features_data)} feature records from {features_json_path}")

    items = _load_all_items()
    id_lookup = _build_id_lookup(items)
    print(f"Loaded {len(items)} items from consolidated metadata ({len(id_lookup)} segment lookups)")

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

    points: list[PointStruct] = []
    skipped_count = 0

    for idx, record in enumerate(features_data):
        segment_id = record["id"]
        meta = id_lookup.get(segment_id)

        if meta is None:
            print(f"Warning: No metadata found for segment_id {segment_id}, skipping...")
            skipped_count += 1
            continue

        inst = meta["instance_info"]
        seg = meta["segmentation"]
        seg_idx = meta["index"]

        bbox = {}
        for method in ("kmeans", "contour"):
            method_bboxes = seg.get(method, [])
            if seg_idx < len(method_bboxes):
                bbox = method_bboxes[seg_idx]
                break

        vectors = {
            feat_name: feat_data["vector"]
            for feat_name, feat_data in record["features"].items()
        }

        payload = {
            "image_id": segment_id,
            "feature_types": list(vectors.keys()),
            "parent_item_id": segment_id.rsplit("_seg", 1)[0] if "_seg" in segment_id else segment_id,
            "segment_index": seg_idx,
            "bbox": bbox,
            "species": inst.get("species", "unknown"),
            "strain": inst.get("strain", "unknown"),
            "environment": inst.get("environment", "unknown"),
            "angle": inst.get("angle", "unknown"),
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
        default=None,
        help="Path to metadata JSON file (optional; reads consolidated metadata by default)",
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
