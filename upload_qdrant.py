"""
Script to upload extracted features from feature_extractors.py to Qdrant vector database.
"""
import json
import sys
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_configs: Dict[str, int],
    distance: Distance = Distance.COSINE
) -> None:
    """
    Create a Qdrant collection with multiple named vectors.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
        vector_configs: Dictionary mapping vector names to their dimensions
        distance: Distance metric to use (COSINE, EUCLID, DOT)
    """
    # Check if collection already exists
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        client.delete_collection(collection_name=collection_name)
    
    # Create collection with multiple named vectors
    vectors_config = {
        name: VectorParams(size=dim, distance=distance)
        for name, dim in vector_configs.items()
    }
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config
    )
    print(f"Created collection '{collection_name}' with vectors: {list(vector_configs.keys())}")


def upload_features_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    features_json_path: str,
    metadata_json_path: str,
    batch_size: int = 100
) -> None:
    """
    Upload extracted features from JSON to Qdrant with metadata.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to upload to
        features_json_path: Path to the JSON file with extracted features
        metadata_json_path: Path to the JSON file with image metadata
        batch_size: Number of points to upload in each batch
    """
    # Load features from JSON
    with open(features_json_path, 'r') as f:
        features_data = json.load(f)
    
    print(f"Loaded {len(features_data)} feature records from {features_json_path}")
    
    # Load metadata from JSON
    with open(metadata_json_path, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"Loaded {len(metadata_list)} metadata records from {metadata_json_path}")
    
    # Create a lookup dictionary for metadata by ID
    metadata_by_id = {item['id']: item for item in metadata_list}
    print(f"Created metadata lookup for {len(metadata_by_id)} records")
    
    if not features_data:
        print("No data to upload!")
        return
    
    # Get vector configurations from first record
    vector_configs = {
        feat_name: feat_data['dimension']
        for feat_name, feat_data in features_data[0]['features'].items()
    }
    print(f"Detected feature types: {list(vector_configs.keys())}")
    print(f"Feature dimensions: {vector_configs}")
    
    # Create collection
    create_collection(client, collection_name, vector_configs)
    
    # Prepare points for upload
    points: List[PointStruct] = []
    skipped_count = 0
    
    for idx, record in enumerate(features_data):
        image_id = record['id']
        
        # Prepare vectors dictionary
        vectors = {
            feat_name: feat_data['vector']
            for feat_name, feat_data in record['features'].items()
        }
        
        # Get metadata for this image
        metadata = metadata_by_id.get(image_id)
        if metadata is None:
            print(f"Warning: No metadata found for image_id {image_id}, skipping...")
            skipped_count += 1
            continue
        
        # Create payload with full metadata
        payload = {
            'image_id': image_id,
            'feature_types': list(vectors.keys()),
            'parent_id': metadata.get('parent_id', ''),
            'segment_index': metadata.get('segment_index', -1),
            'bbox': metadata.get('bbox', {}),
            # Include all data fields from metadata
            'strain': metadata.get('data', {}).get('strain', 'unknown'),
            'environment': metadata.get('data', {}).get('environment', 'unknown'),
            'angle': metadata.get('data', {}).get('angle', 'unknown'),
            'specy': metadata.get('data', {}).get('specy', 'unknown'),
        }
        
        # Create point with payload containing metadata
        point = PointStruct(
            id=idx,  # Use index as point ID
            vector=vectors,
            payload=payload
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Uploaded batch of {len(points)} points (total: {idx + 1}/{len(features_data)})")
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Uploaded final batch of {len(points)} points")
    
    # Verify upload
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"\nUpload complete!")
    print(f"Total points in collection: {collection_info.points_count}")
    print(f"Vectors per point: {list(vector_configs.keys())}")
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} records due to missing metadata")


def main() -> None:
    """Main function to upload features to Qdrant."""
    # Configuration
    QDRANT_URL = "http://localhost:6333"
    COLLECTION_NAME = "myco_fungi_features"
    FEATURES_JSON_PATH = "../Dataset/segmented_features.json"
    METADATA_JSON_PATH = "../Dataset/segmented_image_metadata.json"
    BATCH_SIZE = 100
    
    # Initialize Qdrant client
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL)
        # Test connection
        client.get_collections()
        print("Successfully connected to Qdrant!")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Make sure Qdrant is running at http://localhost:6333")
        sys.exit(1)
    
    # Upload features
    try:
        upload_features_to_qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            features_json_path=FEATURES_JSON_PATH,
            metadata_json_path=METADATA_JSON_PATH,
            batch_size=BATCH_SIZE
        )
        print(f"\n✅ Successfully uploaded features to collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
