"""
Example script for querying the Qdrant vector database.
"""
import random
import os
from qdrant_client import QdrantClient
from query_utils import (
    find_nearest_neighbors_by_id,
    get_image_metadata,
    visualize_neighbors
)
from feature_extractors import ResNet50Extractor, HOGExtractor, ColorHistogramExtractor, GaborExtractor, FeatureExtractor

def get_random_image_id() -> str:
    segmented_image_dir = "../Dataset/segmented_image"
    image_files = [f for f in os.listdir(segmented_image_dir) if f.endswith('.jpg')]
    if not image_files:
        raise ValueError("No image files found in the specified directory.")
    random_image_file = random.choice(image_files)
    return os.path.splitext(random_image_file)[0]

extractors: list[FeatureExtractor] = [GaborExtractor(), ResNet50Extractor(), HOGExtractor(), ColorHistogramExtractor()]


def get_one_result(client, collection_name, feature_type, image_id) -> None:
    try:
        # Get query image metadata
        query_metadata = get_image_metadata(
            client=client,
            collection_name=collection_name,
            image_id=image_id
        )
        
        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_type,
            environment=query_metadata.get('environment', 'unknown'),
            num_neighbors=7,
            exclude_self=True
        )       
        # Create visualization
        visualize_neighbors(
            query_image_path=f"../Dataset/segmented_image/{image_id}.jpg",
            neighbors=neighbors,
            segmented_image_dir="../Dataset/segmented_image",
            output_path=f"./results/{image_id}_{feature_type}.jpg",
            query_metadata=query_metadata,
            max_neighbors=7
        )
    except Exception as e:
        print(f"Error: {e}")

def main():
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    for _ in range(10):
        image_id = get_random_image_id()
        for extractor in extractors:
            get_one_result(client, collection_name, extractor.name.lower(), image_id)


if __name__ == "__main__":
    main()
