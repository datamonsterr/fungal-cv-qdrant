"""
Extract features using the Triplet Loss EfficientNet B1 model.
This script extracts only the EfficientNet B1 Triplet features using the fine-tuned weights.
"""

import json
from pathlib import Path

import cv2

from src.config import (
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    WEIGHTS_DIR,
    WORKSPACE_ROOT,
)
from src.experiments.feature_extraction.feature_extractors import (
    EfficientNetB1TripletExtractor,
)


def extract_triplet_features(
    segmented_image_path: Path,
    metadata_path: Path,
    weights_dir: Path,
    output_json_path: Path,
) -> list[dict]:
    """
    Extract features using EfficientNet B1 Triplet model.

    Args:
        segmented_image_path: Path to segmented images directory
        metadata_path: Path to metadata JSON file
        weights_dir: Path to directory containing weights
        output_json_path: Path to save extracted features JSON

    Returns:
        List of feature dictionaries
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata")

    # Initialize extractor
    weights_path = weights_dir / "EfficientNetB1_triplet.pth"

    # Check if weights exist, but let the extractor handle the fallback/warning
    # if we want to follow the pattern strictly, or check here to be explicit.
    if weights_path.exists():
        print(f"Initializing EfficientNetB1 Triplet with weights: {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}, using ImageNet weights")

    extractor = EfficientNetB1TripletExtractor(weights_path=str(weights_path))
    extractors = [("efficientnetb1_triplet", extractor)]

    print("\nExtracting features with EfficientNetB1 Triplet model...")

    results = []

    for idx, metadata in enumerate(metadata_list):
        image_id = metadata.get("segment_id") or metadata["id"]
        segment_path = metadata.get("segment_path")
        if segment_path:
            image_path = WORKSPACE_ROOT / segment_path
        else:
            image_path = segmented_image_path / f"{image_id}.jpg"

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            print(f"Warning: Failed to load image: {image_path}")
            continue

        feature_data = {"id": image_id, "features": {}}

        try:
            for extractor_name, extractor in extractors:
                features = extractor.extract(image)
                feature_data["features"][extractor_name] = {
                    "vector": features.tolist(),
                    "dimension": len(features),
                }

            results.append(feature_data)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(metadata_list)} images...")

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            continue

    # Save results
    # Create parent directory if it doesn't exist
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    total_features = 0
    if results:
        total_features = sum(
            feat["dimension"] for feat in results[0]["features"].values()
        )

    print("\nTriplet feature extraction complete!")
    print(f"Processed {len(results)} images")
    print(f"Feature types: {list(results[0]['features'].keys()) if results else []}")
    print(f"Total feature dimension: {total_features}")
    print(f"Results saved to: {output_json_path}")

    return results


def main():
    """Main function to extract triplet features."""
    # Output path for triplet features
    output_path = SEGMENTED_IMAGE_DIR.parent / "triplet_features.json"

    print("=" * 60)
    print("EfficientNet B1 Triplet Feature Extraction")
    print("=" * 60)
    print(f"Segmented images: {SEGMENTED_IMAGE_DIR}")
    print(f"Metadata: {SEGMENTED_METADATA_PATH}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Output: {output_path}")
    print("=" * 60 + "\n")

    extract_triplet_features(
        segmented_image_path=SEGMENTED_IMAGE_DIR,
        metadata_path=SEGMENTED_METADATA_PATH,
        weights_dir=WEIGHTS_DIR,
        output_json_path=output_path,
    )


if __name__ == "__main__":
    main()
