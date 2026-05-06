"""
Extract features using fine-tuned deep learning models.
This script extracts only DL features (ResNet50, MobileNetV2, EfficientNetB1)
using the fine-tuned weights from training.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import cv2

from src.config import (
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    WEIGHTS_DIR,
    WORKSPACE_ROOT,
)
from src.experiments.feature_extraction.feature_extractors import (
    EfficientNetB1Extractor,
)


def extract_finetuned_features(  # noqa: C901
    segmented_image_path: Path,
    metadata_path: Path,
    weights_dir: Path,
    output_json_path: Path,
    fold_index: Optional[int] = None,
) -> list[dict]:
    """
    Extract features using fine-tuned deep learning models.

    Args:
        segmented_image_path: Path to segmented images directory
        metadata_path: Path to metadata JSON file
        weights_dir: Path to directory containing fine-tuned weights
        output_json_path: Path to save extracted features JSON

    Returns:
        List of feature dictionaries
    """
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata")

    # EfficientNetB1-only flow for fold-aware CV
    if fold_index is None:
        efficientnet_weights = weights_dir / "EfficientNetB1_finetuned.pth"
    else:
        efficientnet_weights = (
            weights_dir / f"fold{fold_index}_EfficientNetB1_finetuned.pth"
        )

    if not efficientnet_weights.exists():
        print(f"Error: EfficientNetB1 weights not found at {efficientnet_weights}")
        sys.exit(1)

    print(
        "Initializing EfficientNetB1 with fine-tuned weights: "
        f"{efficientnet_weights}"
    )
    extractor_name = "EfficientNetB1_finetuned"
    extractor = EfficientNetB1Extractor(weights_path=str(efficientnet_weights))

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
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    total_features = 0
    if results:
        total_features = sum(
            feat["dimension"] for feat in results[0]["features"].values()
        )

    print("\nFine-tuned feature extraction complete!")
    print(f"Processed {len(results)} images")
    print(f"Feature types: {list(results[0]['features'].keys()) if results else []}")
    print(f"Total feature dimension: {total_features}")
    print(f"Results saved to: {output_json_path}")

    return results


def main(fold_index: Optional[int] = None):
    """Main function to extract fine-tuned features."""
    # Output path for fine-tuned features
    if fold_index is None:
        output_path = SEGMENTED_IMAGE_DIR.parent / "finetuned_dl_features.json"
    else:
        output_path = (
            SEGMENTED_IMAGE_DIR.parent / f"finetuned_dl_features_fold{fold_index}.json"
        )

    print("=" * 60)
    print("Fine-Tuned Deep Learning Feature Extraction")
    print("=" * 60)
    print(f"Segmented images: {SEGMENTED_IMAGE_DIR}")
    print(f"Metadata: {SEGMENTED_METADATA_PATH}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Output: {output_path}")
    print(f"Fold index: {fold_index if fold_index is not None else 'default'}")
    print("=" * 60 + "\n")

    extract_finetuned_features(
        segmented_image_path=SEGMENTED_IMAGE_DIR,
        metadata_path=SEGMENTED_METADATA_PATH,
        weights_dir=WEIGHTS_DIR,
        output_json_path=output_path,
        fold_index=fold_index,
    )


if __name__ == "__main__":
    main()
