"""
Extract features using fine-tuned deep learning models.
This script extracts only DL features (ResNet50, MobileNetV2, EfficientNetB1)
using the fine-tuned weights from training.
"""

import json
import sys
from pathlib import Path

import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    WEIGHTS_DIR,
)  # noqa: E402
from src.feature_extraction.feature_extractors import (  # noqa: E402
    EfficientNetB1Extractor,
    MobileNetV2Extractor,
    ResNet50Extractor,
)


def extract_finetuned_features(  # noqa: C901
    segmented_image_path: Path,
    metadata_path: Path,
    weights_dir: Path,
    output_json_path: Path,
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

    # Initialize extractors with fine-tuned weights
    extractors = []

    resnet_weights = weights_dir / "ResNet50_finetuned.pth"
    if resnet_weights.exists():
        print(f"Initializing ResNet50 with fine-tuned weights: {resnet_weights}")
        extractors.append(
            ("ResNet50_finetuned", ResNet50Extractor(weights_path=str(resnet_weights)))
        )
    else:
        print(f"Warning: ResNet50 weights not found at {resnet_weights}")

    mobilenet_weights = weights_dir / "MobileNetV2_finetuned.pth"
    if mobilenet_weights.exists():
        print(f"Initializing MobileNetV2 with fine-tuned weights: {mobilenet_weights}")
        extractors.append(
            (
                "MobileNetV2_finetuned",
                MobileNetV2Extractor(weights_path=str(mobilenet_weights)),
            )
        )
    else:
        print(f"Warning: MobileNetV2 weights not found at {mobilenet_weights}")

    efficientnet_weights = weights_dir / "EfficientNetB1_finetuned.pth"
    if efficientnet_weights.exists():
        print(
            f"Initializing EfficientNetB1 with fine-tuned weights: {efficientnet_weights}"
        )
        extractors.append(
            (
                "EfficientNetB1_finetuned",
                EfficientNetB1Extractor(weights_path=str(efficientnet_weights)),
            )
        )
    else:
        print(f"Warning: EfficientNetB1 weights not found at {efficientnet_weights}")

    if not extractors:
        print("Error: No fine-tuned weights found!")
        print(f"Expected weights in: {weights_dir}")
        print("  - ResNet50_finetuned.pth")
        print("  - MobileNetV2_finetuned.pth")
        print("  - EfficientNetB1_finetuned.pth")
        sys.exit(1)

    print(f"\nExtracting features with {len(extractors)} fine-tuned models...")

    results = []

    for idx, metadata in enumerate(metadata_list):
        image_id = metadata["id"]
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


def main():
    """Main function to extract fine-tuned features."""
    # Output path for fine-tuned features
    output_path = SEGMENTED_IMAGE_DIR.parent / "finetuned_dl_features.json"

    print("=" * 60)
    print("Fine-Tuned Deep Learning Feature Extraction")
    print("=" * 60)
    print(f"Segmented images: {SEGMENTED_IMAGE_DIR}")
    print(f"Metadata: {SEGMENTED_METADATA_PATH}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Output: {output_path}")
    print("=" * 60 + "\n")

    extract_finetuned_features(
        segmented_image_path=SEGMENTED_IMAGE_DIR,
        metadata_path=SEGMENTED_METADATA_PATH,
        weights_dir=WEIGHTS_DIR,
        output_json_path=output_path,
    )


if __name__ == "__main__":
    main()
