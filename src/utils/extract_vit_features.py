"""
Extract features using Vision Transformer (ViT) models with various pretrained weights.

This script extracts features using ViT models pretrained on:
- CellViT (nuclei segmentation on microscopy)
- SAM (Segment Anything Model encoder-only)
- ViT-256 DINO (self-supervised learning)

The extracted features are saved to a JSON file for upload to Qdrant.
"""

import json
import sys
from pathlib import Path
from typing import List

import cv2

from src.config import DATASET_ROOT, SEGMENTED_IMAGE_DIR, SEGMENTED_METADATA_PATH, WORKSPACE_ROOT
from src.experiments.feature_extraction.feature_extractors import (
    ViT256DinoExtractor,
    ViTCellVitX20Extractor,
    ViTCellVitX40Extractor,
    ViTSAMBExtractor,
    ViTSAMHExtractor,
    ViTSAMLExtractor,
)


def extract_vit_features(
    output_json_path: str = str(DATASET_ROOT / "vit_features.json"),
    weights_type: str = "vit256_dino",
) -> None:
    """
    Extract ViT features from all segmented images.

    Args:
        output_json_path: Path to save the extracted features JSON
        weights_type: Type of ViT weights to use (cellvit_x20, cellvit_x40, sam_vit_b, sam_vit_l, sam_vit_h, vit256_dino)
    """
    # Select extractor based on weights type
    extractor_map = {
        "cellvit_x20": ViTCellVitX20Extractor,
        "cellvit_x40": ViTCellVitX40Extractor,
        "sam_vit_b": ViTSAMBExtractor,
        "sam_vit_l": ViTSAMLExtractor,
        "sam_vit_h": ViTSAMHExtractor,
        "vit256_dino": ViT256DinoExtractor,
    }

    if weights_type not in extractor_map:
        print(f"Error: Unknown weights type '{weights_type}'")
        print(f"Available options: {list(extractor_map.keys())}")
        sys.exit(1)

    print(f"Initializing ViT extractor with {weights_type} weights...")
    extractor = extractor_map[weights_type]()

    # Load metadata
    with open(SEGMENTED_METADATA_PATH, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata")
    print(f"Feature extractor: {extractor.name}")
    print(f"Feature dimension: {extractor._feature_dim}")

    results: List[dict] = []
    processed = 0
    skipped = 0

    for idx, metadata in enumerate(metadata_list):
        image_id = metadata.get("segment_id") or metadata["id"]
        segment_path = metadata.get("segment_path")
        if segment_path:
            image_path = WORKSPACE_ROOT / segment_path
        else:
            image_path = SEGMENTED_IMAGE_DIR / f"{image_id}.jpg"

        if not image_path.exists():
            print(f"Warning: Image {image_path} not found, skipping...")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            print(f"Warning: Failed to read {image_path}, skipping...")
            skipped += 1
            continue

        try:
            features = extractor.extract(image)

            feature_data = {
                "id": image_id,
                "features": {
                    extractor.name.lower(): {
                        "vector": features.tolist(),
                        "dimension": len(features),
                    }
                },
            }

            results.append(feature_data)
            processed += 1

            if (idx + 1) % 10 == 0:
                print(f"Processed {processed}/{len(metadata_list)} images...")

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            skipped += 1
            continue

    # Save results
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ViT Feature Extraction Complete!")
    print("=" * 60)
    print(f"Processed: {processed} images")
    print(f"Skipped: {skipped} images")
    print(f"Feature type: {extractor.name}")
    print(f"Feature dimension: {extractor._feature_dim}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


def main():
    """Main function with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features using Vision Transformer models"
    )
    parser.add_argument(
        "--weights-type",
        type=str,
        default="vit256_dino",
        choices=[
            "cellvit_x20",
            "cellvit_x40",
            "sam_vit_b",
            "sam_vit_l",
            "sam_vit_h",
            "vit256_dino",
        ],
        help="Type of ViT pretrained weights to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASET_ROOT / "vit_features.json"),
        help="Output JSON file path",
    )

    args = parser.parse_args()

    extract_vit_features(output_json_path=args.output, weights_type=args.weights_type)


if __name__ == "__main__":
    main()
