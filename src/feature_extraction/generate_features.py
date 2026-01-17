import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import FEATURES_JSON_PATH, SEGMENTED_IMAGE_DIR, SEGMENTED_METADATA_PATH
from src.feature_extraction.feature_extractors import (
    ColorHistogramExtractor,
    ColorHistogramHSconcatResnet50,
    ColorHistogramHSExtractor,
    EfficientNetB1Extractor,
    GaborExtractor,
    HOGExtractor,
    MobileNetV2Extractor,
    ResNet50Extractor,
)


def generate_features(
    image_dir: Path = SEGMENTED_IMAGE_DIR,
    metadata_path: Path = SEGMENTED_METADATA_PATH,
    output_path: Path = FEATURES_JSON_PATH,
) -> None:

    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} not found.")
        return

    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata.")

    # Initialize extractors
    extractors = [
        ColorHistogramHSconcatResnet50(),  # Added back the combined extractor
        ResNet50Extractor(),
        MobileNetV2Extractor(),
        EfficientNetB1Extractor(),
        HOGExtractor(),
        GaborExtractor(),
        ColorHistogramExtractor(),
        ColorHistogramHSExtractor(),
    ]

    features_data = []

    for item in tqdm(metadata_list, desc="Extracting features"):
        image_id = item["id"]
        image_path = image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            continue

        # Read image
        # Note: Deep learning extractors usually handle
        # reading/transforming internally or expect a path/PIL image.
        # My BaseDeepLearningExtractor.extract takes an image path.
        # The traditional ones take a numpy array (cv2 image).

        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            continue

        record = {"id": image_id, "features": {}}

        for extractor in extractors:
            try:
                if hasattr(extractor, "extract"):
                    vector = extractor.extract(img_cv2)

                    # Convert to list for JSON serialization
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    elif isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()

                    record["features"][extractor.name.lower()] = {
                        "vector": vector,
                        "dimension": len(vector),
                    }
            except Exception as e:
                print(f"Error extracting {extractor.name} for {image_id}: {e}")

        features_data.append(record)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(features_data, f)

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    generate_features()
