import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import FEATURES_JSON_PATH, SEGMENTED_METADATA_PATH, WORKSPACE_ROOT
from src.experiments.feature_extraction.feature_extractors import (
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
    metadata_path: Path = SEGMENTED_METADATA_PATH,
    output_path: Path = FEATURES_JSON_PATH,
    image_dir: Path | None = None,
) -> None:

    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} not found.")
        return

    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    print(f"Found {len(metadata_list)} images in metadata.")

    extractors = [
        ColorHistogramHSconcatResnet50(),
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
        segment_id = item.get("segment_id") or item.get("id")
        segment_path = item.get("segment_path")
        if not segment_id:
            continue

        if segment_path:
            image_path = WORKSPACE_ROOT / segment_path
        elif image_dir is not None:
            image_path = image_dir / f"{segment_id}.jpg"
        else:
            continue

        if not image_path.exists():
            continue

        img_cv2 = cv2.imread(str(image_path))
        if img_cv2 is None:
            continue

        record = {"id": segment_id, "features": {}}

        for extractor in extractors:
            try:
                if hasattr(extractor, "extract"):
                    vector = extractor.extract(img_cv2)

                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    elif isinstance(vector, torch.Tensor):
                        vector = vector.cpu().numpy().tolist()

                    record["features"][extractor.name.lower()] = {
                        "vector": vector,
                        "dimension": len(vector),
                    }
            except Exception as e:
                print(f"Error extracting {extractor.name} for {segment_id}: {e}")

        features_data.append(record)

    with open(output_path, "w") as f:
        json.dump(features_data, f)

    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    generate_features()
