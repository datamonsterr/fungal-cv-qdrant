import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import (
    FEATURES_JSON_PATH,
    COLLECTION_METADATA_PATHS,
    WORKSPACE_ROOT,
)
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


def _load_all_items(metadata_paths: dict[str, Path]) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    for path in metadata_paths.values():
        if not path.exists():
            continue
        with open(path, "r") as f:
            items = json.load(f)
            for item in items:
                item["_metadata_source"] = str(path)
            all_items.extend(items)
    return all_items


def generate_features(
    metadata_path: Path | None = None,
    output_path: Path = FEATURES_JSON_PATH,
    image_dir: Path | None = None,
) -> None:
    if metadata_path is not None:
        if not metadata_path.exists():
            print(f"Error: Metadata file {metadata_path} not found.")
            return
        with open(metadata_path, "r") as f:
            items = json.load(f)
    else:
        items = _load_all_items(COLLECTION_METADATA_PATHS)

    if not items:
        print("No items found in metadata.")
        return

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

    features_data: list[dict[str, Any]] = []
    segment_count = 0

    for item in tqdm(items, desc="Items"):
        item_id = item.get("item_id", "")
        segment_paths = item.get("paths", {}).get("segments", [])
        if not segment_paths:
            continue

        for idx, seg_path in enumerate(segment_paths):
            image_path = WORKSPACE_ROOT / seg_path
            if not image_path.exists():
                continue

            img_cv2 = cv2.imread(str(image_path))
            if img_cv2 is None:
                continue

            segment_id = f"{item_id}_seg{idx}"
            record: dict[str, Any] = {"id": segment_id, "features": {}}

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
            segment_count += 1

    with open(output_path, "w") as f:
        json.dump(features_data, f)

    print(f"Extracted features for {segment_count} segments from {len(items)} items")
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    generate_features()
