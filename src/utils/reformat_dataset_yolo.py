import json
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.config import (
    DATASET_ROOT,
    FULL_IMAGE_METADATA_PATH,
    FULL_IMAGE_PATH,
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
    HEIGHT,
    WIDTH,
    relative_to_workspace,
)

POSSIBLE_ANGLES = ["ob", "rev"]
FILE_EXTENSION = ".jpg"
HIERARCHICAL_DATASET_PATH = DATASET_ROOT / "hierarchical"


YOLO_MODEL_ID = "my-first-project-3ddqp/2"
YOLO_LOCAL_PATH = None


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int | None = None


def yolo_to_detections(results) -> list[Detection]:
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else None
            detections.append(
                Detection(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=cls,
                )
            )
    return detections


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> str | None:
    result = strain_to_specy[strain_to_specy["Strain"] == strain]
    if not result.empty:
        return result["Species"].iloc[0]
    return None


class Metadata:
    def __init__(self, filename: str, id: str, strain_to_specy: pd.DataFrame):
        self.filename = filename
        self._clean_filename()
        self.id = id
        match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", self.filename)

        self.strain = "unknown"
        self.environment = "unknown"
        self.angle = "unknown"

        if match:
            self.strain = match.group(1)
            self.environment = match.group(2)
            self.angle = match.group(3)

        self.specy = get_specy_from_strain(self.strain, strain_to_specy) or "unknown"

    def get_metadata(self) -> dict[str, dict[str, str] | str]:
        return {
            "id": self.id,
            "data": {
                "strain": self.strain,
                "environment": self.environment,
                "angle": self.angle,
                "specy": self.specy,
            },
        }

    def _clean_filename(self):
        self.filename = self.filename.removesuffix(FILE_EXTENSION)
        self.filename = self.filename.removesuffix("_edited")


def _save_hierarchical_segment(
    metadata: Metadata,
    segment_img,
    segment_index: int,
) -> str:
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}"
        f"_seg{segment_index}{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    cv2.imwrite(str(hierarchical_file_path), segment_img)
    return relative_to_workspace(hierarchical_file_path)


def _save_hierarchical_original(
    metadata: Metadata,
    image_id: str,
    full_img_path,
) -> str:
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}_{image_id}_original"
        f"{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    shutil.copyfile(full_img_path, hierarchical_file_path)
    return relative_to_workspace(hierarchical_file_path)


def run_yolo_segmentation(
    source_dir: str,
    model_id: str = YOLO_MODEL_ID,
    local_model_path: str | None = YOLO_LOCAL_PATH,
    confidence_threshold: float = 0.25,
    create_hierarchical: bool = True,
) -> tuple[list[dict], list[dict]]:
    if local_model_path:
        print(f"Loading model from local path: {local_model_path}")
        model = YOLO(local_model_path)
    else:
        try:
            from inference import get_model

            model = get_model(model_id=model_id)
        except ImportError:
            raise RuntimeError(
                f"inference SDK not available for Python {sys.version}. "
                f"Please provide --local-path to your YOLO weights file."
            )

    if os.path.exists(STRAIN_SPECIES_MAPPING_PATH):
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    else:
        print(f"Warning: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    FULL_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    SEGMENTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not FULL_IMAGE_METADATA_PATH.exists():
        with open(FULL_IMAGE_METADATA_PATH, "w") as f:
            json.dump([], f)

    if not SEGMENTED_METADATA_PATH.exists():
        with open(SEGMENTED_METADATA_PATH, "w") as f:
            json.dump([], f)

    metadata_list: list[dict[str, dict[str, str] | str]] = []
    segment_metadata_list: list[dict[str, dict[str, str] | str]] = []

    source_path = Path(source_dir)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path

    if not source_path.exists():
        print(f"Error: {source_path} does not exist.")
        return [], []

    for dir_name in os.listdir(source_path):
        dir_path = source_path / dir_name
        if not dir_path.is_dir():
            continue

        for filename in os.listdir(dir_path):
            if not filename.endswith(FILE_EXTENSION):
                continue

            id = uuid.uuid5(uuid.NAMESPACE_DNS, filename).hex
            metadata = Metadata(filename, id, strain_to_specy)

            original_img_path = dir_path / filename
            full_img_path = FULL_IMAGE_PATH / f"{id}{FILE_EXTENSION}"

            shutil.copyfile(original_img_path, full_img_path)
            full_metadata = metadata.get_metadata()
            full_metadata["file_path"] = relative_to_workspace(full_img_path)
            if create_hierarchical:
                full_metadata["hierarchical_path"] = _save_hierarchical_original(
                    metadata=metadata,
                    image_id=id,
                    full_img_path=full_img_path,
                )
            metadata_list.append(full_metadata)

            print(f"Processing {filename}...")

            img = cv2.imread(str(full_img_path))
            if img is None:
                print(f"Failed to read {full_img_path}")
                continue

            if local_model_path:
                results = model(full_img_path, conf=confidence_threshold)
                detections = yolo_to_detections(results)
            else:
                from inference import get_model

                model = get_model(model_id=model_id)
                results = model.infer(str(full_img_path))[0]
                detections = inference_to_detections(results)

            for i, det in enumerate(detections):
                x1 = max(0, det.x1)
                y1 = max(0, det.y1)
                x2 = min(img.shape[1], det.x2)
                y2 = min(img.shape[0], det.y2)

                segment_id = f"{id}_{i}"
                segment_img = img[y1:y2, x1:x2]

                if segment_img.size == 0:
                    continue

                segment_img = cv2.resize(segment_img, (WIDTH, HEIGHT))

                segment_path = SEGMENTED_IMAGE_DIR / f"{segment_id}{FILE_EXTENSION}"
                cv2.imwrite(str(segment_path), segment_img)

                seg_metadata = metadata.get_metadata()
                seg_metadata["id"] = segment_id
                seg_metadata["parent_id"] = id
                seg_metadata["confidence"] = det.confidence
                seg_metadata["class_id"] = det.class_id
                seg_metadata["file_path"] = relative_to_workspace(segment_path)

                if create_hierarchical:
                    seg_metadata["hierarchical_path"] = _save_hierarchical_segment(
                        metadata=metadata,
                        segment_img=segment_img,
                        segment_index=i,
                    )

                segment_metadata_list.append(seg_metadata)

    with open(FULL_IMAGE_METADATA_PATH, "w") as f:
        json.dump(metadata_list, f, indent=4)

    with open(SEGMENTED_METADATA_PATH, "w") as f:
        json.dump(segment_metadata_list, f, indent=4)

    print("\nProcessing complete!")
    print(f"Full images: {len(metadata_list)}")
    print(f"Segmented images: {len(segment_metadata_list)}")

    return metadata_list, segment_metadata_list


def inference_to_detections(inference_result) -> list[Detection]:
    detections = []
    if hasattr(inference_result, "detections"):
        for det in inference_result.detections:
            detections.append(
                Detection(
                    x1=int(det.x_min),
                    y1=int(det.y_min),
                    x2=int(det.x_max),
                    y2=int(det.y_max),
                    confidence=det.confidence,
                    class_id=det.class_id,
                )
            )
    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO segmentation on images")
    parser.add_argument(
        "--source",
        type=str,
        default="Dataset/new_data",
        help="Source directory containing images",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=YOLO_MODEL_ID,
        help="YOLO model ID (workspace/model/version)",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to local YOLO weights file (e.g., weights/best.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--no-hierarchical",
        action="store_true",
        help="Skip creating hierarchical directory structure",
    )

    args = parser.parse_args()

    run_yolo_segmentation(
        source_dir=args.source,
        model_id=args.model_id,
        local_model_path=args.local_path,
        confidence_threshold=args.confidence,
        create_hierarchical=not args.no_hierarchical,
    )
