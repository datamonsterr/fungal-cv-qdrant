import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.prepare.init_yolo import draw_bboxes, kmeans_bboxes

from src.config import (
    DATASET_ROOT,
    HEIGHT,
    ORIGINAL_DATASET_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
    WIDTH,
    relative_to_workspace,
)
from src.preprocessing.preprocess import prepare_image

POSSIBLE_ENVIRONMENTS = []
POSSIBLE_ANGLES = ["ob", "rev"]
FILE_EXTENSION = ".jpg"
OUTPUT_ROOT = DATASET_ROOT / "all"
IMAGES_ROOT = OUTPUT_ROOT / "images"
FULL_IMAGE_METADATA_PATH = OUTPUT_ROOT / "all_metadata.json"
DEFAULT_YOLO_WEIGHTS_PATH = (
    DATASET_ROOT.parent / "weights" / "segmentation" / "yolo_segmentation_best.pt"
)
COLONY_COLOURS_BGR = [(0, 80, 255), (0, 220, 80), (255, 80, 80)]


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
    def __init__(
        self,
        filename: str,
        id: str,
        strain_to_specy: pd.DataFrame,
        source_path: Path | None = None,
    ):
        self.filename = filename
        self._clean_filename()
        self.id = id

        self.strain = "unknown"
        self.environment = "unknown"
        self.angle = "unknown"
        self.specy = "unknown"

        if source_path is not None and self._apply_path_metadata(source_path):
            return

        match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", self.filename)
        if match:
            self.strain = match.group(1)
            self.environment = match.group(2)
            self.angle = match.group(3)
            self.specy = (
                get_specy_from_strain(self.strain, strain_to_specy) or "unknown"
            )

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

    def _apply_path_metadata(self, source_path: Path) -> bool:
        parts = source_path.parts
        if "new_data" not in parts:
            return False

        new_data_index = parts.index("new_data")
        if len(parts) <= new_data_index + 3:
            return False

        self.specy = parts[new_data_index + 2]
        self.strain = parts[new_data_index + 3]

        stem = Path(self.filename).stem if "." in self.filename else self.filename
        tokens = [token for token in re.split(r"[\s_]+", stem) if token]
        angle_map = {"ob": "ob", "rev": "rev"}
        environment_tokens: list[str] = []

        for token in tokens:
            token_lower = token.lower()
            if token_lower in angle_map:
                self.angle = angle_map[token_lower]
                break
            environment_tokens.append(token)

        if environment_tokens:
            self.environment = environment_tokens[-1]

        return True


def _species_slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def _strain_slug(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def _image_dir(metadata: Metadata, source_name: str) -> Path:
    return (
        IMAGES_ROOT
        / _species_slug(metadata.specy)
        / _strain_slug(metadata.strain)
        / metadata.environment
    )


def _image_stem(metadata: Metadata, id: str) -> str:
    clean_strain = _strain_slug(metadata.strain)
    return f"{clean_strain}_{metadata.angle}_{id[:8]}"


def _write_image(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return relative_to_workspace(path)


def _save_segment(
    metadata: Metadata,
    source_name: str,
    id: str,
    segment_img,
    segment_index: int,
) -> str:
    file_path = _image_dir(metadata, source_name) / (
        f"{_image_stem(metadata, id)}_seg{segment_index}{FILE_EXTENSION}"
    )
    return _write_image(file_path, segment_img)


def _save_step_image(
    metadata: Metadata,
    source_name: str,
    id: str,
    suffix: str,
    image,
) -> str:
    file_path = _image_dir(metadata, source_name) / (
        f"{_image_stem(metadata, id)}_{suffix}{FILE_EXTENSION}"
    )
    return _write_image(file_path, image)


def _save_yolo_bounding_box_image(
    metadata: Metadata,
    source_name: str,
    id: str,
    image,
    detections: list[Detection],
) -> str:
    bbox_img = image.copy()
    for index, det in enumerate(detections):
        colour = COLONY_COLOURS_BGR[index % len(COLONY_COLOURS_BGR)]
        cv2.rectangle(bbox_img, (det.x1, det.y1), (det.x2, det.y2), colour, 3)
    return _save_step_image(metadata, source_name, id, "yolo", bbox_img)


def _save_kmeans_bounding_box_image(
    metadata: Metadata,
    source_name: str,
    id: str,
    image,
) -> tuple[str, list[dict[str, int]]]:
    bboxes = kmeans_bboxes(image)
    bbox_img = draw_bboxes(image, bboxes)
    return _save_step_image(metadata, source_name, id, "bboxes", bbox_img), bboxes


def _iter_image_directories(source_path: Path) -> list[Path]:
    image_dirs: list[Path] = []

    if any(
        child.is_file() and child.suffix.lower() == FILE_EXTENSION
        for child in source_path.iterdir()
    ):
        image_dirs.append(source_path)

    for path in source_path.rglob("*"):
        if not path.is_dir():
            continue
        if any(
            child.is_file() and child.suffix.lower() == FILE_EXTENSION
            for child in path.iterdir()
        ):
            image_dirs.append(path)

    return image_dirs


def _process_image(
    original_img_path: Path,
    source_path: Path,
    strain_to_specy: pd.DataFrame,
    model: YOLO,
    confidence_threshold: float,
    source_name: str,
) -> dict[str, object] | None:
    source_key = str(original_img_path.resolve().relative_to(source_path.parent))
    id = uuid.uuid5(uuid.NAMESPACE_URL, source_key).hex
    metadata = Metadata(
        original_img_path.name,
        id,
        strain_to_specy,
        source_path=original_img_path,
    )

    print(f"Processing {original_img_path}...")

    original_img = cv2.imread(str(original_img_path))
    if original_img is None:
        print(f"Failed to read {original_img_path}")
        return None

    artifacts = prepare_image(original_img, export_size=max(WIDTH, HEIGHT))
    centered_img = artifacts.export_image
    preprocessed_img = artifacts.masked_export_image

    results = model(preprocessed_img, conf=confidence_threshold, verbose=False)
    detections = sorted(
        yolo_to_detections(results),
        key=lambda detection: detection.confidence,
        reverse=True,
    )[:3]
    if not detections:
        return None

    original_rel = _save_step_image(metadata, source_name, id, "original", original_img)
    centered_rel = _save_step_image(metadata, source_name, id, "centered", centered_img)
    processed_rel = _save_step_image(
        metadata, source_name, id, "processed", preprocessed_img
    )
    kmeans_rel, kmeans_bbox_records = _save_kmeans_bounding_box_image(
        metadata, source_name, id, preprocessed_img
    )
    yolo_rel = _save_yolo_bounding_box_image(
        metadata, source_name, id, preprocessed_img, detections
    )

    segment_paths: list[str] = []
    bbox_records: list[dict[str, int]] = []
    for i, det in enumerate(detections):
        x1 = max(0, det.x1)
        y1 = max(0, det.y1)
        x2 = min(preprocessed_img.shape[1], det.x2)
        y2 = min(preprocessed_img.shape[0], det.y2)

        segment_img = preprocessed_img[y1:y2, x1:x2]
        if segment_img.size == 0:
            continue

        segment_img = cv2.resize(segment_img, (WIDTH, HEIGHT))
        segment_paths.append(
            _save_segment(
                metadata=metadata,
                source_name=source_name,
                id=id,
                segment_img=segment_img,
                segment_index=i,
            )
        )
        bbox_records.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2})

    if not segment_paths:
        return None

    full_metadata = {
        "id": id,
        "source": source_name,
        "source_path": relative_to_workspace(original_img_path),
        "file_path": processed_rel,
        "step_images": {
            "original": original_rel,
            "centered": centered_rel,
            "preprocessed": processed_rel,
            "bboxes": kmeans_rel,
            "yolo": yolo_rel,
        },
        "data": {
            "species": metadata.specy,
            "strain": metadata.strain,
            "environment": metadata.environment,
            "angle": metadata.angle,
            "bboxes": kmeans_bbox_records,
            "yolo_bboxes": bbox_records,
            "num_colonies": len(segment_paths),
            "segment_paths": segment_paths,
        },
    }
    return full_metadata


def reformat_dataset(
    sources: list[str] | None = None,
    local_model_path: str | None = None,
    confidence_threshold: float = 0.25,
    source_name: str = "new_data",
):
    if os.path.exists(STRAIN_SPECIES_MAPPING_PATH):
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    else:
        print(f"Warning: {STRAIN_SPECIES_MAPPING_PATH} not found.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    model_path = Path(local_model_path or DEFAULT_YOLO_WEIGHTS_PATH)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")

    print(f"Loading model from local path: {model_path}")
    model = YOLO(str(model_path))

    metadata_list: list[dict[str, object]] = []

    source_values = sources or [str(ORIGINAL_DATASET_PATH)]
    source_paths = [Path(source).resolve() for source in source_values]

    for source_path in source_paths:
        if not source_path.exists():
            print(f"Error: {source_path} does not exist.")
            continue

        for dir_path in _iter_image_directories(source_path):
            for filename in os.listdir(dir_path):
                if not filename.endswith(FILE_EXTENSION):
                    continue

                full_metadata = _process_image(
                    original_img_path=dir_path / filename,
                    source_path=source_path,
                    strain_to_specy=strain_to_specy,
                    model=model,
                    confidence_threshold=confidence_threshold,
                    source_name=source_name,
                )
                if full_metadata is not None:
                    metadata_list.append(full_metadata)

    with open(FULL_IMAGE_METADATA_PATH, "w") as f:
        json.dump({"images": metadata_list}, f, indent=2)

    print("\nProcessing complete!")
    print(f"Images: {len(metadata_list)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation on dataset sources"
    )
    parser.add_argument(
        "--source",
        type=str,
        action="append",
        help="Source directory containing species folders or direct images",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=str(DEFAULT_YOLO_WEIGHTS_PATH),
        help="Path to local YOLO weights file",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="new_data",
        help="Dataset/all/images source folder name",
    )

    args = parser.parse_args()

    reformat_dataset(
        sources=args.source,
        local_model_path=args.local_path,
        confidence_threshold=args.confidence,
        source_name=args.source_name,
    )
