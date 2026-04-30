from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.config import DATASET_ROOT, RESULTS_DIR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def iter_dataset_images(images_root: Path) -> list[Path]:
    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def parse_yolo_bbox_line(
    line: str,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line: {line}")

    class_id = int(float(parts[0]))
    coords = [float(value) for value in parts[1:]]

    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        xmin = max(0, min(xs) * image_width)
        xmax = min(image_width, max(xs) * image_width)
        ymin = max(0, min(ys) * image_height)
        ymax = min(image_height, max(ys) * image_height)
    else:
        x_center, y_center, width, height = coords[:4]
        xmin = max(0, (x_center - width / 2) * image_width)
        xmax = min(image_width, (x_center + width / 2) * image_width)
        ymin = max(0, (y_center - height / 2) * image_height)
        ymax = min(image_height, (y_center + height / 2) * image_height)

    return {
        "class_id": class_id,
        "xmin": int(round(xmin)),
        "ymin": int(round(ymin)),
        "xmax": max(int(round(xmax)), int(round(xmin)) + 1),
        "ymax": max(int(round(ymax)), int(round(ymin)) + 1),
    }


def create_crop_dataset(
    source_dataset_root: Path,
    output_root: Path,
    crop_size: int = 224,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {"train": 0, "test": 0}

    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "test"]:
        images_root = source_dataset_root / split_name / "images"
        labels_root = source_dataset_root / split_name / "labels"
        if not images_root.exists():
            continue

        for image_path in iter_dataset_images(images_root):
            relative = image_path.relative_to(images_root)
            label_path = labels_root / relative.with_suffix(".txt")
            if not label_path.exists():
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            for crop_index, raw_line in enumerate(label_path.read_text().splitlines()):
                line = raw_line.strip()
                if not line:
                    continue
                bbox = parse_yolo_bbox_line(line, image.shape[1], image.shape[0])
                crop = image[bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]]
                if crop.size == 0:
                    continue
                resized = cv2.resize(
                    crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                )

                class_id = bbox["class_id"]
                target_dir = output_root / split_name / str(class_id)
                target_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"{image_path.stem}__crop{crop_index}.jpg"
                crop_path = target_dir / crop_name
                if not cv2.imwrite(str(crop_path), resized):
                    raise RuntimeError(f"Failed to write crop image: {crop_path}")

                split_counts[split_name] = split_counts.get(split_name, 0) + 1
                rows.append(
                    {
                        "split_name": split_name,
                        "source_image_path": str(image_path),
                        "source_label_path": str(label_path),
                        "crop_image_path": str(crop_path),
                        "class_id": class_id,
                        "xmin": bbox["xmin"],
                        "ymin": bbox["ymin"],
                        "xmax": bbox["xmax"],
                        "ymax": bbox["ymax"],
                    }
                )

    assignment_path = output_root / "crop_assignment.csv"
    pd.DataFrame(rows).to_csv(assignment_path, index=False)

    return {
        "source_dataset_root": str(source_dataset_root),
        "crop_dataset_root": str(output_root),
        "assignment_path": str(assignment_path),
        "train_count": split_counts.get("train", 0),
        "test_count": split_counts.get("test", 0),
        "crop_size": crop_size,
    }


def default_crop_output_root(dataset_root: Path) -> Path:
    if dataset_root.name.startswith("fold_"):
        return dataset_root.parent / f"{dataset_root.name}_crops"
    return DATASET_ROOT / f"{dataset_root.name}_crops"


def build_crop_dataset_summary(summary: dict[str, Any]) -> Path:
    results_root = RESULTS_DIR / "cross_validation_yolo"
    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / "crop_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary_path
