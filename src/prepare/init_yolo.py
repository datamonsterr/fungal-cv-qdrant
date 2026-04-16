import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from src.config import DATASET_ROOT, ORIGINAL_DATASET_PATH, STRAIN_SPECIES_MAPPING_PATH
from src.preprocessing.kmeans import segment_kmeans
from src.preprocessing.preprocess import process_image

YOLO_DATASET_PATH = DATASET_ROOT / "yolo"
FILE_EXTENSION = ".jpg"


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> str | None:
    result = strain_to_specy[strain_to_specy["Strain"] == strain]
    if not result.empty:
        return result["Species"].iloc[0]
    return None


def parse_filename(filename: str) -> dict[str, str]:
    clean = filename.removesuffix(FILE_EXTENSION).removesuffix("_edited")
    match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", clean)
    if match:
        return {
            "strain": match.group(1),
            "environment": match.group(2),
            "angle": match.group(3),
        }
    return {"strain": "unknown", "environment": "unknown", "angle": "unknown"}


def bboxes_to_yolo(
    bboxes: list[dict[str, int]], img_w: int, img_h: int
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for label_id, bbox in enumerate(bboxes):
        x_min, y_min = bbox["xmin"], bbox["ymin"]
        x_max, y_max = bbox["xmax"], bbox["ymax"]
        bw = max(0, x_max - x_min)
        bh = max(0, y_max - y_min)
        x_center = (x_min + bw / 2) / img_w
        y_center = (y_min + bh / 2) / img_h
        annotations.append(
            {
                "label_id": label_id,
                "x_center": round(x_center, 6),
                "y_center": round(y_center, 6),
                "width": round(bw / img_w, 6),
                "height": round(bh / img_h, 6),
            }
        )
    return annotations


def _contour_circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter**2)


def contour_bboxes(preprocessed: Any) -> list[dict[str, int]]:
    gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blur, 30, 80)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        circ = _contour_circularity(cnt)
        if circ < 0.1:
            continue
        scored.append((area * circ, cnt))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [cnt for _, cnt in scored[:3]]

    bboxes: list[dict[str, int]] = []
    for cnt in selected:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append(
            {"xmin": int(x), "ymin": int(y), "xmax": int(x + w), "ymax": int(y + h)}
        )
    return bboxes


def kmeans_bboxes(preprocessed: Any) -> list[dict[str, int]]:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
    try:
        cv2.imwrite(temp_path, preprocessed)
        return segment_kmeans(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def draw_bboxes(image: Any, bboxes: list[dict[str, int]]) -> Any:
    out = image.copy()
    colors = [(0, 80, 255), (0, 220, 80), (255, 80, 80)]
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        cv2.rectangle(
            out,
            (bbox["xmin"], bbox["ymin"]),
            (bbox["xmax"], bbox["ymax"]),
            color,
            2,
        )
    return out


def save_pipeline_visualization(
    source_img: Any,
    preprocessed_img: Any,
    bbox_img: Any,
    out_path: Path,
) -> None:
    h, w = preprocessed_img.shape[:2]
    src = cv2.resize(source_img, (w, h), interpolation=cv2.INTER_AREA)
    panel = cv2.hconcat([src, preprocessed_img, bbox_img])
    cv2.imwrite(str(out_path), panel)


def process_single_image(
    image_path: str,
    output_base: Path,
    strain_to_specy: pd.DataFrame,
    method: str,
    include_bbox: bool,
    full_process: bool,
    include_original: bool,
) -> dict[str, Any] | None:
    source = cv2.imread(image_path)
    if source is None:
        print(f"  [ERROR] Cannot read {image_path}")
        return None

    filename = Path(image_path).name
    parsed = parse_filename(filename)
    strain = parsed["strain"]
    environment = parsed["environment"]
    angle = parsed["angle"]
    specy = get_specy_from_strain(strain, strain_to_specy) or "unknown_species"

    preprocessed = process_image(source)

    if method == "contour":
        bboxes = contour_bboxes(preprocessed)
    else:
        bboxes = kmeans_bboxes(preprocessed)

    target_dir = output_base / specy / strain / environment
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_strain = strain.replace(" ", "_").replace("/", "-")
    stem = f"{clean_strain}_{environment}_{angle}"

    processed_path = target_dir / f"{stem}_processed{FILE_EXTENSION}"
    cv2.imwrite(str(processed_path), preprocessed)

    original_rel: str | None = None
    if include_original:
        original_path = target_dir / f"{stem}_original{FILE_EXTENSION}"
        cv2.imwrite(str(original_path), source)
        original_rel = str(original_path.relative_to(output_base))

    bbox_img_rel: str | None = None
    pipeline_rel: str | None = None

    bbox_img = draw_bboxes(preprocessed, bboxes)

    if include_bbox:
        bbox_path = target_dir / f"{stem}_bbox{FILE_EXTENSION}"
        cv2.imwrite(str(bbox_path), bbox_img)
        bbox_img_rel = str(bbox_path.relative_to(output_base))

    if full_process:
        pipeline_path = target_dir / f"{stem}_pipeline{FILE_EXTENSION}"
        save_pipeline_visualization(source, preprocessed, bbox_img, pipeline_path)
        pipeline_rel = str(pipeline_path.relative_to(output_base))

    h, w = preprocessed.shape[:2]
    record: dict[str, Any] = {
        "image": str(processed_path.relative_to(output_base)),
        "method": method,
        "width": w,
        "height": h,
        "metadata": {
            "strain": strain,
            "environment": environment,
            "angle": angle,
            "specy": specy,
        },
        "annotations": bboxes_to_yolo(bboxes, w, h),
    }

    if original_rel is not None:
        record["original_image"] = original_rel
    if bbox_img_rel is not None:
        record["bbox_image"] = bbox_img_rel
    if pipeline_rel is not None:
        record["pipeline_image"] = pipeline_rel

    print(f"  [OK] {stem}: {len(bboxes)} bbox(es) -> {target_dir}")
    return record


def run_full(
    output_base: Path,
    method: str,
    include_bbox: bool,
    full_process: bool,
    include_original: bool,
) -> None:
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: {ORIGINAL_DATASET_PATH} does not exist.")
        return

    if STRAIN_SPECIES_MAPPING_PATH.exists():
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
        print(f"Loaded {len(strain_to_specy)} strain-to-species mappings.")
    else:
        print("Warning: mapping not found - species will be unknown.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    if output_base.exists():
        response = input(f"{output_base} already exists. Remove and recreate? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(output_base)
        else:
            print("Aborted.")
            return

    output_base.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    stats = {"total": 0, "ok": 0, "failed": 0}

    for dir_name in sorted(os.listdir(ORIGINAL_DATASET_PATH)):
        dir_path = ORIGINAL_DATASET_PATH / dir_name
        if not dir_path.is_dir():
            continue

        print(f"\nDirectory: {dir_name}")
        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(FILE_EXTENSION):
                continue

            stats["total"] += 1
            record = process_single_image(
                image_path=str(dir_path / filename),
                output_base=output_base,
                strain_to_specy=strain_to_specy,
                method=method,
                include_bbox=include_bbox,
                full_process=full_process,
                include_original=include_original,
            )
            if record is None:
                stats["failed"] += 1
            else:
                stats["ok"] += 1
                records.append(record)

    metadata_path = output_base / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(records, f, indent=2)

    print("\n" + "=" * 60)
    print("YOLO INIT SUMMARY")
    print("=" * 60)
    print(f"Total files:   {stats['total']}")
    print(f"Processed OK:  {stats['ok']}")
    print(f"Failed:        {stats['failed']}")
    print(f"Metadata:      {metadata_path}")
    print(f"Output:        {output_base}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize YOLO dataset and metadata from Dataset/original."
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=str(YOLO_DATASET_PATH),
        help=f"Output directory (default: {YOLO_DATASET_PATH})",
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "contour"],
        default="kmeans",
        help="Bounding box method (default: kmeans)",
    )
    parser.add_argument(
        "--include-bbox",
        action="store_true",
        help="Save bbox-ready images alongside processed images.",
    )
    parser.add_argument(
        "--full-process",
        action="store_true",
        help="Save pipeline visualization image for each sample.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Also save the original source image in output folders.",
    )

    args = parser.parse_args()

    run_full(
        output_base=Path(args.output),
        method=args.method,
        include_bbox=args.include_bbox,
        full_process=args.full_process,
        include_original=args.original,
    )


if __name__ == "__main__":
    main()
