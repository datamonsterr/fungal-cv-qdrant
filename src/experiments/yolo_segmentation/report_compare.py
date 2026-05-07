"""YOLOv26 inference + kmeans comparison + visualization for report."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from src.config import PREPARED_DATASET_DIR, WORKSPACE_ROOT
from src.preprocessing.kmeans import draw_bbox, segment_kmeans_image
from ultralytics import YOLO

WEIGHTS_PATH = WORKSPACE_ROOT / "weights" / "yolo26" / "yolo26n-seg_species_best.pt"
OUTPUT_ROOT = WORKSPACE_ROOT / "results" / "yolo26_comparison"
N_SAMPLES = 25
CONFIDENCE = 0.25
SEED = 42


def collect_prepared_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("prepared.jpg") if p.is_file()
    )


def yolo26_detect(model, image: np.ndarray, conf: float):
    results = model(image, conf=conf, verbose=False)
    bboxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        h, w = image.shape[:2]
        for box in results[0].boxes:
            cx, cy, bw, bh = box.xywhn[0].tolist()
            px = int((cx - bw / 2) * w)
            py = int((cy - bh / 2) * h)
            pw = int(bw * w)
            ph = int(bh * h)
            bboxes.append({
                "x": max(0, px), "y": max(0, py),
                "w": pw, "h": ph,
                "conf": round(float(box.conf[0]), 3),
            })
    bboxes.sort(key=lambda b: b["conf"], reverse=True)
    return bboxes[:3]


def kmeans_detect(prepared_img: np.ndarray):
    bboxes, labels = segment_kmeans_image(prepared_img)
    result = []
    for b in (bboxes or []):
        if "xmin" in b:
            result.append({
                "x": int(b["xmin"]), "y": int(b["ymin"]),
                "w": int(b["xmax"] - b["xmin"]), "h": int(b["ymax"] - b["ymin"]),
                "conf": 1.0,
            })
    return result[:3]


def build_comparison_image(
    source_img: np.ndarray | None,
    prepared_img: np.ndarray,
    yolo_bboxes: list[dict],
    kmeans_bboxes: list[dict],
) -> np.ndarray:
    h, w = prepared_img.shape[:2]
    row1 = prepared_img.copy()
    row2 = prepared_img.copy()

    if yolo_bboxes:
        xyxy = [{"xmin": b["x"], "ymin": b["y"], "xmax": b["x"]+b["w"], "ymax": b["y"]+b["h"]} for b in yolo_bboxes]
        row1 = draw_bbox(row1, xyxy)
    cv2.putText(row1, f"YOLOv26 ({len(yolo_bboxes)} det)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if kmeans_bboxes:
        xyxy = [{"xmin": b["x"], "ymin": b["y"], "xmax": b["x"]+b["w"], "ymax": b["y"]+b["h"]} for b in kmeans_bboxes]
        row2 = draw_bbox(row2, xyxy)
    cv2.putText(row2, f"KMeans ({len(kmeans_bboxes)} det)", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    comparison = np.hstack([row1, row2])
    return comparison


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    images = collect_prepared_images(PREPARED_DATASET_DIR)
    print(f"Total prepared images: {len(images)}")

    random.seed(SEED)
    samples = random.sample(images, min(N_SAMPLES, len(images)))
    print(f"Selected {len(samples)} samples")

    model = YOLO(str(WEIGHTS_PATH))
    print(f"Loaded model: {WEIGHTS_PATH}")

    summary = {"yolo26_detections": 0, "kmeans_detections": 0, "samples": []}

    for i, img_path in enumerate(samples):
        leaf_dir = img_path.parent
        species = img_path.relative_to(PREPARED_DATASET_DIR).parts[0]
        prepared = cv2.imread(str(img_path))
        if prepared is None:
            continue

        source_path = leaf_dir / "source.jpg"
        source = cv2.imread(str(source_path)) if source_path.exists() else None

        yolo_bboxes = yolo26_detect(model, prepared, CONFIDENCE)
        kmeans_bboxes = kmeans_detect(prepared)
        comparison = build_comparison_image(source, prepared, yolo_bboxes, kmeans_bboxes)

        out_name = f"{species}_{img_path.parent.name}_{img_path.parent.parent.name if len(img_path.relative_to(PREPARED_DATASET_DIR).parts) > 1 else 'unknown'}"
        out_path = OUTPUT_ROOT / f"{i:02d}_{out_name}.jpg"
        cv2.imwrite(str(out_path), comparison)

        sample_info = {
            "path": str(img_path.relative_to(PREPARED_DATASET_DIR)),
            "species": species,
            "yolo26_count": len(yolo_bboxes),
            "kmeans_count": len(kmeans_bboxes),
            "yolo26_bboxes": yolo_bboxes,
            "kmeans_bboxes": kmeans_bboxes,
        }
        summary["samples"].append(sample_info)
        summary["yolo26_detections"] += len(yolo_bboxes)
        summary["kmeans_detections"] += len(kmeans_bboxes)
        print(f"  [{i+1}/{len(samples)}] {out_name}: yolo26={len(yolo_bboxes)} kmeans={len(kmeans_bboxes)}")

    summary["total_samples"] = len(samples)
    summary["yolo26_avg"] = round(summary["yolo26_detections"] / len(samples), 2)
    summary["kmeans_avg"] = round(summary["kmeans_detections"] / len(samples), 2)

    summary_path = OUTPUT_ROOT / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_path}")
    print(f"YOLOv26 avg detections/sample: {summary['yolo26_avg']}")
    print(f"KMeans avg detections/sample: {summary['kmeans_avg']}")


if __name__ == "__main__":
    main()
