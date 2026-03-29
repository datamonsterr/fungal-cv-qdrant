"""
Add contour-based segmentation outputs to the existing YOLO dataset.

For each image the following files are written next to the KMeans outputs
inside Dataset/yolo/{species}/{strain}/{environment}/:

  {stem}_contour_montage.jpg   — full pipeline strip (for inspection)
  {stem}_contour_bbox.jpg      — final bbox result (compare with _bbox.jpg)
  {stem}_contour.json          — YOLO-format annotations (normalised coords)

Algorithm mirrors segment_contours_debug.py exactly:
  0. Load  1. Resize 256×256  2. Circle crop  3. Gauss blur
  4. Canny  5. Morph close
  6. Circularity filter: score contours by area × circularity, pick top-3
  7. Colored colony fill overlay (one colour per selected colony)
  8. Bboxes from cv2.boundingRect per selected contour

Usage — single image (smoke test):
    uv run python src/scripts/reformat_yolo_contour.py --test-image path/to/img.jpg

Usage — full dataset (appends to Dataset/yolo/):
    uv run python src/scripts/reformat_yolo_contour.py
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.config import DATASET_ROOT, ORIGINAL_DATASET_PATH, STRAIN_SPECIES_MAPPING_PATH

# ---------------------------------------------------------------------------
# Algorithm parameters (same as segment_contours_debug.py)
# ---------------------------------------------------------------------------
IMG_SIZE = 256
CIRCLE_RADIUS = 112
BLUR_KERNEL = (9, 9)
BLUR_SIGMA = 1.5
CANNY_LOW = 30
CANNY_HIGH = 80
DILATE_KERNEL_SIZE = 5
ERODE_KERNEL_SIZE = 3
DILATE_ITER = 3
ERODE_ITER = 2
COLONY_COUNT = 3
MIN_CONTOUR_AREA = 400
MAX_CONTOUR_AREA = int(0.60 * 3.14159 * CIRCLE_RADIUS**2)
MIN_CIRCULARITY = 0.25
MIN_CIRCULARITY_RELAXED = 0.10

# ---------------------------------------------------------------------------
# Montage layout constants (same as segment_contours_debug.py)
# ---------------------------------------------------------------------------
THUMB = 180
CAPTION_H = 28
ARROW_W = 48
HEADER_H = 64
FOOTER_H = 72
PANEL_PAD = 8
BG = (30, 30, 30)
FG = (220, 220, 220)
ACCENT = (80, 180, 255)
ARROW_CLR = (100, 200, 100)

COLONY_COLOURS_BGR = [(0, 80, 255), (0, 220, 80), (255, 80, 80)]

YOLO_DATASET_PATH = DATASET_ROOT / "yolo"
FILE_EXTENSION = ".jpg"


# ---------------------------------------------------------------------------
# Circularity helper
# ---------------------------------------------------------------------------

def _contour_circularity(cnt: np.ndarray) -> float:
    """Return 4π·A/P² circularity (1.0 = perfect circle)."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)


def select_colony_contours(
    contours: List[np.ndarray],
    n: int = COLONY_COUNT,
) -> List[np.ndarray]:
    """
    Pick up to *n* contours that best represent circular colonies.

    Scoring: area × circularity. Two-pass: strict threshold first,
    then relaxed fallback if fewer than n candidates found.
    """
    def _candidates(min_circ: float) -> List[Tuple[float, np.ndarray]]:
        scored = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            circ = _contour_circularity(cnt)
            if circ < min_circ:
                continue
            scored.append((area * circ, cnt))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    candidates = _candidates(MIN_CIRCULARITY)
    if len(candidates) < n:
        candidates = _candidates(MIN_CIRCULARITY_RELAXED)

    return [cnt for _, cnt in candidates[:n]]


# ---------------------------------------------------------------------------
# Montage helpers
# ---------------------------------------------------------------------------

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _thumb(img: np.ndarray, size: int = THUMB) -> np.ndarray:
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _text(
    img: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    color: Tuple,
    scale: float = 0.38,
    thickness: int = 1,
) -> int:
    lh = int(scale * 38)
    for line in lines:
        cv2.putText(
            img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
        )
        y += lh
    return y


def _draw_arrow(canvas: np.ndarray, x: int, y_mid: int) -> None:
    tip = (x + ARROW_W - 10, y_mid)
    tail = (x + 8, y_mid)
    cv2.arrowedLine(canvas, tail, tip, ARROW_CLR, 2, tipLength=0.4)


def build_montage(
    steps: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> np.ndarray:
    """Build pipeline montage strip and return as BGR ndarray."""
    n = len(steps)
    total_w = n * (THUMB + 2 * PANEL_PAD) + (n - 1) * ARROW_W
    total_h = HEADER_H + CAPTION_H + THUMB + FOOTER_H

    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    fname = Path(meta["image_path"]).name
    _text(canvas, [f"Contour Pipeline  |  {fname}"], 10, 20, ACCENT, 0.50, 1)
    params = (
        f"R={CIRCLE_RADIUS}  blur={BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} s={BLUR_SIGMA}"
        f"  canny={CANNY_LOW}/{CANNY_HIGH}"
        f"  dil={DILATE_KERNEL_SIZE}x{DILATE_ITER}  ero={ERODE_KERNEL_SIZE}x{ERODE_ITER}"
        f"  circ>={MIN_CIRCULARITY}  n={COLONY_COUNT}"
    )
    _text(canvas, [params], 10, 46, FG, 0.33)
    cv2.line(canvas, (0, HEADER_H - 2), (total_w, HEADER_H - 2), (70, 70, 70), 1)

    x_cursor = 0
    for i, step in enumerate(steps):
        panel_x = x_cursor + PANEL_PAD
        img_y = HEADER_H + CAPTION_H
        cap_y0 = HEADER_H + 4
        cap_lines = [f"[{i}] {step['label']}"] + step.get("caption", [])
        _text(canvas, cap_lines[:1], panel_x, cap_y0 + 16, ACCENT, 0.38)
        if len(cap_lines) > 1:
            _text(canvas, cap_lines[1:], panel_x, cap_y0 + 28, FG, 0.30)
        thumb = _thumb(step["img"])
        canvas[img_y : img_y + THUMB, panel_x : panel_x + THUMB] = thumb
        cv2.rectangle(
            canvas,
            (panel_x, img_y),
            (panel_x + THUMB - 1, img_y + THUMB - 1),
            (80, 80, 80),
            1,
        )
        x_cursor += THUMB + 2 * PANEL_PAD
        if i < n - 1:
            _draw_arrow(canvas, x_cursor, img_y + THUMB // 2)
            x_cursor += ARROW_W

    sep_y = HEADER_H + CAPTION_H + THUMB + 4
    cv2.line(canvas, (0, sep_y), (total_w, sep_y), (70, 70, 70), 1)
    orig_h, orig_w = meta["original_size"]
    row1 = (
        f"Source: {meta['image_path']}   "
        f"Original: {orig_w}x{orig_h}   "
        f"Contours total: {meta['contours_total']}   "
        f"Colonies found: {meta['colonies_found']}/{COLONY_COUNT}"
    )
    row2 = (
        f"Bboxes kept: {meta['bboxes_kept']}   "
        f"max_contour_area={MAX_CONTOUR_AREA} px²   "
        f"min_contour_area={MIN_CONTOUR_AREA} px²   "
        f"min_circularity={MIN_CIRCULARITY}"
    )
    _text(canvas, [row1], 10, sep_y + 18, FG, 0.33)
    _text(canvas, [row2], 10, sep_y + 36, FG, 0.33)
    return canvas


# ---------------------------------------------------------------------------
# Core contour segmentation pipeline (in-memory)
# ---------------------------------------------------------------------------

def run_contour_pipeline(
    image_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Run the full contour segmentation pipeline on one image.

    Returns:
        montage_img  — BGR pipeline strip (HEADER+CAPTION+THUMB+FOOTER tall)
        bbox_img     — 256×256 BGR image with bounding boxes drawn
        bboxes       — list of {xmin, ymin, xmax, ymax} dicts (pixel coords)
    """
    steps: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"image_path": image_path}

    # Step 0 – load
    src = cv2.imread(image_path)
    if src is None:
        raise ValueError(f"Cannot read image: {image_path}")
    orig_h, orig_w = src.shape[:2]
    meta["original_size"] = (orig_h, orig_w)
    steps.append({"label": "Original", "caption": [f"{orig_w}x{orig_h}"], "img": src})

    # Step 1 – resize
    img = cv2.resize(src, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    steps.append({"label": "Resize", "caption": [f"{IMG_SIZE}x{IMG_SIZE}"], "img": img})

    # Step 2 – circular crop
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    circle_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), CIRCLE_RADIUS, 255, -1)
    img_cropped = img.copy()
    img_cropped[circle_mask == 0] = 0
    steps.append({"label": "Circle Crop", "caption": [f"R={CIRCLE_RADIUS}"], "img": img_cropped})

    # Step 3 – Gaussian blur
    blurred = cv2.GaussianBlur(img_cropped, BLUR_KERNEL, BLUR_SIGMA)
    steps.append(
        {
            "label": "Gauss Blur",
            "caption": [f"{BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} s={BLUR_SIGMA}"],
            "img": blurred,
        }
    )

    # Step 4 – Canny edges
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_mask = cv2.erode(circle_mask, rim_kernel, iterations=1)
    edges[inner_mask == 0] = 0
    steps.append(
        {"label": "Canny Edges", "caption": [f"lo={CANNY_LOW} hi={CANNY_HIGH}"], "img": edges}
    )

    # Step 5 – morphological close
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE)
    )
    dilated = cv2.dilate(edges, dil_kernel, iterations=DILATE_ITER)
    closed = cv2.erode(dilated, erode_kernel, iterations=ERODE_ITER)
    closed[inner_mask == 0] = 0
    steps.append(
        {
            "label": "Morphology",
            "caption": [
                f"dil={DILATE_KERNEL_SIZE}x{DILATE_ITER} ero={ERODE_KERNEL_SIZE}x{ERODE_ITER}"
            ],
            "img": closed,
        }
    )

    # Step 6 – circularity filter: pick top-COLONY_COUNT colonies
    all_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meta["contours_total"] = len(all_contours)

    selected = select_colony_contours(all_contours)
    meta["colonies_found"] = len(selected)

    filled = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.drawContours(filled, selected, -1, 255, thickness=cv2.FILLED)
    filled[inner_mask == 0] = 0
    overlay = img.copy()
    overlay[filled == 255] = (255, 255, 255)
    steps.append(
        {
            "label": "Circ. Filter",
            "caption": [f"{len(selected)}/{len(all_contours)} kept  circ>={MIN_CIRCULARITY}"],
            "img": overlay,
        }
    )

    # Step 7 – colored colony overlay
    colony_vis = img.copy()
    for idx, cnt in enumerate(selected):
        colour = COLONY_COLOURS_BGR[idx % len(COLONY_COLOURS_BGR)]
        cv2.drawContours(colony_vis, [cnt], -1, colour, thickness=cv2.FILLED)
    steps.append(
        {
            "label": "Colonies",
            "caption": [f"{len(selected)} colonies"],
            "img": colony_vis,
        }
    )

    # Step 8 – bounding boxes from boundingRect
    result = img.copy()
    bboxes: List[Dict[str, int]] = []

    for idx, cnt in enumerate(selected):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_CONTOUR_AREA:
            continue
        bboxes.append(
            {"xmin": int(x), "ymin": int(y), "xmax": int(x + w), "ymax": int(y + h)}
        )
        colour = COLONY_COLOURS_BGR[idx % len(COLONY_COLOURS_BGR)]
        cv2.rectangle(result, (x, y), (x + w, y + h), colour, 2)
        circ = _contour_circularity(cnt)
        cv2.putText(
            result,
            f"C{idx} {circ:.2f}",
            (x + 2, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
        )

    meta["bboxes_kept"] = len(bboxes)
    steps.append(
        {
            "label": "Bboxes",
            "caption": [f"{len(bboxes)} kept"],
            "img": result,
        }
    )

    montage = build_montage(steps, meta)
    return montage, result, bboxes


# ---------------------------------------------------------------------------
# YOLO helpers (duplicated from reformat_yolo_dataset for self-containment)
# ---------------------------------------------------------------------------

def bboxes_to_yolo(
    bboxes: List[Dict[str, int]], img_w: int, img_h: int
) -> List[Dict]:
    annotations = []
    for label_id, bbox in enumerate(bboxes):
        bw = bbox["xmax"] - bbox["xmin"]
        bh = bbox["ymax"] - bbox["ymin"]
        x_center = (bbox["xmin"] + bw / 2) / img_w
        y_center = (bbox["ymin"] + bh / 2) / img_h
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


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> Optional[str]:
    result = strain_to_specy[strain_to_specy["Strain"] == strain]
    if not result.empty:
        return result["Species"].iloc[0]
    return None


def parse_filename(filename: str) -> Dict[str, str]:
    clean = filename.removesuffix(FILE_EXTENSION).removesuffix("_edited")
    match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", clean)
    if match:
        return {
            "strain": match.group(1),
            "environment": match.group(2),
            "angle": match.group(3),
        }
    return {"strain": "unknown", "environment": "unknown", "angle": "unknown"}


# ---------------------------------------------------------------------------
# Per-image writer
# ---------------------------------------------------------------------------

def process_single_image(
    original_img_path: str,
    filename: str,
    strain: str,
    specy: str,
    environment: str,
    angle: str,
    output_base: Path,
) -> bool:
    target_dir = output_base / specy / strain / environment
    target_dir.mkdir(parents=True, exist_ok=True)

    clean_strain = strain.replace(" ", "_").replace("/", "-")
    stem = f"{clean_strain}_{environment}_{angle}"

    # Save a temp copy so segment pipeline reads from disk
    import shutil
    temp_path = str(target_dir / f"_temp_{filename}")
    shutil.copy(original_img_path, temp_path)

    try:
        montage, bbox_img, bboxes = run_contour_pipeline(temp_path)
    except Exception as exc:
        print(f"  [ERROR] {filename}: {exc}")
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Save montage
    montage_out = target_dir / f"{stem}_contour_montage.jpg"
    cv2.imwrite(str(montage_out), montage)

    # Save final bbox image
    bbox_out = target_dir / f"{stem}_contour_bbox.jpg"
    cv2.imwrite(str(bbox_out), bbox_img)

    # Save YOLO JSON
    h, w = bbox_img.shape[:2]
    annotation_record = {
        "image": str((target_dir / f"{stem}_original.jpg").relative_to(output_base)),
        "width": w,
        "height": h,
        "method": "contour",
        "metadata": {
            "strain": strain,
            "environment": environment,
            "angle": angle,
            "specy": specy,
        },
        "annotations": bboxes_to_yolo(bboxes, w, h),
    }
    json_out = target_dir / f"{stem}_contour.json"
    with open(json_out, "w") as f:
        json.dump(annotation_record, f, indent=2)

    n = len(bboxes)
    print(f"  [OK] {stem}: {n} bbox(es) → {target_dir}")
    return True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_single(image_path: str, output_base: Path) -> None:
    if STRAIN_SPECIES_MAPPING_PATH.exists():
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    else:
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    filename = Path(image_path).name
    meta = parse_filename(filename)
    strain = meta["strain"]
    specy = get_specy_from_strain(strain, strain_to_specy) or "unknown_species"

    print(f"Test run — image : {image_path}")
    print(f"           strain: {strain}, specy: {specy}")

    output_base.mkdir(parents=True, exist_ok=True)
    ok = process_single_image(
        original_img_path=image_path,
        filename=filename,
        strain=strain,
        specy=specy,
        environment=meta["environment"],
        angle=meta["angle"],
        output_base=output_base,
    )
    print("Single-image test passed." if ok else "Single-image test produced no output.")


def run_full(output_base: Path) -> None:
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: {ORIGINAL_DATASET_PATH} does not exist.")
        return

    if STRAIN_SPECIES_MAPPING_PATH.exists():
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
        print(f"Loaded {len(strain_to_specy)} strain-to-species mappings.")
    else:
        print("Warning: mapping not found — species will be unknown.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    output_base.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "ok": 0, "failed": 0, "unknown_species": 0}

    for dir_name in sorted(os.listdir(ORIGINAL_DATASET_PATH)):
        dir_path = ORIGINAL_DATASET_PATH / dir_name
        if not dir_path.is_dir():
            continue
        print(f"\nDirectory: {dir_name}")

        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(FILE_EXTENSION):
                continue
            stats["total"] += 1
            meta = parse_filename(filename)
            strain = meta["strain"]
            specy = get_specy_from_strain(strain, strain_to_specy)
            if specy is None:
                specy = "unknown_species"
                stats["unknown_species"] += 1

            ok = process_single_image(
                original_img_path=str(dir_path / filename),
                filename=filename,
                strain=strain,
                specy=specy,
                environment=meta["environment"],
                angle=meta["angle"],
                output_base=output_base,
            )
            if ok:
                stats["ok"] += 1
            else:
                stats["failed"] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:       {stats['total']}")
    print(f"Processed OK:      {stats['ok']}")
    print(f"Failed:            {stats['failed']}")
    print(f"Unknown species:   {stats['unknown_species']}")
    print(f"Output:            {output_base}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add contour-based segmentation outputs to the YOLO dataset."
    )
    parser.add_argument(
        "--test-image",
        metavar="PATH",
        help="Process a single image and exit (smoke test).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=str(YOLO_DATASET_PATH),
        help=f"Output base directory (default: {YOLO_DATASET_PATH})",
    )
    args = parser.parse_args()

    output_base = Path(args.output)

    if args.test_image:
        run_single(args.test_image, output_base)
    else:
        run_full(output_base)


if __name__ == "__main__":
    main()
