"""
Reformat dataset into a YOLO-compatible hierarchical structure.

Output per image:
  {species}/{strain}/{environment}/
    {stem}_original.jpg          — preprocessed Petri dish (256×256)
    {stem}_bbox.jpg              — same image with coloured bounding boxes drawn
    {stem}_kmeans_montage.jpg    — full pipeline strip (for inspection)
    {stem}.json                  — YOLO-format annotation (normalised x_center, y_center, w, h)

Pipeline steps captured in the montage:
  0. Original  1. Preprocess (256×256 HoughCircles crop)  2. HSV + Gaussian blur
  3. KMeans colour (k=3)  4. Foreground detection (corner-walk vote)
  5. Foreground mask  6. KMeans position (k=3)  7. Bounding boxes

YOLO label IDs correspond to KMeans cluster indices (0, 1, 2).

Usage — single image test:
    uv run python src/scripts/reformat_yolo_dataset.py --test-image path/to/image.jpg

Usage — full dataset:
    uv run python src/scripts/reformat_yolo_dataset.py
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import DATASET_ROOT, ORIGINAL_DATASET_PATH, STRAIN_SPECIES_MAPPING_PATH
from src.preprocessing.kmeans import get_bbox
from src.preprocessing.preprocess import process_image

YOLO_DATASET_PATH = DATASET_ROOT / "yolo"
FILE_EXTENSION = ".jpg"

# ---------------------------------------------------------------------------
# KMeans pipeline constants
# ---------------------------------------------------------------------------

_CLUSTER_COLORS: List[Tuple[int, int, int]] = [
    (0, 80, 255),
    (0, 220, 80),
    (255, 80, 80),
]  # BGR per cluster

# Montage layout (mirrors reformat_yolo_contour.py style)
_THUMB = 180
_CAPTION_H = 28
_ARROW_W = 48
_HEADER_H = 64
_FOOTER_H = 72
_PANEL_PAD = 8
_BG = (30, 30, 30)
_FG = (220, 220, 220)
_ACCENT = (80, 180, 255)
_ARROW_CLR = (100, 200, 100)


# ---------------------------------------------------------------------------
# Montage helpers
# ---------------------------------------------------------------------------


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _thumb(img: np.ndarray, size: int = _THUMB) -> np.ndarray:
    bgr = _to_bgr(img)
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
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
    tip = (x + _ARROW_W - 10, y_mid)
    tail = (x + 8, y_mid)
    cv2.arrowedLine(canvas, tail, tip, _ARROW_CLR, 2, tipLength=0.4)


def _build_montage(
    steps: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> np.ndarray:
    """Build KMeans pipeline montage strip and return as BGR ndarray."""
    n = len(steps)
    total_w = n * (_THUMB + 2 * _PANEL_PAD) + (n - 1) * _ARROW_W
    total_h = _HEADER_H + _CAPTION_H + _THUMB + _FOOTER_H

    canvas = np.full((total_h, total_w, 3), _BG, dtype=np.uint8)

    fname = Path(meta["image_path"]).name
    _text(canvas, [f"KMeans Pipeline  |  {fname}"], 10, 20, _ACCENT, 0.50, 1)
    params = (
        "HoughCircles 256x256  ->  HSV GaussBlur k=9  ->  KMeans(k=3) colour"
        "  ->  corner-walk FG  ->  KMeans(k=3) position  ->  BBoxes"
    )
    _text(canvas, [params], 10, 46, _FG, 0.33)
    cv2.line(canvas, (0, _HEADER_H - 2), (total_w, _HEADER_H - 2), (70, 70, 70), 1)

    x_cursor = 0
    for i, step in enumerate(steps):
        panel_x = x_cursor + _PANEL_PAD
        img_y = _HEADER_H + _CAPTION_H
        cap_y0 = _HEADER_H + 4
        cap_lines = [f"[{i}] {step['label']}"] + step.get("caption", [])
        _text(canvas, cap_lines[:1], panel_x, cap_y0 + 16, _ACCENT, 0.38)
        if len(cap_lines) > 1:
            _text(canvas, cap_lines[1:], panel_x, cap_y0 + 28, _FG, 0.30)
        thumb = _thumb(step["img"])
        canvas[img_y : img_y + _THUMB, panel_x : panel_x + _THUMB] = thumb
        cv2.rectangle(
            canvas,
            (panel_x, img_y),
            (panel_x + _THUMB - 1, img_y + _THUMB - 1),
            (80, 80, 80),
            1,
        )
        x_cursor += _THUMB + 2 * _PANEL_PAD
        if i < n - 1:
            _draw_arrow(canvas, x_cursor, img_y + _THUMB // 2)
            x_cursor += _ARROW_W

    sep_y = _HEADER_H + _CAPTION_H + _THUMB + 4
    cv2.line(canvas, (0, sep_y), (total_w, sep_y), (70, 70, 70), 1)
    orig_h, orig_w = meta["original_size"]
    row1 = (
        f"Source: {meta['image_path']}   Original: {orig_w}x{orig_h}"
        f"   Bboxes: {meta.get('bboxes', '?')}"
    )
    _text(canvas, [row1], 10, sep_y + 18, _FG, 0.33)
    return canvas


# ---------------------------------------------------------------------------
# KMeans pipeline (step-by-step, with visualisation captures)
# ---------------------------------------------------------------------------


def run_kmeans_pipeline(
    image_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Run the full KMeans segmentation pipeline on one image.

    Returns:
        montage      — BGR pipeline strip
        preprocessed — 256×256 BGR image (Petri dish crop, no boxes)
        bbox_img     — 256×256 BGR image with bounding boxes drawn
        bboxes       — list of {xmin, ymin, xmax, ymax} dicts (pixel coords)
    """
    steps: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"image_path": image_path}

    # ── Step 0  original ────────────────────────────────────────────────────
    src = cv2.imread(image_path)
    if src is None:
        raise ValueError(f"Cannot read image: {image_path}")
    orig_h, orig_w = src.shape[:2]
    meta["original_size"] = (orig_h, orig_w)
    steps.append({"label": "Original", "caption": [f"{orig_w}x{orig_h}"], "img": src})

    # ── Step 1  preprocess (HoughCircles crop → 256×256 BGR) ────────────────
    preprocessed = process_image(src)
    steps.append({"label": "Preprocess", "caption": ["256x256 HoughCrop"], "img": preprocessed})

    # ── Step 2  HSV + Gaussian blur ──────────────────────────────────────────
    img_hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(img_hsv, (9, 9), 0.95)
    steps.append({
        "label": "HSV+Blur",
        "caption": ["GaussBlur k=9 s=0.95"],
        "img": cv2.cvtColor(blur, cv2.COLOR_HSV2BGR),
    })

    # ── Step 3  KMeans on HSV color space ───────────────────────────────────
    mat = np.array(blur)
    h, w, _ = mat.shape
    D = np.full((h * w, 3), 1000.0)

    crop_mask = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(crop_mask, (128, 128), 115, [255, 255, 255], -1)
    circle_valid = crop_mask[:, :, 0] == 255  # (h, w) bool

    flat_valid = circle_valid.ravel()
    D[flat_valid] = mat.reshape(-1, 3)[flat_valid]

    km_color = KMeans(n_clusters=3, random_state=0).fit(D)
    labels = km_color.labels_
    labels_2d = labels.reshape(h, w)

    color_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in enumerate(_CLUSTER_COLORS):
        color_vis[(labels_2d == cid) & circle_valid] = col
    steps.append({"label": "KMeans Color", "caption": ["k=3 HSV clusters"], "img": color_vis})

    # ── Step 4  foreground detection via corner-walking ──────────────────────
    # argmax of internal colour encoding: label 0→2, label 1→0, label 2→1
    _argmax_map = {0: 2, 1: 0, 2: 1}

    corners = np.array(
        [[0, 0], [0, w], [h, 0], [h, w],
         [h // 2, 0], [0, w // 2], [h // 2, w], [h, w // 2]],
        dtype=float,
    )
    directions = np.array(
        [[1, 1], [1, -1], [-1, 1], [-1, -1], [0, 1], [1, 0], [0, -1], [-1, 0]],
        dtype=float,
    )
    votes = np.zeros(8)
    for _ in range(120):
        corners += directions
        for j, corner in enumerate(corners):
            xi, yi = int(corner[0]), int(corner[1])
            if 0 <= xi < h and 0 <= yi < w and votes[j] == 0:
                color_val = _argmax_map[int(labels_2d[xi, yi])]
                if color_val != 0:
                    votes[j] = color_val

    avg_vote = int(np.round(np.average(votes)))
    seg_id = {0: 1, 1: 0, 2: 2}.get(avg_vote, 0)

    fg_mask_2d = (labels_2d == seg_id) & circle_valid
    fg_vis = preprocessed.copy()
    fg_vis[fg_mask_2d] = (0, 255, 255)  # cyan = detected foreground cluster
    steps.append({
        "label": "FG Detect",
        "caption": [f"seg={seg_id} vote~{avg_vote}"],
        "img": fg_vis,
    })

    # ── Step 5  foreground mask ──────────────────────────────────────────────
    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    mask_vis[fg_mask_2d] = (255, 255, 255)
    steps.append({
        "label": "FG Mask",
        "caption": [f"{int(fg_mask_2d.sum())} px"],
        "img": mask_vis,
    })

    # ── Step 6  KMeans on foreground pixel positions ─────────────────────────
    P = np.argwhere(fg_mask_2d)  # (N, 2) as [row, col]
    if len(P) == 0:
        for lbl in ("KMeans Pos", "Bboxes"):
            steps.append({"label": lbl, "caption": ["no FG pixels"], "img": preprocessed.copy()})
        meta["bboxes"] = 0
        return _build_montage(steps, meta), preprocessed, preprocessed, []

    km_pos = KMeans(n_clusters=3, random_state=0).fit(P)
    labels_P = km_pos.labels_

    pos_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, col in enumerate(_CLUSTER_COLORS):
        idx = P[labels_P == cid]
        if len(idx):
            pos_vis[idx[:, 0], idx[:, 1]] = col
    steps.append({"label": "KMeans Pos", "caption": ["k=3 position"], "img": pos_vis})

    # ── Step 7  bounding boxes ───────────────────────────────────────────────
    bboxes = get_bbox(labels_P, P.astype(np.int32))
    result = preprocessed.copy()
    for i, bbox in enumerate(bboxes):
        col = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        cv2.rectangle(result, (bbox["xmin"], bbox["ymin"]), (bbox["xmax"], bbox["ymax"]), col, 2)
    steps.append({"label": "Bboxes", "caption": [f"{len(bboxes)} bboxes"], "img": result})

    meta["bboxes"] = len(bboxes)
    return _build_montage(steps, meta), preprocessed, result, bboxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def bboxes_to_yolo(
    bboxes: List[Dict[str, int]], img_w: int, img_h: int
) -> List[Dict]:
    """Convert pixel bboxes to YOLO normalised format."""
    annotations = []
    for label_id, bbox in enumerate(bboxes):
        x_min, y_min = bbox["xmin"], bbox["ymin"]
        x_max, y_max = bbox["xmax"], bbox["ymax"]
        bw = x_max - x_min
        bh = y_max - y_min
        x_center = (x_min + bw / 2) / img_w
        y_center = (y_min + bh / 2) / img_h
        norm_w = bw / img_w
        norm_h = bh / img_h
        annotations.append(
            {
                "label_id": label_id,
                "x_center": round(x_center, 6),
                "y_center": round(y_center, 6),
                "width": round(norm_w, 6),
                "height": round(norm_h, 6),
            }
        )
    return annotations


# ---------------------------------------------------------------------------
# Core processing
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
    """
    Preprocess one image, run KMeans segmentation, then save:
      - *_original.jpg
      - *_bbox.jpg
      - *_kmeans_montage.jpg  (pipeline strip)
      - *.json  (YOLO annotations)

    Returns True if at least one bounding box was produced.
    """
    target_dir = output_base / specy / strain / environment
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        montage, preprocessed, bbox_img, bboxes = run_kmeans_pipeline(original_img_path)
    except Exception as exc:
        print(f"  [ERROR] KMeans pipeline failed for {filename}: {exc}")
        return False

    if not bboxes:
        print(f"  [WARN] No segments found for {filename}")
        return False

    clean_strain = strain.replace(" ", "_").replace("/", "-")
    stem = f"{clean_strain}_{environment}_{angle}"

    # --- montage ---
    montage_out = target_dir / f"{stem}_kmeans_montage{FILE_EXTENSION}"
    cv2.imwrite(str(montage_out), montage)

    # --- original image ---
    original_out = target_dir / f"{stem}_original{FILE_EXTENSION}"
    cv2.imwrite(str(original_out), preprocessed)

    # --- bbox image ---
    bbox_out = target_dir / f"{stem}_bbox{FILE_EXTENSION}"
    cv2.imwrite(str(bbox_out), bbox_img)

    # --- YOLO JSON ---
    h, w = preprocessed.shape[:2]
    yolo_annotations = bboxes_to_yolo(bboxes, w, h)
    annotation_record = {
        "image": str(original_out.relative_to(output_base)),
        "width": w,
        "height": h,
        "metadata": {
            "strain": strain,
            "environment": environment,
            "angle": angle,
            "specy": specy,
        },
        "annotations": yolo_annotations,
    }
    json_out = target_dir / f"{stem}.json"
    with open(json_out, "w") as f:
        json.dump(annotation_record, f, indent=2)

    print(f"  [OK] {stem}: {len(bboxes)} bbox(es) → {target_dir}")
    return True


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_single(image_path: str, output_base: Path) -> None:
    """Process one image for smoke-testing."""
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
    if ok:
        print("Single-image test passed.")
    else:
        print("Single-image test produced no output.")


def run_full(output_base: Path) -> None:
    """Process the whole original dataset."""
    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: {ORIGINAL_DATASET_PATH} does not exist.")
        return

    if STRAIN_SPECIES_MAPPING_PATH.exists():
        strain_to_specy = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
        print(f"Loaded {len(strain_to_specy)} strain-to-species mappings.")
    else:
        print(f"Warning: {STRAIN_SPECIES_MAPPING_PATH} not found — species will be unknown.")
        strain_to_specy = pd.DataFrame(columns=["Strain", "Species"])

    if output_base.exists():
        response = input(f"{output_base} already exists. Remove and recreate? (y/n): ")
        if response.lower() == "y":
            shutil.rmtree(output_base)
        else:
            print("Aborted.")
            return

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
        description="Build a YOLO-format dataset from fungal colony images."
    )
    parser.add_argument(
        "--test-image",
        metavar="PATH",
        help="Process a single image and exit (for smoke-testing).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=str(YOLO_DATASET_PATH),
        help=f"Output directory (default: {YOLO_DATASET_PATH})",
    )
    args = parser.parse_args()

    output_base = Path(args.output)

    if args.test_image:
        run_single(args.test_image, output_base)
    else:
        run_full(output_base)


if __name__ == "__main__":
    main()
