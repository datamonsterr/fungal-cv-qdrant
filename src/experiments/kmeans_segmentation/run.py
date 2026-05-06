"""
Contour-based colony segmentation pipeline with per-step debug output.

Usage:
    uv run python src/scripts/segment_contours_debug.py <image_path> [--out <output_dir>]

Steps:
    1. Resize to 256×256
    2. Circular crop mask (R=112, centred)
    3. Gaussian blur to smooth before edge detection
    4. Canny edge detection
    5. Morphological close (dilate + erode) to seal gaps
    6. Circularity filter: score contours by area × circularity, pick top-3
    7. Colored colony fill overlay (one colour per selected colony)
    8. Bounding boxes drawn from cv2.boundingRect per selected contour
"""

import argparse
import math
import sys
from dataclasses import dataclass as _dc_autolab
from typing import List as _List_autolab
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
IMG_SIZE = 256
CIRCLE_RADIUS = 112  # plate interior radius (pixels at 256×256)
BLUR_KERNEL = (9, 9)  # gaussian blur kernel
BLUR_SIGMA = 1.5
CANNY_LOW = 30
CANNY_HIGH = 80
DILATE_KERNEL_SIZE = 5  # morphological kernel for dilation
ERODE_KERNEL_SIZE = 3  # morphological kernel for erosion
DILATE_ITER = 3
ERODE_ITER = 2
COLONY_COUNT = 3  # desired number of colonies to detect
MIN_CONTOUR_AREA = 400  # px² – filter tiny noise contours
MAX_CONTOUR_AREA = int(0.60 * 3.14159 * CIRCLE_RADIUS**2)  # ≈ 23 700 px²
MIN_CIRCULARITY = 0.25  # 4π·A/P² — perfect circle = 1.0; elongated streaks → ~0
MIN_CIRCULARITY_RELAXED = 0.10  # fallback threshold when strict yields < COLONY_COUNT

# Montage layout
THUMB = 180  # thumbnail size for each step panel
CAPTION_H = 28  # height of caption bar above each panel
ARROW_W = 48  # width of the arrow column between panels
HEADER_H = 64  # top header (title + params)
FOOTER_H = 72  # bottom metadata bar
PANEL_PAD = 8  # horizontal gap outside each panel
BG = (30, 30, 30)  # dark background (BGR)
FG = (220, 220, 220)  # light text
ACCENT = (80, 180, 255)  # yellow-ish accent for captions (BGR)
ARROW_CLR = (100, 200, 100)

COLONY_COLOURS_BGR = [(0, 80, 255), (0, 220, 80), (255, 80, 80)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img)
    print(f"  saved → {path}")


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR for montage assembly."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _thumb(img: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Resize to square thumbnail, maintaining aspect ratio with black padding."""
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
    """Draw multiple lines of text; returns y after last line."""
    lh = int(scale * 38)
    for line in lines:
        cv2.putText(
            img,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += lh
    return y


def _draw_arrow(canvas: np.ndarray, x: int, y_mid: int) -> None:
    """Draw a right-pointing arrow centred at (x, y_mid)."""
    tip = (x + ARROW_W - 10, y_mid)
    tail = (x + 8, y_mid)
    cv2.arrowedLine(canvas, tail, tip, ARROW_CLR, 2, tipLength=0.4)


def _contour_circularity(cnt: np.ndarray) -> float:
    """Return 4π·A/P² circularity (1.0 = perfect circle, 0 = degenerate)."""
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter**2)


def select_colony_contours(
    contours: List[np.ndarray],
    n: int = COLONY_COUNT,
) -> List[np.ndarray]:
    """
    From a list of contours pick up to *n* that best represent circular colonies.

    Scoring: area × circularity (rewards large AND round contours).
    Two-pass: strict circularity first, then relaxed fallback.
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
# Montage builder
# ---------------------------------------------------------------------------
def build_montage(
    steps: List[Dict[str, Any]],
    meta: Dict[str, Any],
    out_path: Path,
) -> None:
    """
    Build a horizontal strip of all pipeline steps.

    steps: list of {"label": str, "caption": list[str], "img": np.ndarray}
    meta:  dict with processing metadata for the footer
    """
    n = len(steps)
    # Total width: panels + arrows between them + outer padding
    total_w = n * (THUMB + 2 * PANEL_PAD) + (n - 1) * ARROW_W
    total_h = HEADER_H + CAPTION_H + THUMB + FOOTER_H

    canvas = np.full((total_h, total_w, 3), BG, dtype=np.uint8)

    # ---- Header -----------------------------------------------------------
    fname = Path(meta["image_path"]).name
    _text(canvas, [f"Segmentation Pipeline  |  {fname}"], 10, 20, ACCENT, 0.50, 1)
    params = (
        f"R={CIRCLE_RADIUS}  blur={BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} s={BLUR_SIGMA}"
        f"  canny={CANNY_LOW}/{CANNY_HIGH}"
        f"  dil={DILATE_KERNEL_SIZE}x{DILATE_ITER}  ero={ERODE_KERNEL_SIZE}x{ERODE_ITER}"
        f"  circ>={MIN_CIRCULARITY}  n={COLONY_COUNT}"
    )
    _text(canvas, [params], 10, 46, FG, 0.33)

    # Separator line under header
    cv2.line(canvas, (0, HEADER_H - 2), (total_w, HEADER_H - 2), (70, 70, 70), 1)

    # ---- Step panels ------------------------------------------------------
    x_cursor = 0
    for i, step in enumerate(steps):
        panel_x = x_cursor + PANEL_PAD
        img_y = HEADER_H + CAPTION_H

        # Caption bar
        cap_y0 = HEADER_H + 4
        cap_lines = [f"[{i}] {step['label']}"] + step.get("caption", [])
        _text(canvas, cap_lines[:1], panel_x, cap_y0 + 16, ACCENT, 0.38)
        if len(cap_lines) > 1:
            _text(canvas, cap_lines[1:], panel_x, cap_y0 + 28, FG, 0.30)

        # Thumbnail
        thumb = _thumb(step["img"])
        canvas[img_y : img_y + THUMB, panel_x : panel_x + THUMB] = thumb

        # Thin border around thumbnail
        cv2.rectangle(
            canvas,
            (panel_x, img_y),
            (panel_x + THUMB - 1, img_y + THUMB - 1),
            (80, 80, 80),
            1,
        )

        x_cursor += THUMB + 2 * PANEL_PAD

        # Arrow (between panels, not after last)
        if i < n - 1:
            arrow_x = x_cursor
            arrow_y_mid = img_y + THUMB // 2
            _draw_arrow(canvas, arrow_x, arrow_y_mid)
            x_cursor += ARROW_W

    # ---- Footer -----------------------------------------------------------
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

    cv2.imwrite(str(out_path), canvas)
    print(f"  montage → {out_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(image_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"image_path": image_path}

    # ------------------------------------------------------------------
    # Step 0 – load
    # ------------------------------------------------------------------
    src = cv2.imread(image_path)
    if src is None:
        sys.exit(f"ERROR: cannot read '{image_path}'")
    orig_h, orig_w = src.shape[:2]
    meta["original_size"] = (orig_h, orig_w)
    print(f"[0] Loaded  {orig_w}×{orig_h}  {image_path}")
    _save(out / "00_original.jpg", src)
    steps.append(
        {
            "label": "Original",
            "caption": [f"{orig_w}x{orig_h}"],
            "img": src,
        }
    )

    # ------------------------------------------------------------------
    # Step 1 – resize to 256×256
    # ------------------------------------------------------------------
    img = cv2.resize(src, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    print(f"[1] Resized → {IMG_SIZE}×{IMG_SIZE}")
    _save(out / "01_resized.jpg", img)
    steps.append(
        {
            "label": "Resize",
            "caption": [f"{IMG_SIZE}x{IMG_SIZE}"],
            "img": img,
        }
    )

    # ------------------------------------------------------------------
    # Step 2 – circular crop
    # ------------------------------------------------------------------
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    circle_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(circle_mask, (cx, cy), CIRCLE_RADIUS, 255, -1)
    img_cropped = img.copy()
    img_cropped[circle_mask == 0] = 0
    print(f"[2] Circular crop  R={CIRCLE_RADIUS}  centre=({cx},{cy})")
    _save(out / "02_circle_crop.jpg", img_cropped)
    _save(out / "02_circle_mask.png", circle_mask)
    steps.append(
        {
            "label": "Circle Crop",
            "caption": [f"R={CIRCLE_RADIUS}"],
            "img": img_cropped,
        }
    )

    # ------------------------------------------------------------------
    # Step 3 – Gaussian blur
    # ------------------------------------------------------------------
    blurred = cv2.GaussianBlur(img_cropped, BLUR_KERNEL, BLUR_SIGMA)
    print(f"[3] Gaussian blur  kernel={BLUR_KERNEL}  sigma={BLUR_SIGMA}")
    _save(out / "03_blurred.jpg", blurred)
    steps.append(
        {
            "label": "Gauss Blur",
            "caption": [f"{BLUR_KERNEL[0]}x{BLUR_KERNEL[1]} s={BLUR_SIGMA}"],
            "img": blurred,
        }
    )

    # ------------------------------------------------------------------
    # Step 4 – Canny edge detection
    # ------------------------------------------------------------------
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    rim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_mask = cv2.erode(circle_mask, rim_kernel, iterations=1)
    edges[inner_mask == 0] = 0
    print(f"[4] Canny edges  low={CANNY_LOW}  high={CANNY_HIGH}  (rim stripped ~5 px)")
    _save(out / "04_edges.png", edges)
    steps.append(
        {
            "label": "Canny Edges",
            "caption": [f"lo={CANNY_LOW} hi={CANNY_HIGH}"],
            "img": edges,
        }
    )

    # ------------------------------------------------------------------
    # Step 5 – morphological close
    # ------------------------------------------------------------------
    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE)
    )
    dilated = cv2.dilate(edges, dil_kernel, iterations=DILATE_ITER)
    closed = cv2.erode(dilated, erode_kernel, iterations=ERODE_ITER)
    closed[inner_mask == 0] = 0
    print(
        f"[5] Morphology  dilate={DILATE_KERNEL_SIZE}×{DILATE_ITER}  "
        f"erode={ERODE_KERNEL_SIZE}×{ERODE_ITER}"
    )
    _save(out / "05_morphology_closed.png", closed)
    steps.append(
        {
            "label": "Morphology",
            "caption": [
                f"dil={DILATE_KERNEL_SIZE}x{DILATE_ITER} ero={ERODE_KERNEL_SIZE}x{ERODE_ITER}"
            ],
            "img": closed,
        }
    )

    # ------------------------------------------------------------------
    # Step 6 – circularity filter: pick top-COLONY_COUNT colonies
    # ------------------------------------------------------------------
    all_contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    meta["contours_total"] = len(all_contours)

    selected = select_colony_contours(all_contours)
    meta["colonies_found"] = len(selected)

    # Print per-contour diagnostics
    for cnt in all_contours:
        area = cv2.contourArea(cnt)
        circ = _contour_circularity(cnt)
        kept = any(np.array_equal(cnt, s) for s in selected)
        status = "KEPT" if kept else "dropped"
        print(
            f"  contour  area={int(area)}  circ={circ:.3f}  score={int(area * circ)}  {status}"
        )

    print(
        f"[6] Circularity filter  {len(selected)}/{len(all_contours)} contours selected"
        f"  (circ>={MIN_CIRCULARITY}, area {MIN_CONTOUR_AREA}–{MAX_CONTOUR_AREA} px²)"
    )

    # Filled mask of selected contours (white)
    filled = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.drawContours(filled, selected, -1, 255, thickness=cv2.FILLED)
    filled[inner_mask == 0] = 0
    _save(out / "06_circ_filter.png", filled)

    overlay = img.copy()
    overlay[filled == 255] = (255, 255, 255)
    _save(out / "06_circ_overlay.jpg", overlay)
    steps.append(
        {
            "label": "Circ. Filter",
            "caption": [
                f"{len(selected)}/{len(all_contours)} kept  circ>={MIN_CIRCULARITY}"
            ],
            "img": overlay,
        }
    )

    # ------------------------------------------------------------------
    # Step 7 – colored colony overlay (one colour per colony)
    # ------------------------------------------------------------------
    colony_vis = img.copy()
    for idx, cnt in enumerate(selected):
        colour = COLONY_COLOURS_BGR[idx % len(COLONY_COLOURS_BGR)]
        cv2.drawContours(colony_vis, [cnt], -1, colour, thickness=cv2.FILLED)
    _save(out / "07_colony_colours.jpg", colony_vis)
    steps.append(
        {
            "label": "Colonies",
            "caption": [f"{len(selected)} colonies"],
            "img": colony_vis,
        }
    )

    # ------------------------------------------------------------------
    # Step 8 – bounding boxes from contour boundingRect
    # ------------------------------------------------------------------
    result = img.copy()
    bboxes = []

    for idx, cnt in enumerate(selected):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_CONTOUR_AREA:
            print(
                f"  colony {idx}: bbox_area={area}  FILTERED (too small < {MIN_CONTOUR_AREA})"
            )
            continue
        bboxes.append((x, y, x + w, y + h, idx))
        colour = COLONY_COLOURS_BGR[idx % len(COLONY_COLOURS_BGR)]
        cv2.rectangle(result, (x, y), (x + w, y + h), colour, 2)
        circ = _contour_circularity(cnt)
        cv2.putText(
            result,
            f"C{idx} circ={circ:.2f}",
            (x + 2, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            colour,
            1,
        )
        print(
            f"  colony {idx}: bbox=({x},{y})→({x+w},{y+h})  area={area}  circ={circ:.3f}  OK"
        )

    meta["bboxes_kept"] = len(bboxes)
    print(f"[8] {len(bboxes)} bounding boxes kept")
    _save(out / "08_bboxes.jpg", result)
    steps.append(
        {
            "label": "Bboxes",
            "caption": [f"{len(bboxes)} kept"],
            "img": result,
        }
    )

    # ------------------------------------------------------------------
    # Montage
    # ------------------------------------------------------------------
    build_montage(steps, meta, out / "pipeline_montage.jpg")
    print(f"\nDone. All debug images written to  {out}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Contour-based colony segmentation with per-step debug images"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--out",
        default="debug_segment",
        help="Output directory (default: debug_segment/)",
    )
    args = parser.parse_args()
    run(args.image, args.out)


if __name__ == "__main__":
    main()


@_dc_autolab
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@_dc_autolab
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: _List_autolab[str]
    run_id: str


def run_experiment(params: ExperimentParams) -> ExperimentResult:
    """Uniform experiment contract wrapper for kmeans_segmentation."""
    import json as _json_autolab
    from pathlib import Path as _Path_autolab
    output_root = _Path_autolab(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = params.description[:30] if params.description else "kmeans_segmentation"
    result_data = {"f1_score": 0.0, "strategy_name": strategy, "artifact_paths": [], "run_id": params.run_id}
    (output_root / "results.json").write_text(_json_autolab.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
