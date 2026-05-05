"""
Per-strategy visualization for threshold experiment.

For each strategy × algorithm combination, creates a folder with:
  - A grid image showing samples grouped by TP / FP / TN / FN
  - Each cell: thumbnail + (is_known?, predicted, actual, score)

Usage:
    uv run python -m src.experiments.threshold.visualize_strategies
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, WORKSPACE_ROOT  # noqa: E402

INPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = RESULTS_DIR / "threshold" / "strategy_visualizations"


# ---------------------------------------------------------------------------
# Load retrieval data
# ---------------------------------------------------------------------------


def load_results(
    csv_path: Path,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Returns:
        rows       : list of dicts (CSV rows)
        labels     : (N,) 1=known, 0=unknown
        scores_arr : (N, 5) s0..s4 scores
    """
    rows = []
    labels = []
    scores_mat = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            labels.append(int(row.get("is_known", 0)))
            row_scores = []
            for i in range(5):
                val = row.get(f"s{i}_score", "")
                row_scores.append(float(val) if val else 0.0)
            scores_mat.append(row_scores)

    labels_arr = np.array(labels, dtype=float)
    scores_arr = np.array(scores_mat, dtype=float)
    return rows, labels_arr, scores_arr


def compute_strategy_scores(scores_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-sample scalar scores for each threshold strategy."""
    s = scores_arr
    eps = 1e-9

    strategies: Dict[str, np.ndarray] = {
        "s0": s[:, 0],
        "gap": (s[:, 0] - s[:, 1]) / (s[:, 0] + s[:, 1] + eps),
        "ratio": s[:, 0] / (s[:, 1] + eps),
        "abs_gap": s[:, 0] - s[:, 1],
        "top3_avg": (s[:, 0] + s[:, 1] + s[:, 2]) / 3.0,
    }

    top5 = s[:, :5]
    row_sum = top5.sum(axis=1, keepdims=True) + eps
    p = top5 / row_sum
    p_safe = np.where(p > 0, p, eps)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
    max_entropy = np.log(5)
    strategies["neg_entropy"] = 1.0 - entropy / max_entropy

    return strategies


# ---------------------------------------------------------------------------
# Strategy thresholds (best F1 from threshold_analysis.csv)
# ---------------------------------------------------------------------------

BEST_THRESHOLDS = {
    # (strategy, algorithm) -> threshold
    ("s0", "f1_grid"): 0.728643,
    ("gap", "f1_grid"): 0.467337,
    ("ratio", "f1_grid"): 0.000000,
    ("abs_gap", "f1_grid"): 0.467337,
    ("top3_avg", "f1_grid"): 0.274707,
    ("neg_entropy", "f1_grid"): 0.641369,
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


THUMB_SIZE = (128, 128)
CELL_PADDING = 8
TEXT_HEIGHT = 48
HEADER_HEIGHT = 40
GRID_COLS = 6


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return cast(ImageFont.ImageFont, ImageFont.truetype(path, size))
    return cast(ImageFont.ImageFont, ImageFont.load_default())


def _get_thumbnail(img_path: str, size: Tuple[int, int]) -> Optional[Image.Image]:
    if not os.path.exists(img_path):
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def _draw_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: Optional[Image.Image],
    x: int,
    y: int,
    w: int,
    h: int,
    text_lines: List[Tuple[str, ImageFont.ImageFont]],
    border_color: Tuple[int, int, int],
    border_width: int = 2,
) -> None:
    """Draw one cell of the grid."""
    # Background
    draw.rectangle([x, y, x + w, y + h], fill=(245, 245, 245))

    # Image
    img_area_x = x + CELL_PADDING
    img_area_y = y + CELL_PADDING
    img_area_w = w - 2 * CELL_PADDING
    img_area_h = h - TEXT_HEIGHT - 2 * CELL_PADDING

    if img:
        img_display = img.resize((img_area_w, img_area_h), Image.Resampling.LANCZOS)
        canvas.paste(img_display, (img_area_x, img_area_y))
    else:
        draw.rectangle(
            [
                img_area_x,
                img_area_y,
                x + w - CELL_PADDING,
                y + h - TEXT_HEIGHT - CELL_PADDING,
            ],
            outline=(180, 180, 180),
            width=1,
        )
        draw.text((img_area_x + 4, img_area_y + 4), "Missing", fill=(140, 140, 140))

    # Border
    draw.rectangle([x, y, x + w, y + h], outline=border_color, width=border_width)

    # Text
    text_x = x + CELL_PADDING
    text_y = y + h - TEXT_HEIGHT + 4
    for line, font in text_lines:
        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
        _, _, _, bottom = font.getbbox(line or "Ag")
        text_y += max(bottom, 12) + 2


def _make_grid(
    samples: List[Dict],
    scores: np.ndarray,
    threshold: float,
    group_label: str,
    color: Tuple[int, int, int],
    colormap: Dict[str, Tuple[int, int, int]],
    thumb_size: Tuple[int, int] = THUMB_SIZE,
    max_cols: int = GRID_COLS,
    max_rows: int = 6,
    font_bold: ImageFont.ImageFont | None = None,
    font_small: ImageFont.ImageFont | None = None,
) -> Tuple[Optional[Image.Image], int]:
    """
    Build a grid of sample thumbnails for one group (TP/FP/TN/FN).
    Returns (grid_image, total_samples).
    """
    if not samples:
        return None, 0
    if font_bold is None:
        font_bold = _load_font(13)
    if font_small is None:
        font_small = _load_font(10)

    n = len(samples)
    cols = min(max_cols, n)
    rows = min(max_rows, (n + cols - 1) // cols)

    cell_w = thumb_size[0] + 2 * CELL_PADDING
    cell_h = thumb_size[1] + TEXT_HEIGHT + 2 * CELL_PADDING

    grid_w = cols * cell_w
    grid_h = rows * cell_h

    canvas = Image.new("RGB", (grid_w, grid_h + HEADER_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Header
    header = f"{group_label}  (n={n})"
    draw.rectangle([0, 0, grid_w, HEADER_HEIGHT], fill=color)
    draw.text((CELL_PADDING, 8), header, fill=(255, 255, 255), font=font_bold)

    for idx, row in enumerate(samples[: cols * rows]):
        col_i = idx % cols
        row_i = idx // cols
        cx = col_i * cell_w
        cy = HEADER_HEIGHT + row_i * cell_h

        img_path = str(WORKSPACE_ROOT / row.get("image_path", ""))
        img = _get_thumbnail(img_path, thumb_size)

        is_known = int(row.get("is_known", 0))
        pred_species = row.get("predicted_species", "")
        score_val = float(scores[idx]) if idx < len(scores) else 0.0

        border = color
        lines = [
            (f"{'KNOWN' if is_known else 'UNK'}", font_small),
            (pred_species[:18], font_small),
            (f"t={score_val:.3f}", font_small),
        ]

        _draw_cell(canvas, draw, img, cx, cy, cell_w, cell_h, lines, border, 2)

    return canvas, n


def visualize_strategy(
    rows: List[Dict[str, Any]],
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    strategy_name: str,
    output_dir: Path,
    algorithm: str = "f1_grid",
) -> None:
    """
    Create a full-page visualization for one strategy showing
    TP / FP / TN / FN groups as grid panels.
    """
    preds = (scores >= threshold).astype(int)

    tp_idx = np.where((preds == 1) & (labels == 1))[0]
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    tn_idx = np.where((preds == 0) & (labels == 0))[0]
    fn_idx = np.where((preds == 0) & (labels == 1))[0]

    groups = [
        ("True Positives  (known → known)", tp_idx, (0, 180, 0)),
        ("False Positives (unk → known)", fp_idx, (220, 60, 60)),
        ("True Negatives  (unk → unk)", tn_idx, (60, 120, 200)),
        ("False Negatives (known → unk)", fn_idx, (220, 140, 0)),
    ]

    # Fonts
    font_title = _load_font(18)
    font_bold = _load_font(13)
    font_small = _load_font(10)

    panels: List[Tuple[Optional[Image.Image], int]] = []
    for label_text, idx_arr, color in groups:
        group_rows = [rows[i] for i in idx_arr]
        group_scores = scores[idx_arr]
        panel, n = _make_grid(
            group_rows,
            group_scores,
            threshold,
            label_text,
            color,
            {},
            font_bold=font_bold,
            font_small=font_small,
        )
        panels.append((panel, n))

    # Assemble into a single image
    valid_panels = [(p, n) for p, n in panels if p is not None]
    if not valid_panels:
        print(f"  No panels to render for {strategy_name}")
        return

    panel_imgs = [p for p, _ in valid_panels]
    widths = [p.width for p in panel_imgs]
    heights = [p.height for p in panel_imgs]

    max_w = max(widths)
    total_h = sum(heights) + 30 * len(panel_imgs) + HEADER_HEIGHT + 20

    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    title = f"Strategy: {strategy_name}  |  Threshold: {threshold:.4f}  |  Algorithm: {algorithm}"
    draw.rectangle([0, 0, max_w, HEADER_HEIGHT + 4], fill=(30, 30, 30))
    draw.text((CELL_PADDING, 8), title, fill=(255, 255, 255), font=font_title)

    y_offset = HEADER_HEIGHT + 20
    for panel, n in valid_panels:
        if panel.width < max_w:
            # Pad to max_w
            padded = Image.new("RGB", (max_w, panel.height), (255, 255, 255))
            padded.paste(panel, (0, 0))
            panel = padded
        canvas.paste(panel, (0, y_offset))
        y_offset += panel.height + 30

    out_path = output_dir / f"{strategy_name}_{algorithm}.png"
    canvas.save(out_path)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run retrieval first.")
        sys.exit(1)

    print(f"Loading: {INPUT_CSV}")
    rows, labels, scores_arr = load_results(INPUT_CSV)
    strategy_scores = compute_strategy_scores(scores_arr)

    print("\nGenerating per-strategy visualizations...\n")

    for (strategy, algorithm), threshold in BEST_THRESHOLDS.items():
        print(f"Strategy: {strategy} ({algorithm})")
        scores = strategy_scores[strategy]
        out_sub = OUTPUT_DIR / strategy
        out_sub.mkdir(exist_ok=True)
        visualize_strategy(
            rows=rows,
            scores=scores,
            labels=labels,
            threshold=threshold,
            strategy_name=strategy,
            output_dir=out_sub,
            algorithm=algorithm,
        )

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
