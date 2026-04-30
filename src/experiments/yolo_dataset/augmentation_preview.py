from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.config import RESULTS_DIR

AUGMENTATION_PREVIEW_ROOT = RESULTS_DIR / "augmentation_preview"


def apply_preview_augmentation(image: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    augmented = image.convert("RGB")

    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)

    angle = rng.uniform(-12, 12)
    augmented = augmented.rotate(angle, resample=Image.Resampling.BILINEAR)

    brightness = ImageEnhance.Brightness(augmented)
    augmented = brightness.enhance(rng.uniform(0.9, 1.1))

    contrast = ImageEnhance.Contrast(augmented)
    augmented = contrast.enhance(rng.uniform(0.9, 1.15))

    color = ImageEnhance.Color(augmented)
    augmented = color.enhance(rng.uniform(0.9, 1.1))

    if rng.random() < 0.35:
        augmented = augmented.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0))
        )

    return augmented


def render_augmentation_preview_grid(
    image_path: Path,
    output_path: Path | None = None,
    preview_count: int = 6,
    columns: int = 3,
    base_seed: int = 42,
) -> Path:
    source_image = Image.open(image_path).convert("RGB")
    frames = [source_image] + [
        apply_preview_augmentation(source_image, base_seed + index)
        for index in range(preview_count)
    ]

    column_count = max(1, columns)
    row_count = math.ceil(len(frames) / column_count)
    tile_width, tile_height = source_image.size
    grid = Image.new(
        "RGB",
        (tile_width * column_count, tile_height * row_count),
        color=(255, 255, 255),
    )

    for index, frame in enumerate(frames):
        x_offset = (index % column_count) * tile_width
        y_offset = (index // column_count) * tile_height
        grid.paste(frame.resize((tile_width, tile_height)), (x_offset, y_offset))

    target_path = output_path or (
        AUGMENTATION_PREVIEW_ROOT / f"{image_path.stem}_preview.jpg"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(target_path)
    return target_path


def build_augmentation_preview_summary(
    image_path: Path,
    output_path: Path | None = None,
    preview_count: int = 6,
    columns: int = 3,
    base_seed: int = 42,
) -> dict[str, object]:
    preview_path = render_augmentation_preview_grid(
        image_path=image_path,
        output_path=output_path,
        preview_count=preview_count,
        columns=columns,
        base_seed=base_seed,
    )
    return {
        "image_path": str(image_path),
        "preview_path": str(preview_path),
        "preview_count": preview_count,
        "columns": columns,
        "base_seed": base_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an augmentation preview grid")
    parser.add_argument("image", type=Path, help="Source image path")
    parser.add_argument("--output", type=Path, default=None, help="Output preview path")
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build_augmentation_preview_summary(
                image_path=args.image,
                output_path=args.output,
                preview_count=args.preview_count,
                columns=args.columns,
                base_seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
