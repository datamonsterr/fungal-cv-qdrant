import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import relative_to_workspace  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def display_path(path: Path) -> str:
    try:
        return relative_to_workspace(path)
    except ValueError:
        return str(path.resolve())


def iter_image_files(images_root: Path) -> list[Path]:
    return sorted(
        path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def yolo_line_to_bbox(line: str, image_width: int, image_height: int) -> dict[str, int]:
    class_id, x_center, y_center, width, height = line.split()
    if class_id != "0":
        raise ValueError(f"Unsupported class id: {class_id}")

    x_center_f = float(x_center) * image_width
    y_center_f = float(y_center) * image_height
    width_f = float(width) * image_width
    height_f = float(height) * image_height

    xmin = max(0, int(round(x_center_f - width_f / 2)))
    ymin = max(0, int(round(y_center_f - height_f / 2)))
    xmax = min(image_width, int(round(x_center_f + width_f / 2)))
    ymax = min(image_height, int(round(y_center_f + height_f / 2)))
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": max(xmin + 1, xmax),
        "ymax": max(ymin + 1, ymax),
    }


def ensure_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output path already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def crop_segments(input_root: Path, output_root: Path, limit: int | None) -> int:
    images_root = input_root / "images"
    labels_root = input_root / "labels"
    image_paths = iter_image_files(images_root)
    if limit is not None:
        image_paths = image_paths[:limit]

    total_crops = 0
    for index, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(images_root)
        label_path = labels_root / relative.with_suffix(".txt")
        if not label_path.exists():
            print(f"[{index}] SKIP missing label file: {display_path(label_path)}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}] SKIP unreadable image: {display_path(image_path)}")
            continue

        lines = [
            line.strip() for line in label_path.read_text().splitlines() if line.strip()
        ]
        crop_dir = output_root / relative.with_suffix("")
        crop_dir.mkdir(parents=True, exist_ok=True)

        for bbox_index, line in enumerate(lines):
            bbox = yolo_line_to_bbox(line, image.shape[1], image.shape[0])
            crop = image[bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]]
            resized = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
            crop_path = crop_dir / f"{relative.stem}__bbox{bbox_index}.jpg"
            ok = cv2.imwrite(str(crop_path), resized)
            if not ok:
                raise RuntimeError(f"Failed to write image to {crop_path}")
            total_crops += 1

        print(
            f"[{index}] OK {relative.stem}: {len(lines)} crop(s) -> {display_path(crop_dir)}"
        )

    return total_crops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop 512x512 colony images from YOLO image/label pairs."
    )
    parser.add_argument("--input", required=True, help="Path to the YOLO dataset root")
    parser.add_argument(
        "--output", required=True, help="Output directory for cropped images"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional maximum number of source images to process",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    ensure_output_root(output_root)
    total_crops = crop_segments(input_root, output_root, args.n)

    print("\nCrop complete")
    print(f"Total crops: {total_crops}")
    print(f"Output root: {display_path(output_root)}")


if __name__ == "__main__":
    main()
