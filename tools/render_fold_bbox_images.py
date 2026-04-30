import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import relative_to_workspace  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 128, 0)


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


def load_class_names(classes_path: Path) -> list[str]:
    return [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]


def polygon_line_to_bbox(line: str, image_width: int, image_height: int) -> tuple[int, dict[str, int]]:
    parts = line.split()
    if len(parts) < 7 or len(parts[1:]) % 2 != 0:
        raise ValueError(f"Unsupported segmentation line: {line}")

    class_id = int(parts[0])
    coords = [float(value) for value in parts[1:]]
    xs = [coords[index] * image_width for index in range(0, len(coords), 2)]
    ys = [coords[index] * image_height for index in range(1, len(coords), 2)]

    xmin = max(0, int(round(min(xs))))
    ymin = max(0, int(round(min(ys))))
    xmax = min(image_width, int(round(max(xs))))
    ymax = min(image_height, int(round(max(ys))))
    return class_id, {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": max(xmin + 1, xmax),
        "ymax": max(ymin + 1, ymax),
    }


def draw_labeled_bbox(image, bbox: dict[str, int], label: str) -> None:
    cv2.rectangle(
        image,
        (bbox["xmin"], bbox["ymin"]),
        (bbox["xmax"], bbox["ymax"]),
        BOX_COLOR,
        2,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    text_x = bbox["xmin"]
    text_y = max(text_height + 4, bbox["ymin"] - 6)
    cv2.rectangle(
        image,
        (text_x, text_y - text_height - 4),
        (text_x + text_width + 6, text_y + baseline),
        TEXT_BG,
        -1,
    )
    cv2.putText(image, label, (text_x + 3, text_y - 2), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


def ensure_output_root(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)


def render_bbox_images(input_root: Path, output_root: Path, limit: int | None) -> int:
    classes = load_class_names(input_root / "classes.txt")
    images_root = input_root / "test" / "images"
    labels_root = input_root / "test" / "labels"
    image_paths = iter_image_files(images_root)
    if limit is not None:
        image_paths = image_paths[:limit]

    rendered = 0
    for index, image_path in enumerate(image_paths, start=1):
        label_path = labels_root / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[{index}] SKIP missing label file: {display_path(label_path)}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[{index}] SKIP unreadable image: {display_path(image_path)}")
            continue

        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        for line in lines:
            class_id, bbox = polygon_line_to_bbox(line, image.shape[1], image.shape[0])
            label = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
            draw_labeled_bbox(image, bbox, label)

        output_path = output_root / image_path.name
        ok = cv2.imwrite(str(output_path), image)
        if not ok:
            raise RuntimeError(f"Failed to write image to {output_path}")
        rendered += 1
        print(f"[{index}] OK {image_path.name} -> {display_path(output_path)}")

    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render labeled bounding boxes for fold_0 test images."
    )
    parser.add_argument(
        "--input",
        default="/home/dat/dev/mycoai/Dataset/manual_labeled_data_roboflow_species_cv/fold_0",
        help="Fold dataset root",
    )
    parser.add_argument(
        "--output",
        default="/home/dat/dev/mycoai/Dataset/manual_labeled_data_roboflow_species_cv/fold_0/bbox_images",
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional max number of test images to render",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    ensure_output_root(output_root)
    total = render_bbox_images(input_root, output_root, args.n)

    print("\nRender complete")
    print(f"Images rendered: {total}")
    print(f"Output root: {display_path(output_root)}")


if __name__ == "__main__":
    main()
