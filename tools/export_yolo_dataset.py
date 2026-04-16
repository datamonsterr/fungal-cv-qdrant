import argparse
import csv
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    ORIGINAL_DATASET_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
    relative_to_workspace,
)
from src.preprocessing.kmeans import draw_bbox, segment_kmeans_image  # noqa: E402
from src.preprocessing.preprocess import DEFAULT_EXPORT_SIZE, prepare_image  # noqa: E402

type BBox = dict[str, int]
type Image = Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAME = "colony"


def load_species_mapping(mapping_path: Path) -> dict[str, str]:
    if not mapping_path.exists():
        return {}

    mapping: dict[str, str] = {}
    with mapping_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strain = (row.get("Strain") or "").strip()
            species = (row.get("Species") or "").strip()
            if strain and species:
                mapping[strain] = species
    return mapping


def iter_source_images(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _parse_parent_folder(
    folder_name: str, species_mapping: dict[str, str]
) -> tuple[str, str]:
    match = re.match(r"^(DTO\s+\d+-[A-Z0-9]+)\s+(.*)$", folder_name)
    if match:
        strain = match.group(1).strip()
        species = match.group(2).strip() or species_mapping.get(strain, "unknown")
        return strain, species

    species = species_mapping.get(folder_name, "unknown")
    return folder_name, species


def _clean_suffixes(stem: str) -> str:
    cleaned = re.sub(r"_edited\d*$", "", stem, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(contaminated|delete)\b", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _parse_environment_and_angle(remainder: str) -> tuple[str, str]:
    token = re.sub(r"[^A-Za-z0-9]+", "", remainder).lower()
    if not token:
        return ("unknown", "unknown")

    if "rev" in token:
        environment = token.split("rev", 1)[0]
        return (environment.upper() or "unknown", "rev")
    if "ob" in token:
        environment = token.split("ob", 1)[0]
        return (environment.upper() or "unknown", "ob")
    if token.endswith(("r", "c")):
        environment = token[:-1]
        return (environment.upper() or "unknown", "rev")
    if token.endswith(("o", "a", "b")):
        environment = token[:-1]
        return (environment.upper() or "unknown", "ob")
    return (token.upper(), "unknown")


def parse_source_metadata(
    source_path: Path, species_mapping: dict[str, str]
) -> dict[str, str]:
    strain, species = _parse_parent_folder(source_path.parent.name, species_mapping)
    cleaned_stem = _clean_suffixes(source_path.stem)
    remainder = cleaned_stem
    if remainder.startswith(strain):
        remainder = remainder[len(strain) :].strip()
    environment, angle = _parse_environment_and_angle(remainder)
    return {
        "strain": strain,
        "species": species or "unknown",
        "environment": environment,
        "angle": angle,
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "unknown"


def display_path(path: Path) -> str:
    try:
        return relative_to_workspace(path)
    except ValueError:
        return str(path.resolve())


def write_image(path: Path, image: Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")


def scale_bboxes(
    bboxes: list[BBox],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> list[BBox]:
    from_height, from_width = from_shape
    to_height, to_width = to_shape
    scale_x = to_width / from_width
    scale_y = to_height / from_height

    scaled: list[BBox] = []
    for bbox in bboxes:
        xmin = max(0, min(int(round(bbox["xmin"] * scale_x)), to_width - 1))
        ymin = max(0, min(int(round(bbox["ymin"] * scale_y)), to_height - 1))
        xmax = max(xmin + 1, min(int(round(bbox["xmax"] * scale_x)), to_width))
        ymax = max(ymin + 1, min(int(round(bbox["ymax"] * scale_y)), to_height))
        scaled.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
    return scaled


def bbox_to_yolo_line(bbox: BBox, image_width: int, image_height: int) -> str:
    width = bbox["xmax"] - bbox["xmin"]
    height = bbox["ymax"] - bbox["ymin"]
    x_center = bbox["xmin"] + width / 2
    y_center = bbox["ymin"] + height / 2
    return (
        f"0 {x_center / image_width:.6f} {y_center / image_height:.6f} "
        f"{width / image_width:.6f} {height / image_height:.6f}"
    )


def write_label_file(
    path: Path, bboxes: list[BBox], image_width: int, image_height: int
) -> None:
    lines = [bbox_to_yolo_line(bbox, image_width, image_height) for bbox in bboxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_dataset_descriptor(output_root: Path) -> None:
    dataset_yaml = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images",
            "val: images",
            "names:",
            f"  0: {CLASS_NAME}",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(dataset_yaml)
    (output_root / "classes.txt").write_text(f"{CLASS_NAME}\n")


def _to_bgr(image: Image) -> Image:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _panel(label: str, image: Image, size: int = 320) -> Image:
    bgr = _to_bgr(image)
    height, width = bgr.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size + 40, size, 3), 245, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    cv2.putText(
        canvas,
        label,
        (12, size + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_pipeline_visualization(
    original_image: Image,
    preprocessed_image: Image,
    debug_images: dict[str, Image],
    bbox_image: Image,
) -> Image:
    panels = [
        _panel("original", original_image),
        _panel("preprocessed", preprocessed_image),
        _panel("kmeans-color", debug_images["color_dimension"]),
        _panel("foreground-mask", debug_images["foreground_mask"]),
        _panel("kmeans-location", debug_images["location_clusters"]),
        _panel("bounding-box", bbox_image),
    ]
    gap = np.full((panels[0].shape[0], 12, 3), 255, dtype=np.uint8)
    strip = panels[0]
    for panel in panels[1:]:
        strip = np.hstack((strip, gap, panel))
    return strip


def ensure_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"Output path already exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(exist_ok=True)
    (output_root / "labels").mkdir(exist_ok=True)
    (output_root / "hierarchical").mkdir(exist_ok=True)


def export_dataset(
    output_root: Path,
    *,
    limit: int | None,
    visualize: bool,
    quality_threshold: float = 0.5,
) -> list[dict[str, object]]:
    species_mapping = load_species_mapping(STRAIN_SPECIES_MAPPING_PATH)
    source_images = iter_source_images(ORIGINAL_DATASET_PATH)

    import random

    rng = random.Random(42)
    rng.shuffle(source_images)

    if limit is not None:
        source_images = source_images[:limit]

    records: list[dict[str, object]] = []

    # Ensure test output folders are created for easy viewing if this is the test run
    is_test_run = "yolo_quality_test_20" in str(output_root)
    if is_test_run:
        (output_root / "with_3_boxes").mkdir(parents=True, exist_ok=True)
        (output_root / "pipeline_viz").mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(source_images, start=1):
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"[{index}] SKIP unreadable image: {source_path}")
            continue

        metadata = parse_source_metadata(source_path, species_mapping)
        sample_id = uuid.uuid5(uuid.NAMESPACE_URL, str(source_path.resolve())).hex[:12]
        sample_basename = "_".join(
            [
                slugify(metadata["strain"]),
                slugify(metadata["environment"]),
                slugify(metadata["angle"]),
                sample_id,
            ]
        )

        artifacts = prepare_image(image, export_size=DEFAULT_EXPORT_SIZE)
        if visualize:
            working_bboxes, quality_score, debug_images = segment_kmeans_image(
                artifacts.masked_working_image,
                plate_mask=artifacts.working_mask,
                return_debug=True,
            )
        else:
            working_bboxes, quality_score = segment_kmeans_image(
                artifacts.masked_working_image,
                plate_mask=artifacts.working_mask,
                return_debug=False,
            )
            debug_images = {}

        if quality_score < quality_threshold or len(working_bboxes) != 3:
            export_bboxes = []
        else:
            export_bboxes = scale_bboxes(
                working_bboxes,
                artifacts.masked_working_image.shape[:2],
                artifacts.export_image.shape[:2],
            )

        viz_bboxes = scale_bboxes(
            working_bboxes,
            artifacts.masked_working_image.shape[:2],
            artifacts.export_image.shape[:2],
        )
        bbox_image = draw_bbox(artifacts.export_image, viz_bboxes)

        image_path = output_root / "images" / f"{sample_basename}.jpg"
        label_path = output_root / "labels" / f"{sample_basename}.txt"
        write_image(image_path, artifacts.export_image)
        write_label_file(
            label_path,
            export_bboxes,
            artifacts.export_image.shape[1],
            artifacts.export_image.shape[0],
        )

        leaf_dir = (
            output_root
            / "hierarchical"
            / metadata["species"]
            / metadata["strain"]
            / metadata["environment"]
            / sample_basename
        )
        original_leaf = leaf_dir / "original_resized.jpg"
        bbox_leaf = leaf_dir / "bbox_visualization.jpg"
        write_image(original_leaf, artifacts.export_image)
        write_image(bbox_leaf, bbox_image)

        visualization_paths: dict[str, str] = {
            "original_resized": display_path(original_leaf),
            "bbox_visualization": display_path(bbox_leaf),
        }
        if visualize:
            pipeline_leaf = leaf_dir / "pipeline_visualization.jpg"
            pipeline_image = build_pipeline_visualization(
                artifacts.export_image,
                artifacts.masked_export_image,
                debug_images,
                bbox_image,
            )
            write_image(pipeline_leaf, pipeline_image)
            visualization_paths["pipeline_visualization"] = display_path(pipeline_leaf)

            if is_test_run:
                write_image(
                    output_root / "pipeline_viz" / f"{sample_basename}.jpg",
                    pipeline_image,
                )

        if (
            is_test_run
            and len(working_bboxes) == 3
            and quality_score >= quality_threshold
        ):
            write_image(
                output_root / "with_3_boxes" / f"{sample_basename}.jpg",
                artifacts.export_image,
            )

        record: dict[str, object] = {
            "sample_id": sample_id,
            "source_path": display_path(source_path),
            "image_path": display_path(image_path),
            "label_path": display_path(label_path),
            "hierarchical_dir": display_path(leaf_dir),
            "metadata": metadata,
            "image_size": {
                "width": artifacts.export_image.shape[1],
                "height": artifacts.export_image.shape[0],
            },
            "circle_detected": artifacts.circle_detected,
            "bboxes": export_bboxes,
            "quality_score": quality_score,
            "visualizations": visualization_paths,
        }
        records.append(record)

        status = "OK" if len(export_bboxes) == 3 else "BAD"
        print(
            f"[{index}] {status} {sample_basename}: {len(working_bboxes)} box(es), score {quality_score:.3f} -> {display_path(leaf_dir)}"
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Dataset/original into a YOLO-style curation dataset."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for the export"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional maximum number of images to process",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write pipeline visualization assets for each processed image",
    )
    args = parser.parse_args()

    output_root = Path(args.output).resolve()
    ensure_output_root(output_root)
    records = export_dataset(output_root, limit=args.n, visualize=args.visualize)

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2))
    write_dataset_descriptor(output_root)

    print("\nExport complete")
    print(f"Processed images: {len(records)}")
    print(f"Metadata: {display_path(metadata_path)}")
    print(f"Output root: {display_path(output_root)}")


if __name__ == "__main__":
    main()
