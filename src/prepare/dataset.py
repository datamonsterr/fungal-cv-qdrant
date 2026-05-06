from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    CURATED_SOURCE_DATASET_PATH,
    FILE_EXTENSION,
    INCOMING_SOURCE_DATASET_PATH,
    PREPARED_DATASET_DIR,
    PREPARED_ITEMS_METADATA_PATH,
    PREPARED_SEGMENTS_METADATA_PATH,
    SOURCE_COLLECTIONS,
    STRAIN_SPECIES_MAPPING_PATH,
    TARGET_SIZE,
    relative_to_workspace,
)
from src.preprocessing.kmeans import (
    draw_bbox,
    segment_kmeans_image,
)
from src.preprocessing.preprocess import process_image

FALLBACK_VALUE = "unknown"
FILENAME_PATTERN = re.compile(r"(DTO\s[0-9]+-[A-Z0-9]+)\s+([A-Z0-9]+)(rev|ob)", re.IGNORECASE)
FOLDER_PATTERN = re.compile(r"(DTO\s[0-9]+-[A-Z0-9]+)\s+(.+)")


@dataclass(frozen=True)
class SourceCollection:
    key: str
    display_name: str
    quality_tier: str
    path: Path


@dataclass(frozen=True)
class ParsedMetadata:
    species: str
    strain: str
    environment: str
    angle: str
    parse_status: str


@dataclass(frozen=True)
class DatasetItemRecord:
    item_id: str
    source_collection: str
    source_collection_path: str
    source_filename: str
    species: str
    strain: str
    environment: str
    angle: str
    parse_status: str
    source_image_path: str
    prepared_image_path: str
    artifact_root: str
    item_record_path: str


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    parent_item_id: str
    method: str
    segment_index: int
    segment_path: str
    species: str
    strain: str
    environment: str
    angle: str
    bbox: dict[str, int]


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or FALLBACK_VALUE


def normalize_label(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return FALLBACK_VALUE
    return cleaned.replace("/", "-")


def sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem.removesuffix("_edited")
    return slugify(stem)


def load_source_collections() -> dict[str, SourceCollection]:
    return {
        key: SourceCollection(
            key=key,
            display_name=str(config["display_name"]),
            quality_tier=str(config["quality_tier"]),
            path=Path(config["path"]),
        )
        for key, config in SOURCE_COLLECTIONS.items()
    }


def load_strain_species_mapping(
    mapping_path: Path = STRAIN_SPECIES_MAPPING_PATH,
) -> dict[str, str]:
    if not mapping_path.exists():
        return {}
    frame = pd.read_csv(mapping_path)
    if "Strain" not in frame.columns or "Species" not in frame.columns:
        return {}
    return {
        str(row["Strain"]): str(row["Species"])
        for _, row in frame[["Strain", "Species"]].dropna().iterrows()
    }


def parse_source_metadata(
    image_path: Path,
    strain_species_mapping: dict[str, str],
) -> ParsedMetadata:
    filename = image_path.name.removesuffix(FILE_EXTENSION).removesuffix("_edited")
    match = FILENAME_PATTERN.match(filename)

    strain = FALLBACK_VALUE
    environment = FALLBACK_VALUE
    angle = FALLBACK_VALUE
    parse_status = "fallback"

    if match:
        strain = normalize_label(match.group(1))
        environment = normalize_label(match.group(2).upper())
        angle = normalize_label(match.group(3).lower())
        parse_status = "parsed"

    species = normalize_label(strain_species_mapping.get(strain, FALLBACK_VALUE))

    if species == FALLBACK_VALUE:
        folder_match = FOLDER_PATTERN.match(image_path.parent.name)
        if folder_match:
            species = normalize_label(folder_match.group(2))
            if strain == FALLBACK_VALUE:
                strain = normalize_label(folder_match.group(1))
                parse_status = "folder_inferred"
        elif image_path.parent.name:
            species = normalize_label(image_path.parent.name)

    return ParsedMetadata(
        species=species,
        strain=strain,
        environment=environment,
        angle=angle,
        parse_status=parse_status,
    )


def iter_source_images(source_collection: SourceCollection) -> list[Path]:
    if not source_collection.path.exists():
        return []
    return sorted(
        path
        for path in source_collection.path.rglob(f"*{FILE_EXTENSION}")
        if path.is_file()
    )


def build_item_id(source_collection: str, image_path: Path) -> str:
    seed = f"{source_collection}:{image_path.as_posix()}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex


def build_artifact_root(
    prepared_root: Path,
    metadata: ParsedMetadata,
    image_stem: str,
) -> Path:
    return (
        prepared_root
        / slugify(metadata.species)
        / slugify(metadata.strain)
        / slugify(metadata.environment)
        / image_stem
    )


def write_item_record(artifact_root: Path, record: DatasetItemRecord) -> None:
    with open(artifact_root / "item.json", "w") as handle:
        json.dump(asdict(record), handle, indent=2)


def prepare_dataset(
    *,
    source_collections: list[str] | None = None,
    prepared_root: Path = PREPARED_DATASET_DIR,
    items_metadata_path: Path = PREPARED_ITEMS_METADATA_PATH,
    segments_metadata_path: Path = PREPARED_SEGMENTS_METADATA_PATH,
    limit: int | None = None,
) -> tuple[list[DatasetItemRecord], list[SegmentRecord]]:
    available_collections = load_source_collections()
    requested_collections = source_collections or list(available_collections)
    unknown = [name for name in requested_collections if name not in available_collections]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown source collections: {names}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    strain_species_mapping = load_strain_species_mapping()
    item_records: list[DatasetItemRecord] = []
    segment_records: list[SegmentRecord] = []
    processed = 0

    for collection_name in requested_collections:
        collection = available_collections[collection_name]
        for image_path in iter_source_images(collection):
            if limit is not None and processed >= limit:
                break

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            metadata = parse_source_metadata(image_path, strain_species_mapping)
            item_id = build_item_id(collection_name, image_path)
            artifact_root = build_artifact_root(
                prepared_root,
                metadata,
                sanitize_stem(image_path.name),
            )
            artifact_root.mkdir(parents=True, exist_ok=True)

            source_output_path = artifact_root / f"source{FILE_EXTENSION}"
            prepared_output_path = artifact_root / f"prepared{FILE_EXTENSION}"
            shutil.copyfile(image_path, source_output_path)
            prepared_image = process_image(image, output_size=TARGET_SIZE[0])
            cv2.imwrite(str(prepared_output_path), prepared_image)

            record = DatasetItemRecord(
                item_id=item_id,
                source_collection=collection.display_name,
                source_collection_path=relative_to_workspace(collection.path),
                source_filename=image_path.name,
                species=metadata.species,
                strain=metadata.strain,
                environment=metadata.environment,
                angle=metadata.angle,
                parse_status=metadata.parse_status,
                source_image_path=relative_to_workspace(source_output_path),
                prepared_image_path=relative_to_workspace(prepared_output_path),
                artifact_root=relative_to_workspace(artifact_root),
                item_record_path=relative_to_workspace(artifact_root / "item.json"),
            )
            write_item_record(artifact_root, record)
            item_records.append(record)
            processed += 1

        if limit is not None and processed >= limit:
            break

    with open(items_metadata_path, "w") as handle:
        json.dump([asdict(record) for record in item_records], handle, indent=2)
    with open(segments_metadata_path, "w") as handle:
        json.dump([asdict(record) for record in segment_records], handle, indent=2)

    return item_records, segment_records


def resolve_source_collection_names(selected: list[str] | None) -> list[str]:
    if not selected:
        return list(SOURCE_COLLECTIONS)
    return selected


SEGMENT_METHODS = ["kmeans", "contour"]
SEGMENT_METHOD_KMEANS = "kmeans"
SEGMENT_METHOD_CONTOUR = "contour"


@dataclass
class SegmentationResult:
    method: str
    status: str
    segment_count: int
    segments_dir: str
    bbox_visualization_path: str | None = None
    pipeline_visualization_path: str | None = None
    failure_reason: str | None = None


def _contour_bboxes(image: np.ndarray) -> list[dict[str, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blur, 30, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scored: list[tuple[float, np.ndarray]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circ = (4 * math.pi * area) / (perimeter**2)
        if circ < 0.1:
            continue
        scored.append((area * circ, cnt))

    scored.sort(key=lambda x: x[0], reverse=True)
    bboxes: list[dict[str, int]] = []
    for _, cnt in scored[:3]:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append({"xmin": int(x), "ymin": int(y), "xmax": int(x + w), "ymax": int(y + h)})
    return bboxes


def _build_pipeline_visualization(
    source_image: np.ndarray,
    prepared_image: np.ndarray,
    bbox_image: np.ndarray,
) -> np.ndarray:
    h, w = prepared_image.shape[:2]
    src = cv2.resize(source_image, (w, h), interpolation=cv2.INTER_AREA)
    return np.hstack([src, prepared_image, bbox_image])


def segment_item(
    item_record: DatasetItemRecord,
    *,
    methods: list[str] | None = None,
) -> list[SegmentationResult]:
    selected_methods = methods or SEGMENT_METHODS
    unknown = [m for m in selected_methods if m not in SEGMENT_METHODS]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown segment methods: {names}")

    from src.config import WORKSPACE_ROOT

    artifact_root = WORKSPACE_ROOT / item_record.artifact_root
    prepared_path = artifact_root / f"prepared{FILE_EXTENSION}"
    source_path = artifact_root / f"source{FILE_EXTENSION}"

    prepared_image = cv2.imread(str(prepared_path))
    if prepared_image is None:
        return [
            SegmentationResult(method=m, status="failed", segment_count=0,
                               segments_dir=str(artifact_root / f"segments_{m}"),
                               failure_reason="prepared image not found or unreadable")
            for m in selected_methods
        ]

    source_image = cv2.imread(str(source_path))

    results: list[SegmentationResult] = []
    for method in selected_methods:
        result = _segment_with_method(item_record, prepared_image, source_image, artifact_root, method)
        results.append(result)
    return results


def _segment_with_method(
    item_record: DatasetItemRecord,
    prepared_image: np.ndarray,
    source_image: np.ndarray | None,
    artifact_root: Path,
    method: str,
) -> SegmentationResult:
    segments_dir = artifact_root / f"segments_{method}"
    bbox_path = artifact_root / f"bbox_{method}{FILE_EXTENSION}"
    pipeline_path = artifact_root / f"pipeline_{method}{FILE_EXTENSION}"

    try:
        if method == SEGMENT_METHOD_KMEANS:
            bboxes, _ = segment_kmeans_image(prepared_image)
        elif method == SEGMENT_METHOD_CONTOUR:
            bboxes = _contour_bboxes(prepared_image)
        else:
            return SegmentationResult(
                method=method, status="skipped", segment_count=0,
                segments_dir=str(segments_dir),
                failure_reason=f"unknown method {method}",
            )
    except Exception as exc:
        return SegmentationResult(
            method=method, status="failed", segment_count=0,
            segments_dir=str(segments_dir),
            failure_reason=str(exc),
        )

    if not bboxes:
        segments_dir.mkdir(parents=True, exist_ok=True)
        result = SegmentationResult(
            method=method, status="empty", segment_count=0,
            segments_dir=relative_to_workspace(segments_dir),
        )
    else:
        segments_dir.mkdir(parents=True, exist_ok=True)
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            crop = prepared_image[max(0, y1):max(y1, y2), max(0, x1):max(x1, x2)]
            if crop.size > 0:
                seg_path = segments_dir / f"seg_{i}{FILE_EXTENSION}"
                cv2.imwrite(str(seg_path), crop)

        bbox_image = draw_bbox(prepared_image, bboxes)
        cv2.imwrite(str(bbox_path), bbox_image)

        if source_image is not None:
            pipeline_img = _build_pipeline_visualization(source_image, prepared_image, bbox_image)
            cv2.imwrite(str(pipeline_path), pipeline_img)

        result = SegmentationResult(
            method=method, status="success", segment_count=len(bboxes),
            segments_dir=relative_to_workspace(segments_dir),
            bbox_visualization_path=relative_to_workspace(bbox_path),
            pipeline_visualization_path=relative_to_workspace(pipeline_path),
        )

    return result


def run_segmentation(
    item_records: list[DatasetItemRecord],
    *,
    methods: list[str] | None = None,
    limit: int | None = None,
) -> list[SegmentRecord]:
    from src.config import WORKSPACE_ROOT as ws_root

    segment_records: list[SegmentRecord] = []
    processed = 0

    for item_record in item_records:
        if limit is not None and processed >= limit:
            break

        results = segment_item(item_record, methods=methods)
        for result in results:
            if result.status not in ("success", "partial"):
                continue
            segments_dir = ws_root / result.segments_dir
            for seg_path in sorted(segments_dir.glob(f"seg_*{FILE_EXTENSION}")):
                idx = int(seg_path.stem.removeprefix("seg_"))
                seg_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{item_record.item_id}:{result.method}:{idx}",
                ).hex
                h, w = (256, 256)
                segment_records.append(
                    SegmentRecord(
                        segment_id=seg_id,
                        parent_item_id=item_record.item_id,
                        method=result.method,
                        segment_index=idx,
                        segment_path=relative_to_workspace(seg_path),
                        species=item_record.species,
                        strain=item_record.strain,
                        environment=item_record.environment,
                        angle=item_record.angle,
                        bbox={"xmin": 0, "ymin": 0, "xmax": w, "ymax": h},
                    )
                )
        processed += 1

    return segment_records


def write_segment_metadata(
    segment_records: list[SegmentRecord],
    path: Path = PREPARED_SEGMENTS_METADATA_PATH,
) -> None:
    with open(path, "w") as handle:
        json.dump([asdict(record) for record in segment_records], handle, indent=2)


def required_source_roots() -> list[Path]:
    return [CURATED_SOURCE_DATASET_PATH, INCOMING_SOURCE_DATASET_PATH]
