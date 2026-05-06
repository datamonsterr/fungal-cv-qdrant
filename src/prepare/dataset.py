from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import (
    COLLECTION_METADATA_PATHS,
    CURATED_FILENAME_PATTERN,
    CURATED_SOURCE_DATASET_PATH,
    FILE_EXTENSION,
    INCOMING_SOURCE_DATASET_PATH,
    LETTER_RANGE_PATTERN,
    PREPARED_DATASET_DIR,
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


@dataclass
class InstanceInfo:
    species: str
    strain: str
    environment: str
    angle: str


@dataclass
class DatasetItemRecord:
    item_id: str
    source_collection: str
    source_collection_path: str
    source_filename: str
    instance_info: InstanceInfo
    parse_status: str
    paths: dict[str, object] = field(default_factory=dict)
    segmentation: dict[str, list[dict[str, int]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "source_collection": self.source_collection,
            "source_collection_path": self.source_collection_path,
            "source_filename": self.source_filename,
            "instance_info": asdict(self.instance_info),
            "parse_status": self.parse_status,
            "paths": self.paths,
            "segmentation": self.segmentation,
        }


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


SEGMENT_METHODS = ["kmeans", "contour"]
SEGMENT_METHOD_KMEANS = "kmeans"
SEGMENT_METHOD_CONTOUR = "contour"


def _is_letter_range(name: str) -> bool:
    return bool(LETTER_RANGE_PATTERN.match(name))


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
    mapping_path: Path | None = None,
) -> dict[str, str]:
    if mapping_path is None:
        mapping_path = STRAIN_SPECIES_MAPPING_PATH
    if not mapping_path.exists():
        return {}
    frame = pd.read_csv(mapping_path)
    if "Strain" not in frame.columns or "Species" not in frame.columns:
        return {}
    return {
        str(row["Strain"]): str(row["Species"])
        for _, row in frame[["Strain", "Species"]].dropna().iterrows()
    }


def parse_curated_metadata(
    image_path: Path,
    strain_species_mapping: dict[str, str],
) -> ParsedMetadata:
    filename = image_path.name.removesuffix(FILE_EXTENSION).removesuffix("_edited")
    match = CURATED_FILENAME_PATTERN.match(filename)

    strain = FALLBACK_VALUE
    environment = FALLBACK_VALUE
    angle = FALLBACK_VALUE
    parse_status = "fallback"

    if match:
        strain = normalize_label(match.group("strain"))
        environment = normalize_label(match.group("environment").upper())
        angle = normalize_label(match.group("angle").lower())
        parse_status = "parsed"

    species = normalize_label(strain_species_mapping.get(strain, FALLBACK_VALUE))

    if species == FALLBACK_VALUE:
        parent_name = image_path.parent.name
        if parent_name and parent_name != image_path.parent.parent.name:
            matched = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s+(.+)", parent_name)
            if matched:
                species = normalize_label(matched.group(2))
                if strain == FALLBACK_VALUE:
                    strain = normalize_label(matched.group(1))
                parse_status = "folder_inferred"
            else:
                species = normalize_label(parent_name)

    return ParsedMetadata(
        species=species,
        strain=strain,
        environment=environment,
        angle=angle,
        parse_status=parse_status,
    )


def _normalize_angle(raw: str) -> str:
    """Map angle tokens to canonical 'ob' or 'rev'. Returns FALLBACK_VALUE if unknown."""
    token = raw.strip().lower().rstrip("0123456789")
    if token in ("ob", "o"):
        return "ob"
    if token in ("rev", "r"):
        return "rev"
    return FALLBACK_VALUE


def _parse_incoming_filename(stem: str) -> tuple[str, str, str] | None:
    """Return (strain, environment, angle) or None if completely unparseable.

    Handles:
      - Standard:          "T230 CREA ob"  / "T230 CREA ob_1"
      - ob1/ob2 variant:   "T395 MEA ob1"
      - Reversed:          "T382 ob YES"
      - No-angle:          "T342 MEA"
      - IBT/CBS strain:    "IBT 22520 CREA ob" / "CBS 641.95 CYA ob"
      - T(N) strain:       "T(N) CREA ob"
      - Species-ref flat:  "flavigenum CBS 419_89 MEAo"  (angle = o/r suffix on env token)
    """
    # 1. Standard / IBT / T(N): STRAIN ENV ANGLE[_N] or STRAIN ENV_ANGLE (no space)
    m = re.match(
        r"^(?P<strain>T\([^)]+\)|IBT\s*[0-9]+|CBS\s*[0-9._]+|T[0-9]+|[0-9]+)"
        r"\s+(?P<environment>[A-Z][A-Z0-9]*)"
        r"\s+(?P<angle>(?:ob|rev|o|r)[0-9]*(?:[_\s][0-9]+)?)"
        r"(?:\s.*)?$",
        stem, re.IGNORECASE,
    )
    if m:
        angle_raw = re.sub(r"[0-9_].*$", "", m.group("angle"))
        angle = _normalize_angle(angle_raw)
        if angle != FALLBACK_VALUE:
            return (
                normalize_label(m.group("strain")),
                normalize_label(m.group("environment").upper()),
                angle,
            )

    # 2. Reversed: STRAIN ANGLE ENV
    m = re.match(
        r"^(?P<strain>T[0-9]+|IBT\s+[0-9]+)"
        r"\s+(?P<angle>ob|rev|o|r)"
        r"\s+(?P<environment>[A-Z][A-Z0-9]*)$",
        stem, re.IGNORECASE,
    )
    if m:
        angle = _normalize_angle(m.group("angle"))
        if angle != FALLBACK_VALUE:
            return (
                normalize_label(m.group("strain")),
                normalize_label(m.group("environment").upper()),
                angle,
            )

    # 3. No angle: STRAIN ENV  (keep angle as fallback)
    m = re.match(
        r"^(?P<strain>T[0-9]+|IBT\s+[0-9]+|CBS\s+[0-9._]+)"
        r"\s+(?P<environment>[A-Z][A-Z0-9]*)$",
        stem, re.IGNORECASE,
    )
    if m:
        return (
            normalize_label(m.group("strain")),
            normalize_label(m.group("environment").upper()),
            FALLBACK_VALUE,
        )

    # 4. Species-ref flat: SPECIES CBS/IBT_id ENV{o|r}
    #    e.g. "flavigenum CBS 419_89 CYAo", "harmonense CBS 412_69 MEAr"
    #    Also handles no-space: "dipodomiys CBS170_87 CYAo"
    m = re.match(
        r"^\S+\s+(?P<strain>(?:CBS|IBT)[_\s]?[0-9][0-9._]+)\s+(?P<enva>[A-Z0-9]+)$",
        stem, re.IGNORECASE,
    )
    if m:
        enva = m.group("enva")
        # Angle suffix is last character 'o' or 'r'
        angle_char = enva[-1].lower()
        angle = _normalize_angle(angle_char)
        env = enva[:-1].upper() if angle != FALLBACK_VALUE else enva.upper()
        strain_raw = re.sub(r"[_]", " ", m.group("strain"))
        return (
            normalize_label(strain_raw),
            normalize_label(env),
            angle,
        )

    return None


def _infer_species_from_path(image_path: Path) -> str:
    """Walk ancestry to find a species folder name (skip strain-looking names)."""
    for ancestor in image_path.parents:
        name = ancestor.name
        if not name or name in ("", "/"):
            break
        if _is_letter_range(name):
            break
        if re.match(r"^(?:T[0-9()N]+|IBT\s|CBS\s|[0-9]+)$", name, re.IGNORECASE):
            continue
        if name in ("new_data", "incoming_low_quality", "Dataset"):
            break
        return normalize_label(name)
    return FALLBACK_VALUE


def _infer_strain_from_path(image_path: Path) -> str:
    """Use parent dir name as strain if it looks like a strain id."""
    parent = image_path.parent.name
    if re.match(r"^(?:T[0-9()N]+|IBT\s+[0-9]+|CBS\s+[0-9._]+|[0-9]+)$", parent, re.IGNORECASE):
        # Normalise "317" → "T317" when dir is missing the T prefix
        if re.match(r"^[0-9]+$", parent):
            return normalize_label(f"T{parent}")
        return normalize_label(parent)
    return FALLBACK_VALUE


def parse_incoming_metadata(
    image_path: Path,
    strain_species_mapping: dict[str, str],
) -> ParsedMetadata:
    # Strip extension variants (.jpg, .JPG) and _edited suffix
    raw_stem = image_path.stem
    for suffix in ("_edited", " edited"):
        raw_stem = raw_stem.removesuffix(suffix)

    parsed = _parse_incoming_filename(raw_stem)

    strain = FALLBACK_VALUE
    environment = FALLBACK_VALUE
    angle = FALLBACK_VALUE
    parse_status = "fallback"

    if parsed:
        strain, environment, angle = parsed
        parse_status = "parsed"

    # Strain fallback from directory
    if strain == FALLBACK_VALUE:
        strain = _infer_strain_from_path(image_path)

    # Species from mapping first
    species = normalize_label(strain_species_mapping.get(strain, FALLBACK_VALUE))

    # Species fallback from path hierarchy
    if species == FALLBACK_VALUE:
        species = _infer_species_from_path(image_path)

    return ParsedMetadata(
        species=species,
        strain=strain,
        environment=environment,
        angle=angle,
        parse_status=parse_status,
    )


def parse_source_metadata(
    image_path: Path,
    strain_species_mapping: dict[str, str],
) -> ParsedMetadata:
    return parse_curated_metadata(image_path, strain_species_mapping)


def iter_curated_images(source_collection: SourceCollection) -> list[Path]:
    if not source_collection.path.exists():
        return []
    return sorted(
        path
        for path in source_collection.path.rglob(f"*{FILE_EXTENSION}")
        if path.is_file()
        and not _is_letter_range(path.parent.name)
        and not _is_letter_range(path.parent.parent.name)
    )


def _collect_images_from_dir(directory: Path, results: list[Path]) -> None:
    """Collect .jpg/.JPG images from directory, one level deep or flat."""
    img_exts = {".jpg", ".jpeg"}
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in img_exts:
            results.append(entry)
        elif entry.is_dir():
            for img_path in sorted(entry.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in img_exts:
                    results.append(img_path)


def iter_incoming_images(source_collection: SourceCollection) -> list[Path]:
    if not source_collection.path.exists():
        return []
    results: list[Path] = []
    for root_dir in sorted(source_collection.path.iterdir()):
        if not root_dir.is_dir():
            continue
        if not _is_letter_range(root_dir.name):
            # Non letter-range root: treat as species dir directly
            _collect_images_from_dir(root_dir, results)
            continue
        # Letter-range dir: descend into species dirs
        for species_dir in sorted(root_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            # Check if species dir contains images directly (flat case like scabrosum)
            has_direct_images = any(
                f.is_file() and f.suffix.lower() in {".jpg", ".jpeg"}
                for f in species_dir.iterdir()
            )
            if has_direct_images:
                for img_path in sorted(species_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg"}:
                        results.append(img_path)
            else:
                # Normal: species_dir contains strain subdirs
                for strain_dir in sorted(species_dir.iterdir()):
                    if not strain_dir.is_dir():
                        # Image directly in species dir alongside strain dirs
                        if strain_dir.is_file() and strain_dir.suffix.lower() in {".jpg", ".jpeg"}:
                            results.append(strain_dir)
                        continue
                    for img_path in sorted(strain_dir.iterdir()):
                        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg"}:
                            results.append(img_path)
    return results


def iter_source_images(source_collection: SourceCollection) -> list[Path]:
    if source_collection.key == "incoming":
        return iter_incoming_images(source_collection)
    return iter_curated_images(source_collection)


def build_item_id(instance_info: InstanceInfo) -> str:
    seed = "|".join([
        instance_info.species,
        instance_info.strain,
        instance_info.environment,
        instance_info.angle,
    ])
    return uuid.uuid5(uuid.NAMESPACE_URL, seed).hex


def build_leaf_dir(
    prepared_root: Path,
    metadata: ParsedMetadata,
) -> Path:
    return (
        prepared_root
        / slugify(metadata.species)
        / slugify(metadata.strain)
        / slugify(metadata.environment)
        / metadata.angle
    )


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


def _save_segment_crops(
    prepared_image: np.ndarray,
    bboxes: list[dict[str, int]],
    segments_dir: Path,
) -> list[str]:
    segments_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, bbox in enumerate(bboxes, start=1):
        if "xmin" in bbox:
            x1, y1 = bbox["xmin"], bbox["ymin"]
            x2, y2 = bbox["xmax"], bbox["ymax"]
        else:
            x1, y1 = bbox.get("x", 0), bbox.get("y", 0)
            x2 = x1 + bbox.get("w", 0)
            y2 = y1 + bbox.get("h", 0)
        crop = prepared_image[max(0, y1):max(y1, y2), max(0, x1):max(x1, x2)]
        if crop.size > 0:
            seg_path = segments_dir / f"segment_{i}{FILE_EXTENSION}"
            cv2.imwrite(str(seg_path), crop)
            paths.append(relative_to_workspace(seg_path))
    return paths


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


def _bboxes_to_schema(bboxes: list[dict[str, int]]) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for b in bboxes:
        if "xmin" in b:
            result.append({
                "x": b["xmin"],
                "y": b["ymin"],
                "w": b["xmax"] - b["xmin"],
                "h": b["ymax"] - b["ymin"],
            })
        else:
            result.append(b)
    return result


def _build_pipeline_visualization(
    source_image: np.ndarray,
    prepared_image: np.ndarray,
    bbox_image: np.ndarray,
) -> np.ndarray:
    h, w = prepared_image.shape[:2]
    src = cv2.resize(source_image, (w, h), interpolation=cv2.INTER_AREA)
    return np.hstack([src, prepared_image, bbox_image])


def prepare_dataset(
    *,
    source_collections: list[str] | None = None,
    prepared_root: Path = PREPARED_DATASET_DIR,
    limit: int | None = None,
) -> list[DatasetItemRecord]:
    available_collections = load_source_collections()
    requested_collections = source_collections or list(available_collections)
    unknown = [name for name in requested_collections if name not in available_collections]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown source collections: {names}")

    prepared_root.mkdir(parents=True, exist_ok=True)
    strain_species_mapping = load_strain_species_mapping()
    all_items: list[DatasetItemRecord] = []
    items_by_collection: dict[str, list[DatasetItemRecord]] = {}
    processed = 0

    for collection_name in requested_collections:
        collection = available_collections[collection_name]
        collection_items: list[DatasetItemRecord] = []
        for image_path in iter_source_images(collection):
            if limit is not None and processed >= limit:
                break

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            if collection_name == "incoming":
                metadata = parse_incoming_metadata(image_path, strain_species_mapping)
            else:
                metadata = parse_curated_metadata(image_path, strain_species_mapping)

            instance_info = InstanceInfo(
                species=metadata.species,
                strain=metadata.strain,
                environment=metadata.environment,
                angle=metadata.angle,
            )
            item_id = build_item_id(instance_info)
            leaf_dir = build_leaf_dir(prepared_root, metadata)
            leaf_dir.mkdir(parents=True, exist_ok=True)

            source_output_path = leaf_dir / f"source{FILE_EXTENSION}"
            prepared_output_path = leaf_dir / f"prepared{FILE_EXTENSION}"
            shutil.copyfile(image_path, source_output_path)
            prepared_image = process_image(image, output_size=TARGET_SIZE[0])
            cv2.imwrite(str(prepared_output_path), prepared_image)

            record = DatasetItemRecord(
                item_id=item_id,
                source_collection=collection.display_name,
                source_collection_path=relative_to_workspace(collection.path),
                source_filename=image_path.name,
                instance_info=instance_info,
                parse_status=metadata.parse_status,
                paths={
                    "source": relative_to_workspace(source_output_path),
                    "prepared": relative_to_workspace(prepared_output_path),
                    "segments": [],
                    "bbox_kmeans": None,
                    "bbox_contour": None,
                    "pipeline_kmeans": None,
                    "pipeline_contour": None,
                },
                segmentation={},
            )
            collection_items.append(record)
            all_items.append(record)
            processed += 1

        items_by_collection[collection_name] = collection_items

        if limit is not None and processed >= limit:
            break

    for collection_name, items in items_by_collection.items():
        metadata_path = COLLECTION_METADATA_PATHS.get(collection_name)
        if metadata_path is None:
            continue
        with open(metadata_path, "w") as handle:
            json.dump([item.to_dict() for item in items], handle, indent=2)

    return all_items


def segment_item(
    item_record: DatasetItemRecord,
    *,
    methods: list[str] | None = None,
) -> list[dict]:
    selected_methods = methods or SEGMENT_METHODS
    unknown = [m for m in selected_methods if m not in SEGMENT_METHODS]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown segment methods: {names}")

    from src.config import WORKSPACE_ROOT

    prepared_rel = item_record.paths.get("prepared")
    source_rel = item_record.paths.get("source")
    if not prepared_rel:
        return [{"method": m, "status": "failed", "reason": "no prepared path"} for m in selected_methods]

    leaf_dir = WORKSPACE_ROOT / prepared_rel
    leaf_dir = leaf_dir.parent
    source_path = WORKSPACE_ROOT / source_rel if source_rel else None

    prepared_image = cv2.imread(str(WORKSPACE_ROOT / prepared_rel))
    if prepared_image is None:
        return [
            {"method": m, "status": "failed", "reason": "prepared image unreadable"}
            for m in selected_methods
        ]

    source_image = cv2.imread(str(source_path)) if source_path and source_path.exists() else None

    segments_dir = leaf_dir / "segments"
    all_segments: list[str] = []
    results: list[dict] = []

    for method in selected_methods:
        try:
            if method == SEGMENT_METHOD_KMEANS:
                bboxes, _ = segment_kmeans_image(prepared_image)
            elif method == SEGMENT_METHOD_CONTOUR:
                bboxes = _contour_bboxes(prepared_image)
            else:
                results.append({"method": method, "status": "skipped"})
                continue
        except Exception as exc:
            results.append({"method": method, "status": "failed", "reason": str(exc)})
            continue

        if not bboxes:
            results.append({"method": method, "status": "empty", "bboxes": []})
            continue

        seg_paths = _save_segment_crops(prepared_image, bboxes, segments_dir)
        all_segments.extend(seg_paths)

        bbox_path = leaf_dir / f"bbox_{method}{FILE_EXTENSION}"
        bbox_image = draw_bbox(prepared_image, bboxes)
        cv2.imwrite(str(bbox_path), bbox_image)
        item_record.paths[f"bbox_{method}"] = relative_to_workspace(bbox_path)

        if source_image is not None:
            pipeline_path = leaf_dir / f"pipeline_{method}{FILE_EXTENSION}"
            pipeline_img = _build_pipeline_visualization(source_image, prepared_image, bbox_image)
            cv2.imwrite(str(pipeline_path), pipeline_img)
            item_record.paths[f"pipeline_{method}"] = relative_to_workspace(pipeline_path)

        item_record.segmentation[method] = _bboxes_to_schema(bboxes)
        results.append({"method": method, "status": "success", "count": len(bboxes)})

    unique_segments = list(dict.fromkeys(all_segments))
    item_record.paths["segments"] = unique_segments

    return results


def run_segmentation(
    item_records: list[DatasetItemRecord],
    *,
    methods: list[str] | None = None,
    limit: int | None = None,
) -> None:
    processed = 0
    for item_record in item_records:
        if limit is not None and processed >= limit:
            break
        segment_item(item_record, methods=methods)
        processed += 1

    for collection_name in SOURCE_COLLECTIONS:
        metadata_path = COLLECTION_METADATA_PATHS.get(collection_name)
        if metadata_path is None or not metadata_path.exists():
            continue
        with open(metadata_path, "r") as handle:
            existing = json.load(handle)
        for item_dict in existing:
            for record in item_records:
                if record.item_id == item_dict.get("item_id"):
                    item_dict["paths"] = record.paths
                    item_dict["segmentation"] = record.segmentation
                    break
        with open(metadata_path, "w") as handle:
            json.dump(existing, handle, indent=2)


def resolve_source_collection_names(selected: list[str] | None) -> list[str]:
    if not selected:
        return list(SOURCE_COLLECTIONS)
    return selected


def required_source_roots() -> list[Path]:
    return [CURATED_SOURCE_DATASET_PATH, INCOMING_SOURCE_DATASET_PATH]
