import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import cv2
import pandas as pd  # type: ignore[import-untyped]

from src.config import (
    DATASET_ROOT,
    FULL_IMAGE_METADATA_PATH,
    FULL_IMAGE_PATH,
    ORIGINAL_DATASET_PATH,
    SEGMENTED_IMAGE_DIR,
    SEGMENTED_METADATA_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
    relative_to_workspace,
)
from src.preprocessing.kmeans import segment_kmeans_image
from src.preprocessing.preprocess import prepare_image

FILE_EXTENSION = ".jpg"
HIERARCHICAL_DATASET_PATH = DATASET_ROOT / "hierarchical"
ALL_DATASET_PATH = DATASET_ROOT / "all"
ALL_IMAGES_PATH = ALL_DATASET_PATH / "images"
NEW_DATA_PATH = DATASET_ROOT / "new_data"
ALL_METADATA_PATH = ALL_DATASET_PATH / "all_metadata.json"
PROCESSING_PREVIEW_SIZE = 512
SEGMENT_SIZE = 256
KNOWN_ENVS = {
    "CREA",
    "CYA",
    "CYAS",
    "MEA",
    "DG18",
    "YES",
    "OA",
    "MALT",
    "STEFF",
    "Sabouraud",
}


def get_specy_from_strain(strain: str, strain_to_specy: pd.DataFrame) -> str | None:
    result = strain_to_specy[strain_to_specy["Strain"] == strain]
    if not result.empty:
        return result["Species"].iloc[0]
    return None


class Metadata:
    def __init__(self, filename: str, id: str, strain_to_specy: pd.DataFrame):
        self.filename = filename
        self._clean_filename()
        self.id = id
        match = re.match(r"(DTO\s[0-9]+-[A-Z0-9]+)\s([A-Z0-9]+)(rev|ob)", self.filename)

        self.strain = "unknown"
        self.environment = "unknown"
        self.angle = "unknown"

        if match:
            self.strain = match.group(1)
            self.environment = match.group(2)
            self.angle = match.group(3)

        self.specy = get_specy_from_strain(self.strain, strain_to_specy) or "unknown"

    def get_metadata(self) -> dict[str, dict[str, str] | str]:
        return {
            "id": self.id,
            "data": {
                "strain": self.strain,
                "environment": self.environment,
                "angle": self.angle,
                "specy": self.specy,
            },
        }

    def _clean_filename(self):
        self.filename = self.filename.removesuffix(FILE_EXTENSION)
        self.filename = self.filename.removesuffix("_edited")


def normalize_angle(angle_raw: str) -> str:
    angle = angle_raw.lower().strip()
    if angle in ("ob", "o", "a"):
        return "ob"
    if angle in ("rev", "r", "b"):
        return "rev"
    return angle or "unknown"


def _clean_new_data_name(filename: str) -> str:
    name = Path(filename).stem
    name = " ".join(name.split())
    return re.sub(
        r"\s*(edited|detail\s*colony|Auto)\s*$", "", name, flags=re.IGNORECASE
    )


def _extract_env_angle_from_suffix(value: str) -> tuple[str | None, str | None]:
    upper = value.upper()
    for env in KNOWN_ENVS:
        if not upper.startswith(env.upper()):
            continue
        rest = upper[len(env) :]
        if rest in ("O", "OB", "A", ""):
            return env, "ob"
        if rest in ("R", "REV", "B"):
            return env, "rev"
    return None, None


def _parse_suffix_tokens(name: str) -> dict[str, str] | None:
    parts = name.split()
    if len(parts) >= 2:
        env, angle = _extract_env_angle_from_suffix(parts[-1])
        if env is not None and angle is not None:
            return {
                "strain": " ".join(parts[:-1]),
                "environment": env,
                "angle": angle,
            }

    if len(parts) >= 3 and parts[-2].upper() in KNOWN_ENVS:
        return {
            "strain": " ".join(parts[:-2]),
            "environment": parts[-2].upper(),
            "angle": normalize_angle(parts[-1]),
        }
    return None


def _parse_known_strain_patterns(name: str) -> dict[str, str] | None:
    strain_pattern = r"^(T\(N\)|T\d+|IBT\s+\d+|CBS\s+\d+[._]\d+)"
    for env in KNOWN_ENVS:
        patterns = (
            rf"^\S+\s+(CBS\s+\d+[_\.]\d+)\s+{env}(o|r|ob|rev|b)?$",
            rf"^\S+\s+(IBT\s+\d+)\s+{env}(o|r|ob|rev|b)?$",
            rf"^{strain_pattern}\s+{env}\s+(ob|rev|o|r|a|b)$",
        )
        for pattern in patterns:
            match = re.match(pattern, name, re.IGNORECASE)
            if not match:
                continue
            return {
                "strain": match.group(1),
                "environment": env,
                "angle": normalize_angle(match.group(2) or ""),
            }
    return None


def _parse_env_embedded_name(name: str) -> dict[str, str] | None:
    lower_name = name.lower()
    for env in KNOWN_ENVS:
        if env not in name.upper():
            continue
        angle = "ob"
        if " rev " in lower_name or lower_name.endswith(" rev"):
            angle = "rev"
        idx = name.upper().find(env)
        strain = name[:idx].strip()
        if strain:
            return {"strain": strain, "environment": env, "angle": angle}
    return None


def parse_new_data_filename(filename: str) -> dict[str, str]:
    name = _clean_new_data_name(filename)
    return (
        _parse_suffix_tokens(name)
        or _parse_known_strain_patterns(name)
        or _parse_env_embedded_name(name)
        or {"strain": name, "environment": "unknown", "angle": "unknown"}
    )


def _is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _build_new_data_entry(
    *,
    image_path: Path,
    species_name: str,
    fallback_strain: str | None = None,
) -> dict[str, str]:
    parsed = parse_new_data_filename(image_path.name)
    strain = parsed["strain"]
    if strain == "unknown" and fallback_strain is not None:
        strain = fallback_strain
    return {
        "source_path": str(image_path),
        "species": species_name,
        "strain": strain,
        "environment": parsed["environment"],
        "angle": parsed["angle"],
        "filename": image_path.name,
        "source": "new_data",
    }


def _iter_species_items(species_dir: Path) -> list[dict[str, str]]:
    species_name = species_dir.name
    images: list[dict[str, str]] = []
    for item in sorted(species_dir.iterdir()):
        if _is_supported_image(item):
            images.append(
                _build_new_data_entry(image_path=item, species_name=species_name)
            )
            continue
        if not item.is_dir():
            continue
        for img_file in sorted(item.iterdir()):
            if _is_supported_image(img_file):
                images.append(
                    _build_new_data_entry(
                        image_path=img_file,
                        species_name=species_name,
                        fallback_strain=item.name,
                    )
                )
    return images


def iter_new_data_images() -> list[dict[str, str]]:
    if not NEW_DATA_PATH.exists():
        return []

    images: list[dict[str, str]] = []
    for letter_dir in sorted(NEW_DATA_PATH.iterdir()):
        if not letter_dir.is_dir():
            continue
        for species_dir in sorted(letter_dir.iterdir()):
            if species_dir.is_dir():
                images.extend(_iter_species_items(species_dir))
    return images


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def _all_image_dir(species: str, strain: str, environment: str, source: str) -> Path:
    return (
        ALL_IMAGES_PATH
        / _safe_name(source)
        / _safe_name(species)
        / _safe_name(strain)
        / _safe_name(environment)
    )


def _save_hierarchical_segment(
    metadata: Metadata, segment_img, segment_index: int
) -> str:
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}"
        f"_seg{segment_index}{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    cv2.imwrite(str(hierarchical_file_path), segment_img)
    return relative_to_workspace(hierarchical_file_path)


def _save_hierarchical_original(
    metadata: Metadata, image_id: str, full_img_path: Path
) -> str:
    clean_strain = metadata.strain.replace(" ", "_").replace("/", "-")
    hierarchical_filename = (
        f"{clean_strain}_{metadata.environment}_{metadata.angle}_{image_id}_original"
        f"{FILE_EXTENSION}"
    )
    hierarchical_dir = (
        HIERARCHICAL_DATASET_PATH
        / metadata.specy
        / metadata.strain
        / metadata.environment
    )
    hierarchical_dir.mkdir(parents=True, exist_ok=True)
    hierarchical_file_path = hierarchical_dir / hierarchical_filename
    shutil.copyfile(full_img_path, hierarchical_file_path)
    return relative_to_workspace(hierarchical_file_path)


def _save_image(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return relative_to_workspace(path)


def _copy_image(source_path: Path, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, target_path)
    return relative_to_workspace(target_path)


def _build_kmeans_pipeline_visual(debug_images: dict[str, Any]) -> Any:
    color_dimension = cast(Any, debug_images["color_dimension"])
    foreground_mask = cast(Any, debug_images["foreground_mask"])
    location_clusters = cast(Any, debug_images["location_clusters"])
    bbox_image = cast(Any, debug_images["bbox_image"])
    top = cv2.hconcat([color_dimension, foreground_mask])
    bottom = cv2.hconcat([location_clusters, bbox_image])
    return cv2.vconcat([top, bottom])


def _export_all_variants(
    *,
    source_path: Path,
    species: str,
    strain: str,
    environment: str,
    angle: str,
    image_id: str,
    source_name: str,
) -> dict[str, Any] | None:
    source_image = cv2.imread(str(source_path))
    if source_image is None:
        print(f"Failed to read {source_path}")
        return None

    artifacts = prepare_image(source_image, export_size=SEGMENT_SIZE)
    preprocessed = artifacts.masked_export_image
    bboxes, score, debug_images = segment_kmeans_image(
        preprocessed,
        plate_mask=artifacts.export_mask,
        return_debug=True,
    )

    target_dir = _all_image_dir(species, strain, environment, source_name)
    base_name = f"{_safe_name(strain)}_{_safe_name(angle)}_{image_id[:8]}"

    centered_preview = cv2.resize(
        artifacts.square_image,
        (PROCESSING_PREVIEW_SIZE, PROCESSING_PREVIEW_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    pipeline_visual = _build_kmeans_pipeline_visual(debug_images)

    step_images = {
        "original": _copy_image(
            source_path, target_dir / f"{base_name}_original{FILE_EXTENSION}"
        ),
        "centered": _save_image(
            target_dir / f"{base_name}_centered{FILE_EXTENSION}", centered_preview
        ),
        "preprocessed": _save_image(
            target_dir / f"{base_name}_processed{FILE_EXTENSION}", preprocessed
        ),
        "bboxes": _save_image(
            target_dir / f"{base_name}_bboxes{FILE_EXTENSION}",
            debug_images["bbox_image"],
        ),
        "kmeans_pipeline": _save_image(
            target_dir / f"{base_name}_pipeline{FILE_EXTENSION}", pipeline_visual
        ),
    }

    segment_paths: list[str] = []
    for idx, bbox in enumerate(bboxes):
        segment = preprocessed[bbox["ymin"] : bbox["ymax"], bbox["xmin"] : bbox["xmax"]]
        if segment.size == 0:
            continue
        resized_segment = cv2.resize(segment, (SEGMENT_SIZE, SEGMENT_SIZE))
        segment_paths.append(
            _save_image(
                target_dir / f"{base_name}_seg{idx}{FILE_EXTENSION}", resized_segment
            )
        )

    return {
        "id": image_id,
        "source": source_name,
        "source_path": relative_to_workspace(source_path),
        "file_path": step_images["preprocessed"],
        "step_images": step_images,
        "data": {
            "species": species,
            "strain": strain,
            "environment": environment,
            "angle": angle,
            "bboxes": bboxes,
            "num_colonies": len(segment_paths),
            "segment_paths": segment_paths,
            "bbox_quality": score,
        },
    }


def _load_strain_mapping() -> pd.DataFrame:
    if STRAIN_SPECIES_MAPPING_PATH.exists():
        return pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    print(f"Warning: {STRAIN_SPECIES_MAPPING_PATH} not found.")
    return pd.DataFrame(columns=["Strain", "Species"])


def _ensure_output_dirs() -> None:
    FULL_IMAGE_PATH.mkdir(parents=True, exist_ok=True)
    SEGMENTED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ALL_IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    if not FULL_IMAGE_METADATA_PATH.exists():
        FULL_IMAGE_METADATA_PATH.write_text("[]")
    if not SEGMENTED_METADATA_PATH.exists():
        SEGMENTED_METADATA_PATH.write_text("[]")


def _append_original_segments(
    *,
    export: dict[str, Any],
    metadata: Metadata,
    image_id: str,
    segment_metadata_list: list[dict[str, Any]],
    create_hierarchical: bool,
) -> None:
    export_data = cast(dict[str, Any], export["data"])
    segment_paths = cast(list[str], export_data["segment_paths"])
    for index, segment_path in enumerate(segment_paths):
        segment_id = f"{image_id}_{index}"
        seg_metadata = metadata.get_metadata()
        seg_metadata["id"] = segment_id
        seg_metadata["parent_id"] = image_id
        flat_segment_path = SEGMENTED_IMAGE_DIR / f"{segment_id}{FILE_EXTENSION}"
        shutil.copyfile(Path(DATASET_ROOT.parent) / segment_path, flat_segment_path)
        seg_metadata["file_path"] = relative_to_workspace(flat_segment_path)
        if create_hierarchical:
            segment_img = cv2.imread(str(flat_segment_path))
            if segment_img is not None:
                seg_metadata["hierarchical_path"] = _save_hierarchical_segment(
                    metadata=metadata,
                    segment_img=segment_img,
                    segment_index=index,
                )
        segment_metadata_list.append(seg_metadata)


def _process_original_dataset(
    *,
    strain_to_specy: pd.DataFrame,
    metadata_list: list[dict[str, Any]],
    segment_metadata_list: list[dict[str, Any]],
    all_metadata: list[dict[str, Any]],
    create_hierarchical: bool,
) -> None:
    for dir_path in sorted(
        path for path in ORIGINAL_DATASET_PATH.iterdir() if path.is_dir()
    ):
        for image_path in sorted(
            path for path in dir_path.iterdir() if path.suffix.lower() == FILE_EXTENSION
        ):
            image_id = uuid.uuid5(uuid.NAMESPACE_DNS, image_path.name).hex
            metadata = Metadata(image_path.name, image_id, strain_to_specy)
            full_img_path = FULL_IMAGE_PATH / f"{image_id}{FILE_EXTENSION}"
            shutil.copyfile(image_path, full_img_path)

            full_metadata = metadata.get_metadata()
            full_metadata["file_path"] = relative_to_workspace(full_img_path)
            if create_hierarchical:
                full_metadata["hierarchical_path"] = _save_hierarchical_original(
                    metadata=metadata,
                    image_id=image_id,
                    full_img_path=full_img_path,
                )
            metadata_list.append(full_metadata)

            print(f"Processing original/{image_path.name}...")
            export = _export_all_variants(
                source_path=image_path,
                species=metadata.specy,
                strain=metadata.strain,
                environment=metadata.environment,
                angle=metadata.angle,
                image_id=image_id,
                source_name="original",
            )
            if export is None:
                continue
            all_metadata.append(export)
            _append_original_segments(
                export=export,
                metadata=metadata,
                image_id=image_id,
                segment_metadata_list=segment_metadata_list,
                create_hierarchical=create_hierarchical,
            )


def _process_new_data_dataset(all_metadata: list[dict[str, Any]]) -> None:
    for item in iter_new_data_images():
        source_path = Path(item["source_path"])
        image_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(source_path)).hex
        print(f"Processing new_data/{item['filename']}...")
        export = _export_all_variants(
            source_path=source_path,
            species=item["species"],
            strain=item["strain"],
            environment=item["environment"],
            angle=item["angle"],
            image_id=image_id,
            source_name="new_data",
        )
        if export is not None:
            all_metadata.append(export)


def reformat_dataset(create_hierarchical: bool = True):
    strain_to_specy = _load_strain_mapping()
    _ensure_output_dirs()

    metadata_list: list[dict[str, Any]] = []
    segment_metadata_list: list[dict[str, Any]] = []
    all_metadata: list[dict[str, Any]] = []

    if not ORIGINAL_DATASET_PATH.exists():
        print(f"Error: {ORIGINAL_DATASET_PATH} does not exist.")
        return

    _process_original_dataset(
        strain_to_specy=strain_to_specy,
        metadata_list=metadata_list,
        segment_metadata_list=segment_metadata_list,
        all_metadata=all_metadata,
        create_hierarchical=create_hierarchical,
    )
    _process_new_data_dataset(all_metadata)

    with open(FULL_IMAGE_METADATA_PATH, "w") as f:
        json.dump(metadata_list, f, indent=4)

    with open(SEGMENTED_METADATA_PATH, "w") as f:
        json.dump(segment_metadata_list, f, indent=4)

    with open(ALL_METADATA_PATH, "w") as f:
        json.dump({"images": all_metadata}, f, indent=2)

    print("\nProcessing complete!")
    print(f"Full images: {len(metadata_list)}")
    print(f"Segmented images: {len(segment_metadata_list)}")
    print(f"All dataset images: {len(all_metadata)}")


if __name__ == "__main__":
    reformat_dataset()
