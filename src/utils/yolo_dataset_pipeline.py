from __future__ import annotations

import csv
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import DATASET_ROOT, STRAIN_SPECIES_MAPPING_PATH

DTO_PATTERN = re.compile(r"(DTO[_\s]+\d+-[A-Z0-9]+)")


def parse_dto_strain_id(value: str) -> str | None:
    match = DTO_PATTERN.search(value)
    if not match:
        return None
    return match.group(1).replace(" ", "_")


def normalize_strain_id(value: str) -> str:
    parsed = parse_dto_strain_id(value) or value.strip()
    return re.sub(r"[_\s]+", " ", parsed).strip()


def load_strain_species_mapping(
    csv_path: Path = STRAIN_SPECIES_MAPPING_PATH,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            strain = normalize_strain_id(row.get("Strain", ""))
            species = (row.get("Species") or "").strip()
            if strain and species:
                mapping[strain] = species
    return mapping


def build_species_class_manifest(species_names: list[str]) -> dict[str, int]:
    unique_species = sorted(
        {species.strip() for species in species_names if species.strip()}
    )
    return {species: idx for idx, species in enumerate(unique_species)}


def summarize_species_counts(species_names: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(species_names).items()))


def default_output_root() -> Path:
    return DATASET_ROOT / "manual_labeled_data_roboflow_species"


def build_dataset_yaml(output_root: Path, manifest: dict[str, int]) -> str:
    lines = [
        f"path: {output_root.resolve()}",
        "train: train/images",
        "val: test/images",
        "names:",
    ]
    for species, class_id in sorted(manifest.items(), key=lambda item: item[1]):
        lines.append(f"  {class_id}: {species}")
    lines.append("")
    return "\n".join(lines)


def build_preparation_summary_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def rewrite_label_content(label_text: str, class_id: int) -> str:
    rewritten_lines: list[str] = []
    for raw_line in label_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        parts[0] = str(class_id)
        rewritten_lines.append(" ".join(parts))
    return "\n".join(rewritten_lines) + ("\n" if rewritten_lines else "")


def build_train_test_manifest(
    dataset_root: Path, test_ratio: float = 0.2, seed: int = 42
) -> dict[str, object]:
    image_files = sorted(dataset_root.rglob("*.jpg"))
    rng = random.Random(seed)
    shuffled = image_files[:]
    rng.shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio)) if shuffled else 0
    test_items = shuffled[:test_count]
    train_items = shuffled[test_count:]
    return {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "train": [str(path) for path in train_items],
        "test": [str(path) for path in test_items],
    }


def write_train_test_manifest(
    dataset_root: Path, test_ratio: float = 0.2, seed: int = 42
) -> Path:
    manifest = build_train_test_manifest(dataset_root, test_ratio=test_ratio, seed=seed)
    output_path = dataset_root / "train_test_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pd.Series(manifest).to_json(indent=2))
    return output_path


def prepare_species_labeled_dataset(
    source_root: Path, output_root: Path | None = None
) -> dict[str, object]:
    target_root = output_root or default_output_root()
    mapping = load_strain_species_mapping()
    rows: list[dict[str, object]] = []
    species_names: list[str] = []
    processed_records: list[tuple[Path, Path, str]] = []

    image_files = sorted(source_root.rglob("*.jpg"))
    for image_path in image_files:
        split_name = image_path.parent.parent.name
        label_path = source_root / split_name / "labels" / f"{image_path.stem}.txt"
        parsed = parse_dto_strain_id(image_path.name)
        normalized = normalize_strain_id(parsed or image_path.name)
        species = mapping.get(normalized)
        status = "processed"
        reason = ""

        if not parsed:
            status = "skipped"
            reason = "missing_dto_identifier"
        elif not species:
            status = "skipped"
            reason = "missing_species_mapping"
        elif not label_path.exists():
            status = "failed"
            reason = "missing_label_file"

        rows.append(
            {
                "split_name": split_name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "strain_id": normalized,
                "species_name": species or "",
                "status": status,
                "status_reason": reason,
            }
        )
        if status == "processed":
            species_names.append(species)
            processed_records.append((image_path, label_path, species))

    manifest = build_species_class_manifest(species_names)
    for image_path, label_path, species in processed_records:
        split_name = image_path.parent.parent.name
        target_image = target_root / split_name / "images" / image_path.name
        target_label = target_root / split_name / "labels" / label_path.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        target_label.write_text(
            rewrite_label_content(label_path.read_text(), manifest[species])
        )

    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "dataset.yaml").write_text(build_dataset_yaml(target_root, manifest))
    (target_root / "classes.txt").write_text(
        "\n".join(
            species for species, _ in sorted(manifest.items(), key=lambda item: item[1])
        )
        + "\n"
        if manifest
        else ""
    )
    summary_df = build_preparation_summary_rows(rows)
    summary_path = target_root / "preparation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {
        "output_root": str(target_root),
        "summary_path": str(summary_path),
        "processed_count": (
            int((summary_df["status"] == "processed").sum())
            if not summary_df.empty
            else 0
        ),
        "skipped_count": (
            int((summary_df["status"] == "skipped").sum())
            if not summary_df.empty
            else 0
        ),
        "failed_count": (
            int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
        ),
        "class_count": len(manifest),
    }


def materialize_strain_holdout_dataset(
    dataset_root: Path,
    fold_selection: dict[str, str],
    output_root: Path,
) -> dict[str, object]:
    classes_path = dataset_root / "classes.txt"
    dataset_yaml = dataset_root / "dataset.yaml"
    if classes_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_path, output_root / "classes.txt")
    if dataset_yaml.exists():
        shutil.copy2(dataset_yaml, output_root / "dataset.yaml")

    rows: list[dict[str, object]] = []
    for image_path in sorted(dataset_root.rglob("*.jpg")):
        if image_path.parent.name != "images":
            continue
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        parsed = parse_dto_strain_id(image_path.name)
        strain_id = normalize_strain_id(parsed or image_path.name)
        species = fold_selection.get("_species_lookup", {}).get(strain_id, "")
        selected_test_strain = fold_selection.get(species)
        split_name = "test" if selected_test_strain == strain_id else "train"
        target_image = output_root / split_name / "images" / image_path.name
        target_label = output_root / split_name / "labels" / label_path.name
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        if label_path.exists():
            shutil.copy2(label_path, target_label)
        rows.append(
            {
                "image_path": str(image_path),
                "strain_id": strain_id,
                "species_name": species,
                "assigned_split": split_name,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = output_root / "fold_assignment.csv"
    summary_df.to_csv(summary_path, index=False)
    return {
        "output_root": str(output_root),
        "summary_path": str(summary_path),
        "train_count": int((summary_df["assigned_split"] == "train").sum()) if not summary_df.empty else 0,
        "test_count": int((summary_df["assigned_split"] == "test").sum()) if not summary_df.empty else 0,
    }
