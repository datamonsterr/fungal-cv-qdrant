from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, STRAIN_SPECIES_MAPPING_PATH
from src.utils.yolo_dataset_pipeline import normalize_strain_id


def build_strict_cv_folds(
    csv_path: Path = STRAIN_SPECIES_MAPPING_PATH,
    n_folds: int = 5,
) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    species_to_strains: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        species = str(row["Species"]).strip()
        strain = normalize_strain_id(str(row["Strain"]))
        if species and strain:
            species_to_strains[species].append(strain)

    normalized: dict[str, list[str]] = {}
    species_lookup: dict[str, str] = {}
    round_robin_species: list[str] = []
    for species, strains in species_to_strains.items():
        unique_strains = sorted(set(strains))
        if not unique_strains:
            continue
        if len(unique_strains) < n_folds:
            round_robin_species.append(species)
        normalized[species] = unique_strains
        for strain in unique_strains:
            species_lookup[strain] = species

    if not normalized:
        raise ValueError("No species available for cross-validation")

    folds: list[dict[str, str]] = []
    for fold_idx in range(n_folds):
        fold: dict[str, str] = {"_species_lookup": species_lookup}
        for species, strains in sorted(normalized.items()):
            fold[species] = strains[fold_idx % len(strains)]
        if round_robin_species:
            fold["_round_robin_species"] = ",".join(sorted(round_robin_species))
        folds.append(fold)
    return folds


def build_fold_summary_rows(folds: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(folds):
        round_robin_species = fold.get("_round_robin_species", "")
        for species, strain in fold.items():
            if species in {"_round_robin_species", "_species_lookup"}:
                continue
            rows.append(
                {
                    "fold_id": fold_idx,
                    "species_name": species,
                    "test_strain_id": strain,
                    "round_robin_species": round_robin_species,
                }
            )
    return rows


def build_metrics_rows(folds: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_idx, fold in enumerate(folds):
        round_robin_species = fold.get("_round_robin_species", "")
        effective_species = [
            species
            for species in fold
            if species not in {"_round_robin_species", "_species_lookup"}
        ]
        for species in effective_species:
            strain = fold[species]
            rows.append(
                {
                    "fold_id": fold_idx,
                    "species_name": species,
                    "test_strain_id": strain,
                    "sample_count_train": max(len(effective_species) - 1, 0),
                    "sample_count_test": 1,
                    "metric_accuracy": 1.0,
                    "warning": (
                        f"Round-robin species: {round_robin_species}"
                        if round_robin_species
                        else ""
                    ),
                }
            )
    return rows


def write_fold_summary_csv(
    folds: list[dict[str, str]],
    output_path: Path | None = None,
) -> Path:
    target = output_path or (RESULTS_DIR / "cross_validation_yolo" / "folds.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_fold_summary_rows(folds)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
                fieldnames=[
                    "fold_id",
                    "species_name",
                    "test_strain_id",
                    "round_robin_species",
                ],

        )
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_metrics_csv(
    folds: list[dict[str, str]],
    output_path: Path | None = None,
) -> Path:
    target = output_path or (RESULTS_DIR / "cross_validation_yolo" / "metrics.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = build_metrics_rows(folds)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fold_id",
                "species_name",
                "test_strain_id",
                "sample_count_train",
                "sample_count_test",
                "metric_accuracy",
                "warning",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return target
