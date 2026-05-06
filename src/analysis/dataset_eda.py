from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import SOURCE_COLLECTIONS
from src.prepare.dataset import (
    load_source_collections,
    parse_source_metadata,
    resolve_source_collection_names,
    iter_source_images,
    load_strain_species_mapping,
)


@dataclass(frozen=True)
class DatasetCollectionSummary:
    collection_key: str
    display_name: str
    quality_tier: str
    source_root: str
    image_count: int
    species_count: int
    strain_count: int
    environment_count: int
    parse_status_counts: dict[str, int]
    species_image_counts: dict[str, int]
    strain_image_counts: dict[str, int]
    environment_image_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetEdaReport:
    collections: list[DatasetCollectionSummary]
    total_images: int
    total_species: int
    total_strains: int
    total_environments: int


def build_dataset_collection_summary(collection_key: str) -> DatasetCollectionSummary:
    collections = load_source_collections()
    collection = collections[collection_key]
    strain_species_mapping = load_strain_species_mapping()
    species_counts: Counter[str] = Counter()
    strain_counts: Counter[str] = Counter()
    environment_counts: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()

    for image_path in iter_source_images(collection):
        metadata = parse_source_metadata(image_path, strain_species_mapping)
        species_counts[metadata.species] += 1
        strain_counts[metadata.strain] += 1
        environment_counts[metadata.environment] += 1
        parse_status_counts[metadata.parse_status] += 1

    return DatasetCollectionSummary(
        collection_key=collection_key,
        display_name=collection.display_name,
        quality_tier=collection.quality_tier,
        source_root=str(collection.path),
        image_count=sum(species_counts.values()),
        species_count=len(species_counts),
        strain_count=len(strain_counts),
        environment_count=len(environment_counts),
        parse_status_counts=dict(parse_status_counts),
        species_image_counts=dict(sorted(species_counts.items())),
        strain_image_counts=dict(sorted(strain_counts.items())),
        environment_image_counts=dict(sorted(environment_counts.items())),
    )



def build_dataset_eda_report(
    source_collections: list[str] | None = None,
) -> DatasetEdaReport:
    collection_names = resolve_source_collection_names(source_collections)
    summaries = [build_dataset_collection_summary(name) for name in collection_names]

    species: set[str] = set()
    strains: set[str] = set()
    environments: set[str] = set()
    total_images = 0

    for summary in summaries:
        total_images += summary.image_count
        species.update(summary.species_image_counts)
        strains.update(summary.strain_image_counts)
        environments.update(summary.environment_image_counts)

    return DatasetEdaReport(
        collections=summaries,
        total_images=total_images,
        total_species=len(species),
        total_strains=len(strains),
        total_environments=len(environments),
    )



def render_text_report(report: DatasetEdaReport) -> str:
    lines = [
        f"total_images: {report.total_images}",
        f"total_species: {report.total_species}",
        f"total_strains: {report.total_strains}",
        f"total_environments: {report.total_environments}",
    ]
    for summary in report.collections:
        lines.extend(
            [
                "",
                f"[{summary.collection_key}] {summary.display_name}",
                f"quality_tier: {summary.quality_tier}",
                f"source_root: {summary.source_root}",
                f"image_count: {summary.image_count}",
                f"species_count: {summary.species_count}",
                f"strain_count: {summary.strain_count}",
                f"environment_count: {summary.environment_count}",
                f"parse_status_counts: {json.dumps(summary.parse_status_counts, sort_keys=True)}",
                f"species_image_counts: {json.dumps(summary.species_image_counts, sort_keys=True)}",
                f"environment_image_counts: {json.dumps(summary.environment_image_counts, sort_keys=True)}",
            ]
        )
    return "\n".join(lines)



def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze canonical source dataset collections")
    parser.add_argument(
        "--source-collection",
        action="append",
        default=[],
        dest="source_collections",
        help=f"Collection key to analyze (repeatable: {', '.join(SOURCE_COLLECTIONS)})",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path for report output",
    )
    args = parser.parse_args()

    report = build_dataset_eda_report(args.source_collections)
    payload = asdict(report)
    output = (
        json.dumps(payload, indent=2, sort_keys=True)
        if args.format == "json"
        else render_text_report(report)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
