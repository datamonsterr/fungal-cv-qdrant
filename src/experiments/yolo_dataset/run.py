from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DATASET_ROOT
from src.utils.yolo_dataset_pipeline import (
    default_output_root,
    prepare_species_labeled_dataset,
)


def run_dataset_preparation(
    source_root: Path, output_root: Path | None = None
) -> dict[str, object]:
    return prepare_species_labeled_dataset(source_root, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a species-labeled YOLO dataset"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATASET_ROOT / "manual_labeled_data_roboflow",
        help="Source Roboflow dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_root(),
        help="Output dataset root",
    )
    args = parser.parse_args()
    print(json.dumps(run_dataset_preparation(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
