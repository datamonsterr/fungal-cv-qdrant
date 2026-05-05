from __future__ import annotations

import argparse
import json
from dataclasses import dataclass as _dc_autolab
from pathlib import Path
from typing import List as _List_autolab

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


@_dc_autolab
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@_dc_autolab
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: _List_autolab[str]
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    """Uniform experiment contract wrapper. Scoped to params.output_root."""
    import json as _json_autolab
    from pathlib import Path as _Path_autolab
    output_root = _Path_autolab(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = params.description[:30] if params.description else "yolo_dataset"
    result_data = {"f1_score": 0.0, "strategy_name": strategy, "artifact_paths": [], "run_id": params.run_id}
    (output_root / "results.json").write_text(_json_autolab.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
