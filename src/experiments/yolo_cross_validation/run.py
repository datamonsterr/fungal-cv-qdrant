from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DATASET_ROOT, STRAIN_SPECIES_MAPPING_PATH
from src.utils import default_output_root, materialize_strain_holdout_dataset
from src.utils.yolo_cross_validation import (
    build_strict_cv_folds,
    write_fold_summary_csv,
    write_metrics_csv,
)


def run_yolo_cross_validation(
    mapping_csv: Path = STRAIN_SPECIES_MAPPING_PATH,
    dataset_root: Path | None = None,
    n_folds: int = 5,
) -> dict[str, object]:
    source_dataset = dataset_root or default_output_root()
    folds = build_strict_cv_folds(mapping_csv, n_folds=n_folds)
    summary_csv = write_fold_summary_csv(folds)
    metrics_csv = write_metrics_csv(folds)
    fold_datasets: list[str] = []
    for fold_idx, fold in enumerate(folds):
        fold_root = DATASET_ROOT / "manual_labeled_data_roboflow_species_cv" / f"fold_{fold_idx}"
        materialize_strain_holdout_dataset(source_dataset, fold, fold_root)
        fold_datasets.append(str(fold_root))
    return {
        "fold_count": len(folds),
        "summary_csv": str(summary_csv),
        "metrics_csv": str(metrics_csv),
        "fold_datasets": fold_datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict YOLO cross validation setup"
    )
    parser.add_argument("--mapping-csv", type=Path, default=STRAIN_SPECIES_MAPPING_PATH)
    parser.add_argument("--dataset-root", type=Path, default=default_output_root())
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            run_yolo_cross_validation(
                args.mapping_csv,
                dataset_root=args.dataset_root,
                n_folds=args.folds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
