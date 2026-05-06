from __future__ import annotations

import argparse
import json
from dataclasses import dataclass as _dc_autolab
from pathlib import Path
from typing import List as _List_autolab

from src.config import RESULTS_DIR, WEIGHTS_DIR
from src.utils.yolo_dataset_pipeline import (
    default_output_root,
    write_train_test_manifest,
)


def run_yolo_segmentation(
    dataset_root: Path | None = None,
    train: bool = True,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 10,
) -> dict[str, object]:
    root = dataset_root or default_output_root()
    yaml_path = root / "dataset.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run the dataset preparation step first."
        )

    results_root = RESULTS_DIR / "cross_validation_yolo" / "segmentation"
    results_root.mkdir(parents=True, exist_ok=True)
    weights_dir = WEIGHTS_DIR / "segmentation"
    weights_dir.mkdir(parents=True, exist_ok=True)

    weights_path = results_root / "yolo_segmentation" / "weights" / "best.pt"
    best_weights_dest = weights_dir / "yolo_segmentation_best.pt"

    if train:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLO training. "
                "Install with: uv add ultralytics"
            ) from e

        model = YOLO(
            "yolov8n-seg.pt" if model_size == "n" else f"yolov8{model_size}-seg.pt"
        )

        train_results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=str(results_root),
            name="yolo_segmentation",
            exist_ok=True,
            verbose=True,
        )

        if weights_path.exists():
            import shutil

            best_weights_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_path, best_weights_dest)

        metrics = {
            "status": "trained",
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "has_validation_split": False,
            "train_results_dir": str(results_root / "yolo_segmentation"),
            "best_weights": (
                str(best_weights_dest)
                if best_weights_dest.exists()
                else str(weights_path)
            ),
            "final_metrics": (
                train_results.results_dict
                if hasattr(train_results, "results_dict")
                else {}
            ),
        }
    else:
        metrics = {
            "status": "manifest_ready",
            "model_size": model_size,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "has_validation_split": False,
            "train_results_dir": str(results_root / "yolo_segmentation"),
            "best_weights": str(best_weights_dest),
            "final_metrics": {},
        }

    metrics_path = results_root / "segmentation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest_path = write_train_test_manifest(root)
    manifest = json.loads(manifest_path.read_text())
    metadata = {
        "dataset_root": str(root),
        "manifest_path": str(manifest_path),
        "yaml_path": str(yaml_path),
        "split_seed": manifest["seed"],
        "train_count": len(manifest["train"]),
        "test_count": len(manifest["test"]),
        "has_validation_split": False,
    }
    metadata_path = results_root / "segmentation_split_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "dataset_root": str(root),
        "manifest_path": str(manifest_path),
        "yaml_path": str(yaml_path),
        "split_seed": manifest["seed"],
        "train_count": len(manifest["train"]),
        "test_count": len(manifest["test"]),
        "metrics_path": str(metrics_path),
        "metadata_path": str(metadata_path),
        "best_weights": (
            str(best_weights_dest) if best_weights_dest.exists() else str(weights_path)
        ),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model")
    parser.add_argument("--dataset-root", type=Path, default=default_output_root())
    parser.add_argument(
        "--model-size",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO model size (n= nano, s= small, m= medium, l= large, x= x-large)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    result = run_yolo_segmentation(
        dataset_root=args.dataset_root,
        train=True,
        model_size=args.model_size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
    )
    print(json.dumps(result, indent=2))


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
    strategy = params.description[:30] if params.description else "yolo_segmentation"
    result_data = {"f1_score": 0.0, "strategy_name": strategy, "artifact_paths": [], "run_id": params.run_id}
    (output_root / "results.json").write_text(_json_autolab.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
