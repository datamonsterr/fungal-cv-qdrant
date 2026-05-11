"""YOLOv26 segmentation finetune: training, inference, and experiment contract."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import (
    YOLO_DATASET_DIR,
    YOLO_WEIGHTS_DIR,
    YOLO_RESULTS_DIR,
    PREPARED_DATASET_DIR,
)
from src.preprocessing.kmeans import draw_bbox, segment_kmeans_image


# ---------------------------------------------------------------------------
# T019 — FinetuneConfig
# ---------------------------------------------------------------------------

VALID_VARIANTS = ("n", "s", "m", "l", "x")
VRAM_ESTIMATES: dict[str, float] = {
    "n": 3.0,
    "s": 5.0,
    "m": 8.0,
    "l": 10.0,
    "x": 16.0,
}


@dataclass
class FinetuneConfig:
    model_variant: str = "n"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    train_split: float = 0.8
    pretrained: bool = True
    patience: int = 20

    def __post_init__(self) -> None:
        if self.model_variant not in VALID_VARIANTS:
            raise ValueError(
                f"model_variant must be one of {VALID_VARIANTS}, got {self.model_variant}"
            )

    @property
    def model_name(self) -> str:
        return f"yolo26{self.model_variant}-seg"


# ---------------------------------------------------------------------------
# T020 — TrainingResult
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    model_variant: str = ""
    best_map50: float = 0.0
    best_map50_95: float = 0.0
    epochs_completed: int = 0
    weights_path: str = ""
    last_weights_path: str = ""
    results_csv: str = ""
    status: str = "unknown"


# ---------------------------------------------------------------------------
# T023 — GPU memory guard
# ---------------------------------------------------------------------------

def estimate_vram(model_variant: str, imgsz: int = 640, batch: int = 8) -> float:
    base = VRAM_ESTIMATES.get(model_variant, 16.0)
    scale = (imgsz / 640) ** 2 * (batch / 8)
    return base * scale


def check_vram_fit(
    model_variant: str,
    available_vram_gb: float = 6.0,
    imgsz: int = 640,
    batch: int = 8,
) -> bool:
    return estimate_vram(model_variant, imgsz, batch) <= available_vram_gb * 0.85


# ---------------------------------------------------------------------------
# T021 — train_yolo26
# ---------------------------------------------------------------------------

def run_yolo26_train(
    model_variant: str = "n",
    epochs: int = 30,
    batch: int = 8,
    imgsz: int = 640,
    device: str = "0",
    data_root: Path | None = None,
    workers: int = 8,
    resume: bool = False,
) -> dict[str, Any]:
    cfg = FinetuneConfig(
        model_variant=model_variant,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
    )

    if not check_vram_fit(cfg.model_variant):
        print(
            f"WARNING: {cfg.model_name} may OOM on available GPU. "
            f"Estimate: {estimate_vram(cfg.model_variant):.1f}GB. "
            "Consider using a smaller variant or reducing batch size."
        )

    dataset_dir = data_root or YOLO_DATASET_DIR
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    from src.experiments.yolo_segmentation.prepare import prepare_roboflow_dataset_yaml

    train_count, val_count = prepare_roboflow_dataset_yaml(dataset_dir)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is required for YOLO training. Install with: uv add ultralytics"
        ) from e

    YOLO_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        best_pt = YOLO_WEIGHTS_DIR / f"{cfg.model_name}_species_best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"No checkpoint to resume at {best_pt}")
        model = YOLO(str(best_pt))
    else:
        model = YOLO(f"{cfg.model_name}.pt")

    start = time.time()
    train_results = model.train(
        data=str(yaml_path),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        patience=cfg.patience,
        device=cfg.device,
        workers=workers,
        resume=resume,
        project=str(YOLO_RESULTS_DIR),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    elapsed = time.time() - start
    hours = elapsed / 3600

    trained_dir = YOLO_RESULTS_DIR / "train" / "weights"
    best_src = trained_dir / "best.pt"
    last_src = trained_dir / "last.pt"

    best_dest = YOLO_WEIGHTS_DIR / f"{cfg.model_name}_species_best.pt"
    last_dest = YOLO_WEIGHTS_DIR / f"{cfg.model_name}_species_last.pt"

    if best_src.exists():
        shutil.copy2(best_src, best_dest)
    if last_src.exists():
        shutil.copy2(last_src, last_dest)

    results_csv = YOLO_RESULTS_DIR / "train" / "results.csv"

    metrics: dict[str, Any] = {}
    if hasattr(train_results, "results_dict"):
        metrics = train_results.results_dict

    return {
        "model_variant": cfg.model_variant,
        "model_name": cfg.model_name,
        "epochs_requested": cfg.epochs,
        "best_map50": metrics.get("metrics/mAP50(B)", 0.0),
        "best_map50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "weights_path": str(best_dest) if best_dest.exists() else "",
        "last_weights_path": str(last_dest) if last_dest.exists() else "",
        "results_csv": str(results_csv) if results_csv.exists() else "",
        "train_images": train_count,
        "val_images": val_count,
        "hours": round(hours, 2),
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# T029 — YOLOv26 inference helpers
# ---------------------------------------------------------------------------

def norm_xywh_to_pixel(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> dict[str, int]:
    x = int((cx - w / 2) * img_w)
    y = int((cy - h / 2) * img_h)
    pw = int(w * img_w)
    ph = int(h * img_h)
    return {"x": max(0, x), "y": max(0, y), "w": pw, "h": ph}


def _yolo26_detect(
    model: Any,
    image: np.ndarray,
    confidence: float = 0.25,
) -> list[dict[str, Any]]:
    results = model(image, conf=confidence, verbose=False)
    bboxes: list[dict[str, Any]] = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            h, w = image.shape[:2]
            for box in r.boxes:
                cx, cy, bw, bh = box.xywhn[0].tolist()
                conf = float(box.conf[0])
                pixel: dict[str, Any] = norm_xywh_to_pixel(cx, cy, bw, bh, w, h)
                pixel["confidence"] = round(conf, 4)
                bboxes.append(pixel)
    bboxes.sort(key=lambda b: b["confidence"], reverse=True)
    return bboxes[:3]


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


# ---------------------------------------------------------------------------
# T030 — save inference artifacts
# ---------------------------------------------------------------------------

def save_inference_artifacts(
    leaf_dir: Path,
    source_image: np.ndarray | None,
    prepared_image: np.ndarray,
    yolo26_bboxes: list[dict[str, Any]],
    kmeans_bboxes: list[dict[str, Any]],
) -> dict[str, Any]:
    leaf_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = leaf_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "yolo26": [_bbox_to_schema(b) for b in yolo26_bboxes],
        "kmeans": [_bbox_to_schema(b) for b in kmeans_bboxes],
    }

    img_h, img_w = prepared_image.shape[:2]

    # YOLOv26 visualization
    if yolo26_bboxes:
        yolo_img = draw_bbox(prepared_image, _to_xyxy(yolo26_bboxes))
        cv2.imwrite(str(leaf_dir / "bbox_yolo26.jpg"), yolo_img)

        if source_image is not None:
            pipeline_img = _build_pipeline(source_image, prepared_image, yolo_img)
            cv2.imwrite(str(leaf_dir / "pipeline_yolo26.jpg"), pipeline_img)

        for i, bbox in enumerate(yolo26_bboxes[:3], 1):
            crop = _crop_bbox(prepared_image, bbox)
            if crop is not None and crop.size > 0:
                cv2.imwrite(str(segments_dir / f"segment_yolo26_{i}.jpg"), crop)

    # Metadata JSON
    metadata_path = leaf_dir / "metadata.json"
    metadata_path.write_text(json.dumps(result, indent=2))

    return result


def _bbox_to_schema(bbox: dict[str, Any]) -> dict[str, int]:
    return {"x": int(bbox["x"]), "y": int(bbox["y"]), "w": int(bbox["w"]), "h": int(bbox["h"])}


def _to_xyxy(bboxes: list[dict[str, Any]]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for b in bboxes:
        out.append({
            "xmin": int(b["x"]),
            "ymin": int(b["y"]),
            "xmax": int(b["x"]) + int(b["w"]),
            "ymax": int(b["y"]) + int(b["h"]),
        })
    return out


def _build_pipeline(
    source: np.ndarray,
    prepared: np.ndarray,
    bbox_img: np.ndarray,
) -> np.ndarray:
    h, w = prepared.shape[:2]
    src = cv2.resize(source, (w, h), interpolation=cv2.INTER_AREA)
    return np.hstack([src, prepared, bbox_img])


def _crop_bbox(
    image: np.ndarray,
    bbox: dict[str, Any],
) -> np.ndarray | None:
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(image.shape[1], x1 + int(bbox["w"]))
    y2 = min(image.shape[0], y1 + int(bbox["h"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# T031 — run_yolo26_inference
# ---------------------------------------------------------------------------

def run_yolo26_inference(
    weights_path: str,
    data_root: Path | None = None,
    limit: int | None = None,
    confidence: float = 0.25,
    device: str = "0",
) -> dict[str, Any]:
    from ultralytics import YOLO
    model = YOLO(weights_path)
    try:
        model.to(device if ":" in device or device.lower() == "cpu" else f"cuda:{device}")
    except Exception:
        model.to("cuda")

    root = Path(data_root) if data_root else PREPARED_DATASET_DIR
    img_exts = {".jpg", ".jpeg", ".png"}

    image_paths = sorted(
        img_path
        for img_path in root.rglob("prepared.jpg")
        if img_path.is_file() and img_path.suffix.lower() in img_exts
    )  # type: list[Path]
    total = len(image_paths)
    image_paths = image_paths[:limit] if limit else image_paths

    processed = 0
    skipped = 0
    failed = 0

    for img_path in image_paths:
        leaf_dir = img_path.parent
        prepared_img = cv2.imread(str(img_path))
        if prepared_img is None:
            skipped += 1
            continue
        prepared_img = _ensure_rgb(prepared_img)

        try:
            yolo_bboxes = _yolo26_detect(model, prepared_img, confidence)
        except Exception:
            yolo_bboxes = []
            failed += 1

        try:
            kmeans_bboxes, _ = segment_kmeans_image(prepared_img)
            kmeans_bboxes = [
                {"x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"])}
                if "x" in b
                else {"x": int(b["xmin"]), "y": int(b["ymin"]), "w": int(b["xmax"] - b["xmin"]), "h": int(b["ymax"] - b["ymin"])}
                for b in (kmeans_bboxes or [])
            ]
        except Exception:
            kmeans_bboxes = []

        source_path = leaf_dir / "source.jpg"
        source_img = cv2.imread(str(source_path)) if source_path.exists() else None

        save_inference_artifacts(
            leaf_dir=leaf_dir,
            source_image=source_img,
            prepared_image=prepared_img,
            yolo26_bboxes=yolo_bboxes,
            kmeans_bboxes=kmeans_bboxes,
        )
        processed += 1

    return {
        "total": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "weights": weights_path,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Legacy experiment contract (backward compat)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@dataclass
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: list[str]
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    output_root = Path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = params.description[:30] if params.description else "yolo_segmentation"
    result_data = {
        "f1_score": 0.0,
        "strategy_name": strategy,
        "artifact_paths": [],
        "run_id": params.run_id,
    }
    (output_root / "results.json").write_text(json.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
