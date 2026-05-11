"""Unit tests for yolo_segmentation experiment — prepare, train, infer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestCleanup:
    def test_stale_paths_defined(self):
        from src.experiments.yolo_segmentation.prepare import STALE_DIRS, STALE_FILES
        assert len(STALE_DIRS) >= 2
        assert len(STALE_FILES) >= 2

    def test_cleanup_only_removes_existing(self):
        from src.experiments.yolo_segmentation.prepare import cleanup_stale_datasets
        removed = cleanup_stale_datasets()
        for r in removed:
            assert r.endswith("full_image") or r.endswith("segmented_image") or "metadata" in r


class TestValidateDataset:
    def test_report_on_real_dataset(self):
        from src.experiments.yolo_segmentation.prepare import validate_yolo_dataset
        report = validate_yolo_dataset()
        assert report.total_images == 435
        assert report.total_labels == 435
        assert len(report.class_names) == 8
        assert "Penicillium cyclopium" in report.class_names
        assert report.valid is True

    def test_report_on_missing_dir(self):
        from src.experiments.yolo_segmentation.prepare import validate_yolo_dataset
        report = validate_yolo_dataset(dataset_dir=Path("/nonexistent"))
        assert report.total_images == 0
        assert report.valid is False

    def test_report_summary_contains_counts(self):
        from src.experiments.yolo_segmentation.prepare import validate_yolo_dataset
        report = validate_yolo_dataset()
        s = report.summary()
        assert "Images: 435" in s
        assert "Valid: True" in s


class TestDatasetSplit:
    def test_train_val_split_ratio(self):
        from src.experiments.yolo_segmentation.prepare import create_train_val_split
        train, val, train_txt, val_txt = create_train_val_split(
            train_ratio=0.8, seed=42
        )
        assert len(train) + len(val) == 348
        assert len(train) == 303
        assert len(val) == 45
        assert train_txt.exists()
        assert val_txt.exists()

    def test_split_reproducible(self):
        from src.experiments.yolo_segmentation.prepare import create_train_val_split
        t1, _, _, _ = create_train_val_split(train_ratio=0.8, seed=42)
        t2, _, _, _ = create_train_val_split(train_ratio=0.8, seed=42)
        assert t1 == t2

    def test_split_no_overlap(self):
        from src.experiments.yolo_segmentation.prepare import create_train_val_split
        train, val, _, _ = create_train_val_split(train_ratio=0.8)
        train_set = set(train)
        val_set = set(val)
        assert train_set.isdisjoint(val_set)

    def test_split_empty_dir_raises(self):
        from src.experiments.yolo_segmentation.prepare import create_train_val_split
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "images").mkdir()
            with pytest.raises(FileNotFoundError):
                create_train_val_split(dataset_dir=root)


class TestYamlRewrite:
    def test_rewrite_creates_valid_yaml(self):
        from src.experiments.yolo_segmentation.prepare import rewrite_dataset_yaml
        import yaml
        yaml_path = rewrite_dataset_yaml()
        data = yaml.safe_load(yaml_path.read_text())
        assert "path" in data
        assert "train" in data
        assert "val" in data
        assert len(data["names"]) == 8
        assert data["names"][0] == "Penicillium aurantiogriseum"


class TestFinetuneConfig:
    def test_default_config(self):
        from src.experiments.yolo_segmentation.run import FinetuneConfig
        cfg = FinetuneConfig()
        assert cfg.model_variant == "n"
        assert cfg.epochs == 100
        assert cfg.batch == 8
        assert cfg.imgsz == 640

    def test_invalid_variant_rejected(self):
        from src.experiments.yolo_segmentation.run import FinetuneConfig
        with pytest.raises(ValueError):
            FinetuneConfig(model_variant="z")


class TestGPUGuard:
    def test_nano_variant_passes_6gb(self):
        from src.experiments.yolo_segmentation.run import estimate_vram, check_vram_fit
        vram = estimate_vram("n", imgsz=640)
        assert check_vram_fit("n", available_vram_gb=6.0)

    def test_xlarge_variant_fails_6gb(self):
        from src.experiments.yolo_segmentation.run import check_vram_fit
        assert not check_vram_fit("x", available_vram_gb=6.0)


class TestInferenceBboxConversion:
    def test_normalized_to_pixel(self):
        from src.experiments.yolo_segmentation.run import norm_xywh_to_pixel
        bbox = norm_xywh_to_pixel(0.5, 0.5, 0.2, 0.3, 640, 640)
        assert bbox == {"x": 256, "y": 224, "w": 128, "h": 192}


class TestMetadataJsonFormat:
    def _make_bbox(self, x, y, w, h, confidence=0.9):
        return {"x": x, "y": y, "w": w, "h": h, "confidence": confidence}

    def test_schema_matches_dataset_item_record(self):
        from src.experiments.yolo_segmentation.run import save_inference_artifacts
        import json, tempfile, numpy as np, cv2

        bboxes = [self._make_bbox(10, 20, 30, 40)]
        with tempfile.TemporaryDirectory() as td:
            leaf = Path(td)
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = save_inference_artifacts(
                leaf, source_image=img, prepared_image=img,
                yolo26_bboxes=bboxes, kmeans_bboxes=bboxes,
            )

            assert "yolo26" in result
            assert "kmeans" in result
            for key in ("yolo26", "kmeans"):
                assert isinstance(result[key], list)
                for item in result[key]:
                    assert isinstance(item, dict)
                    for k in ("x", "y", "w", "h"):
                        assert k in item
                        assert isinstance(item[k], int)

            metadata_path = leaf / "metadata.json"
            assert metadata_path.exists()
            loaded = json.loads(metadata_path.read_text())
            assert loaded == result

    def test_empty_detections_write_empty_arrays(self):
        from src.experiments.yolo_segmentation.run import save_inference_artifacts
        import json, tempfile, numpy as np

        with tempfile.TemporaryDirectory() as td:
            leaf = Path(td)
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            result = save_inference_artifacts(
                leaf, source_image=None, prepared_image=img,
                yolo26_bboxes=[], kmeans_bboxes=[],
            )
            assert result["yolo26"] == []
            assert result["kmeans"] == []

            loaded = (Path(td) / "metadata.json").read_text()
            parsed = json.loads(loaded)
            assert parsed["yolo26"] == []
            assert parsed["kmeans"] == []

    def test_bboxes_are_serializable_to_dataset_item_record(self):
        from src.experiments.yolo_segmentation.run import save_inference_artifacts
        from src.prepare.dataset import DatasetItemRecord, InstanceInfo
        import tempfile, numpy as np

        bboxes = [self._make_bbox(5, 10, 100, 200), self._make_bbox(50, 60, 80, 90)]
        with tempfile.TemporaryDirectory() as td:
            leaf = Path(td)
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            result = save_inference_artifacts(
                leaf, source_image=None, prepared_image=img,
                yolo26_bboxes=bboxes, kmeans_bboxes=[self._make_bbox(0, 0, 10, 10)],
            )

            record = DatasetItemRecord(
                item_id="test",
                source_collection="test",
                source_collection_path="test",
                source_filename="test.jpg",
                instance_info=InstanceInfo(
                    species="penicillium-cyclopium",
                    strain="dto-001-a1",
                    environment="cya",
                    angle="ob",
                ),
                parse_status="ok",
            )
            record.segmentation = result

            seg = record.segmentation
            assert seg["yolo26"] == [
                {"x": 5, "y": 10, "w": 100, "h": 200},
                {"x": 50, "y": 60, "w": 80, "h": 90},
            ]
            assert seg["kmeans"] == [
                {"x": 0, "y": 0, "w": 10, "h": 10},
            ]
