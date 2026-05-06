import json
from pathlib import Path

import cv2
import numpy as np

from src.prepare.dataset import (
    DatasetItemRecord,
    parse_source_metadata,
    prepare_dataset,
    run_segmentation,
    segment_item,
    SegmentationResult,
    write_segment_metadata,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 80, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_parse_source_metadata_falls_back_to_folder_species() -> None:
    path = Path("/tmp/species-folder/mystery_edited.jpg")

    metadata = parse_source_metadata(path, {})

    assert metadata.species == "species-folder"
    assert metadata.strain == "unknown"
    assert metadata.environment == "unknown"
    assert metadata.angle == "unknown"
    assert metadata.parse_status == "fallback"


def test_prepare_dataset_writes_canonical_tree(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Dataset"
    curated_root = dataset_root / "curated_primary"
    incoming_root = dataset_root / "incoming_low_quality"
    prepared_root = dataset_root / "prepared"
    items_path = dataset_root / "prepared_items.json"
    segments_path = dataset_root / "prepared_segments.json"
    mapping_path = dataset_root / "strain_to_specy.csv"

    _write_image(
        curated_root
        / "DTO 148-D1 Penicillium polonicum"
        / "DTO 148-D1 MEAob_edited.jpg"
    )
    _write_image(
        incoming_root
        / "unknown-group"
        / "unparsed_sample.jpg"
    )
    mapping_path.write_text("Strain,Species\nDTO 148-D1,Penicillium polonicum\n")

    monkeypatch.setattr(
        "src.prepare.dataset.SOURCE_COLLECTIONS",
        {
            "curated": {
                "display_name": "curated_primary",
                "quality_tier": "curated",
                "path": curated_root,
            },
            "incoming": {
                "display_name": "incoming_low_quality",
                "quality_tier": "incoming",
                "path": incoming_root,
            },
        },
    )
    monkeypatch.setattr("src.prepare.dataset.STRAIN_SPECIES_MAPPING_PATH", mapping_path)
    monkeypatch.setattr("src.prepare.dataset.PREPARED_DATASET_DIR", prepared_root)
    monkeypatch.setattr("src.prepare.dataset.PREPARED_ITEMS_METADATA_PATH", items_path)
    monkeypatch.setattr("src.prepare.dataset.PREPARED_SEGMENTS_METADATA_PATH", segments_path)

    items, segments = prepare_dataset(
        prepared_root=prepared_root,
        items_metadata_path=items_path,
        segments_metadata_path=segments_path,
    )

    assert len(items) == 2
    assert segments == []

    first_root = prepared_root / "penicillium-polonicum" / "dto-148-d1" / "mea" / "dto-148-d1-meaob"
    second_root = prepared_root / "unknown-group" / "unknown" / "unknown" / "unparsed-sample"

    assert (first_root / "source.jpg").exists()
    assert (first_root / "prepared.jpg").exists()
    assert (first_root / "item.json").exists()
    assert (second_root / "source.jpg").exists()
    assert (second_root / "prepared.jpg").exists()

    stored_items = json.loads(items_path.read_text())
    assert stored_items[0]["source_collection"] == "curated_primary"
    assert stored_items[0]["species"] == "Penicillium polonicum"
    assert stored_items[1]["species"] == "unknown-group"
    assert json.loads(segments_path.read_text()) == []


def _mock_item_record(artifact_root: Path) -> DatasetItemRecord:
    return DatasetItemRecord(
        item_id="test-item-1",
        source_collection="curated_primary",
        source_collection_path="Dataset/curated_primary",
        source_filename="test.jpg",
        species="Penicillium test",
        strain="DTO 000-T1",
        environment="MEA",
        angle="ob",
        parse_status="parsed",
        source_image_path=f"{artifact_root}/source.jpg",
        prepared_image_path=f"{artifact_root}/prepared.jpg",
        artifact_root=str(artifact_root),
        item_record_path=f"{artifact_root}/item.json",
    )


def test_segment_item_kmeans_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "prepared" / "test-species" / "dto-000-t1" / "mea" / "stem"
    artifact_root.mkdir(parents=True)
    item = _mock_item_record(artifact_root)

    prepared = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(prepared, (128, 128), 80, (200, 180, 160), -1)
    cv2.imwrite(str(artifact_root / "prepared.jpg"), prepared)
    cv2.imwrite(str(artifact_root / "source.jpg"), prepared)

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    results = segment_item(item, methods=["kmeans"])

    assert len(results) == 1
    assert results[0].method == "kmeans"
    assert results[0].status in ("success", "empty", "failed")
    assert (artifact_root / "segments_kmeans").exists()
    assert (artifact_root / "bbox_kmeans.jpg").exists()
    assert (artifact_root / "pipeline_kmeans.jpg").exists()


def test_segment_item_contour_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "prepared" / "test-species" / "dto-000-t1" / "mea" / "stem"
    artifact_root.mkdir(parents=True)
    item = _mock_item_record(artifact_root)

    prepared = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(prepared, (128, 128), 80, (200, 180, 160), -1)
    cv2.imwrite(str(artifact_root / "prepared.jpg"), prepared)
    cv2.imwrite(str(artifact_root / "source.jpg"), prepared)

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    results = segment_item(item, methods=["contour"])

    assert len(results) == 1
    assert results[0].method == "contour"
    assert results[0].status in ("success", "empty", "failed")
    assert (artifact_root / "segments_contour").exists()


def test_segment_item_both_methods_creates_separate_dirs(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "prepared" / "test-species" / "dto-000-t1" / "mea" / "stem"
    artifact_root.mkdir(parents=True)
    item = _mock_item_record(artifact_root)

    prepared = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(prepared, (128, 128), 80, (200, 180, 160), -1)
    cv2.imwrite(str(artifact_root / "prepared.jpg"), prepared)
    cv2.imwrite(str(artifact_root / "source.jpg"), prepared)

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    results = segment_item(item, methods=["kmeans", "contour"])

    assert len(results) == 2
    methods = {r.method for r in results}
    assert "kmeans" in methods
    assert "contour" in methods
    assert (artifact_root / "segments_kmeans").exists()
    assert (artifact_root / "segments_contour").exists()


def test_segment_item_missing_image_returns_failed(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "prepared" / "test-species" / "dto-000-t1" / "mea" / "stem"
    artifact_root.mkdir(parents=True)
    item = _mock_item_record(artifact_root)

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    results = segment_item(item, methods=["kmeans"])

    assert len(results) == 1
    assert results[0].status == "failed"


def test_run_segmentation_generates_segment_records(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "prepared" / "test-species" / "dto-000-t1" / "mea" / "stem"
    artifact_root.mkdir(parents=True)
    item = _mock_item_record(artifact_root)

    prepared = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(prepared, (128, 128), 80, (200, 180, 160), -1)
    cv2.imwrite(str(artifact_root / "prepared.jpg"), prepared)
    cv2.imwrite(str(artifact_root / "source.jpg"), prepared)

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))

    records = run_segmentation([item], methods=["kmeans"])

    assert isinstance(records, list)
    for rec in records:
        assert rec.method == "kmeans"
        assert rec.parent_item_id == "test-item-1"
        assert "seg_" in rec.segment_path


def test_write_segment_metadata_creates_json(tmp_path: Path) -> None:
    from src.prepare.dataset import SegmentRecord

    records = [
        SegmentRecord(
            segment_id="seg-1",
            parent_item_id="item-1",
            method="kmeans",
            segment_index=0,
            segment_path="Dataset/prepared/test/seg_0.jpg",
            species="Test",
            strain="T1",
            environment="MEA",
            angle="ob",
            bbox={"xmin": 0, "ymin": 0, "xmax": 64, "ymax": 64},
        )
    ]
    path = tmp_path / "segments.json"
    write_segment_metadata(records, path=path)

    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data) == 1
    assert data[0]["segment_id"] == "seg-1"
    assert data[0]["segment_path"] == "Dataset/prepared/test/seg_0.jpg"
