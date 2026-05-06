import json
from pathlib import Path

import cv2
import numpy as np

from src.config import COLLECTION_METADATA_PATHS
from src.prepare.dataset import (
    DatasetItemRecord,
    InstanceInfo,
    build_item_id,
    build_leaf_dir,
    parse_curated_metadata,
    parse_incoming_metadata,
    parse_source_metadata,
    prepare_dataset,
    run_segmentation,
    segment_item,
    _is_letter_range,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 80, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_is_letter_range() -> None:
    assert _is_letter_range("D - L")
    assert _is_letter_range("M - R")
    assert _is_letter_range("S - Z")
    assert not _is_letter_range("glandicola")
    assert not _is_letter_range("T230")


def test_parse_source_metadata_falls_back_to_folder_species() -> None:
    path = Path("/tmp/species-folder/mystery_edited.jpg")

    metadata = parse_source_metadata(path, {})

    assert metadata.species == "species-folder"
    assert metadata.strain == "unknown"
    assert metadata.environment == "unknown"
    assert metadata.angle == "unknown"
    assert metadata.parse_status == "fallback"


def test_parse_incoming_metadata_from_filename() -> None:
    path = Path("/tmp/D - L/glandicola/T230/T230 CREA ob.jpg")

    metadata = parse_incoming_metadata(path, {"T230": "Penicillium glandicola"})

    assert metadata.species == "Penicillium glandicola"
    assert metadata.strain == "T230"
    assert metadata.environment == "CREA"
    assert metadata.angle == "ob"
    assert metadata.parse_status == "parsed"


def test_parse_incoming_metadata_falls_back() -> None:
    path = Path("/tmp/M - R/unknown_sp/T999/mystery_sample.jpg")

    metadata = parse_incoming_metadata(path, {})

    assert metadata.species == "unknown_sp"
    assert metadata.strain == "T999"
    assert metadata.environment == "unknown"
    assert metadata.angle == "unknown"
    assert metadata.parse_status == "fallback"


def test_build_item_id_stable() -> None:
    info = InstanceInfo(species="Penicillium", strain="T230", environment="CREA", angle="ob")
    id1 = build_item_id(info)
    id2 = build_item_id(info)
    assert id1 == id2
    assert len(id1) == 32


def test_build_leaf_dir_uses_angle_as_leaf() -> None:
    from src.prepare.dataset import ParsedMetadata

    meta = ParsedMetadata(
        species="Penicillium", strain="T230", environment="CREA", angle="ob",
        parse_status="parsed",
    )
    leaf = build_leaf_dir(Path("/prepared"), meta)
    assert leaf == Path("/prepared/penicillium/t230/crea/ob")


def test_prepare_dataset_writes_canonical_tree(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Dataset"
    curated_root = dataset_root / "curated_primary"
    incoming_root = dataset_root / "incoming_low_quality"
    prepared_root = dataset_root / "prepared"
    curated_meta = dataset_root / "curated_primary_metadata.json"
    incoming_meta = dataset_root / "incoming_low_quality_metadata.json"
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
    monkeypatch.setattr(
        "src.prepare.dataset.COLLECTION_METADATA_PATHS",
        {"curated": curated_meta, "incoming": incoming_meta},
    )

    items = prepare_dataset(prepared_root=prepared_root)

    assert len(items) == 2

    curated_item = items[0]
    assert curated_item.source_collection == "curated_primary"
    assert curated_item.instance_info.species == "Penicillium polonicum"
    assert curated_item.instance_info.strain == "DTO 148-D1"
    assert curated_item.instance_info.environment == "MEA"
    assert curated_item.instance_info.angle == "ob"
    assert curated_item.parse_status == "parsed"

    leaf_dir = prepared_root / "penicillium-polonicum" / "dto-148-d1" / "mea" / "ob"
    assert leaf_dir.exists()
    assert (leaf_dir / "source.jpg").exists()
    assert (leaf_dir / "prepared.jpg").exists()

    stored = json.loads(curated_meta.read_text())
    assert len(stored) == 1
    assert stored[0]["source_collection"] == "curated_primary"
    assert stored[0]["instance_info"]["species"] == "Penicillium polonicum"
    assert stored[0]["instance_info"]["angle"] == "ob"

    incoming_stored = json.loads(incoming_meta.read_text())
    assert len(incoming_stored) == 1
    assert incoming_stored[0]["instance_info"]["species"] == "unknown-group"


def test_prepare_dataset_with_letter_range_incoming(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Dataset"
    incoming_root = dataset_root / "incoming_low_quality"
    prepared_root = dataset_root / "prepared"
    incoming_meta = dataset_root / "incoming_low_quality_metadata.json"
    curated_meta = dataset_root / "curated_primary_metadata.json"
    mapping_path = dataset_root / "strain_to_specy.csv"

    letter_range_dir = incoming_root / "D - L"
    species_dir = letter_range_dir / "glandicola"
    strain_dir = species_dir / "T230"
    _write_image(strain_dir / "T230 CREA ob.jpg")
    mapping_path.write_text("Strain,Species\nT230,Penicillium glandicola\n")

    monkeypatch.setattr(
        "src.prepare.dataset.SOURCE_COLLECTIONS",
        {
            "curated": {
                "display_name": "curated_primary",
                "quality_tier": "curated",
                "path": dataset_root / "curated_primary",
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
    monkeypatch.setattr(
        "src.prepare.dataset.COLLECTION_METADATA_PATHS",
        {"curated": curated_meta, "incoming": incoming_meta},
    )

    items = prepare_dataset(source_collections=["incoming"], prepared_root=prepared_root)

    assert len(items) == 1
    item = items[0]
    assert item.instance_info.species == "Penicillium glandicola"
    assert item.instance_info.strain == "T230"
    assert item.instance_info.environment == "CREA"
    assert item.instance_info.angle == "ob"

    leaf_dir = prepared_root / "penicillium-glandicola" / "t230" / "crea" / "ob"
    assert leaf_dir.exists()
    assert (leaf_dir / "source.jpg").exists()
    assert (leaf_dir / "prepared.jpg").exists()


def _make_item(tmp_path: Path) -> tuple[DatasetItemRecord, Path]:
    leaf_dir = tmp_path / "prepared" / "test-species" / "t230" / "crea" / "ob"
    leaf_dir.mkdir(parents=True)

    source_path = leaf_dir / "source.jpg"
    prepared_path = leaf_dir / "prepared.jpg"
    prepared = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(prepared, (128, 128), 80, (200, 180, 160), -1)
    cv2.imwrite(str(source_path), prepared)
    cv2.imwrite(str(prepared_path), prepared)

    item = DatasetItemRecord(
        item_id="test-item-1",
        source_collection="curated_primary",
        source_collection_path="Dataset/curated_primary",
        source_filename="test.jpg",
        instance_info=InstanceInfo(
            species="Test", strain="T230", environment="CREA", angle="ob",
        ),
        parse_status="parsed",
        paths={
            "source": str(source_path),
            "prepared": str(prepared_path),
            "segments": [],
        },
        segmentation={},
    )
    return item, leaf_dir


def test_segment_item_kmeans_writes_to_leaf_segments(tmp_path: Path, monkeypatch) -> None:
    item, leaf_dir = _make_item(tmp_path)
    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))

    results = segment_item(item, methods=["kmeans"])

    assert len(results) == 1
    assert results[0]["method"] == "kmeans"
    assert results[0]["status"] in ("success", "empty")

    segments_dir = leaf_dir / "segments"
    assert segments_dir.exists()

    assert item.segmentation.get("kmeans") is not None
    assert "bbox_kmeans" in item.paths
    assert "pipeline_kmeans" in item.paths


def test_segment_item_contour_writes_to_leaf_segments(tmp_path: Path, monkeypatch) -> None:
    item, leaf_dir = _make_item(tmp_path)
    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))

    results = segment_item(item, methods=["contour"])

    assert len(results) == 1
    assert results[0]["method"] == "contour"
    assert results[0]["status"] in ("success", "empty")

    segments_dir = leaf_dir / "segments"
    assert segments_dir.exists()
    assert "bbox_contour" in item.paths or results[0]["status"] == "empty"


def test_segment_item_both_methods_shares_segments_dir(tmp_path: Path, monkeypatch) -> None:
    item, leaf_dir = _make_item(tmp_path)
    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))

    results = segment_item(item, methods=["kmeans", "contour"])

    assert len(results) == 2
    segments_dir = leaf_dir / "segments"
    assert segments_dir.exists()

    if results[0]["status"] == "success":
        assert item.paths.get("segments")


def test_segment_item_segment_naming_is_1_indexed(tmp_path: Path, monkeypatch) -> None:
    item, leaf_dir = _make_item(tmp_path)
    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))

    segment_item(item, methods=["kmeans"])

    segments_dir = leaf_dir / "segments"
    if segments_dir.exists():
        for seg_file in segments_dir.iterdir():
            assert seg_file.name.startswith("segment_")
            num = seg_file.stem.replace("segment_", "")
            assert int(num) >= 1


def test_segment_item_missing_image_returns_failed(tmp_path: Path, monkeypatch) -> None:
    leaf_dir = tmp_path / "prepared" / "test-species" / "t230" / "crea" / "ob"
    leaf_dir.mkdir(parents=True)

    item = DatasetItemRecord(
        item_id="test-item-2",
        source_collection="curated_primary",
        source_collection_path="Dataset/curated_primary",
        source_filename="test.jpg",
        instance_info=InstanceInfo(
            species="Test", strain="T230", environment="CREA", angle="ob",
        ),
        parse_status="parsed",
        paths={
            "source": str(leaf_dir / "missing.jpg"),
            "prepared": str(leaf_dir / "missing.jpg"),
            "segments": [],
        },
        segmentation={},
    )

    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    results = segment_item(item, methods=["kmeans"])

    assert len(results) == 1
    assert results[0]["status"] == "failed"


def test_run_segmentation_updates_items_in_place(tmp_path: Path, monkeypatch) -> None:
    item, leaf_dir = _make_item(tmp_path)
    monkeypatch.setattr("src.config.WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr("src.prepare.dataset.relative_to_workspace", lambda p: str(p))
    monkeypatch.setattr(
        "src.prepare.dataset.COLLECTION_METADATA_PATHS",
        {},
    )

    run_segmentation([item], methods=["kmeans"])

    assert item.segmentation.get("kmeans") is not None
    segments = item.paths.get("segments", [])
    if segments:
        assert all("segment_" in p for p in segments)


def test_item_to_dict_contains_required_fields() -> None:
    item = DatasetItemRecord(
        item_id="abc123",
        source_collection="curated_primary",
        source_collection_path="Dataset/curated_primary",
        source_filename="image.jpg",
        instance_info=InstanceInfo(
            species="Penicillium", strain="T230", environment="CREA", angle="ob",
        ),
        parse_status="parsed",
        paths={"source": "a", "prepared": "b", "segments": []},
        segmentation={"kmeans": [{"x": 0, "y": 0, "w": 64, "h": 64}]},
    )

    d = item.to_dict()
    assert "item_id" in d
    assert "instance_info" in d
    assert d["instance_info"]["species"] == "Penicillium"
    assert d["instance_info"]["angle"] == "ob"
    assert d["paths"]["source"] == "a"
    assert "segments" in d["paths"]
    assert len(d["segmentation"]["kmeans"]) == 1
    assert d["segmentation"]["kmeans"][0]["x"] == 0
