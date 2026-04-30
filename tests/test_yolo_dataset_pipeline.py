import json
from pathlib import Path

from src.experiments.yolo_segmentation.run import run_yolo_segmentation
from src.utils.yolo_dataset_pipeline import (
    build_dataset_yaml,
    build_species_class_manifest,
    build_train_test_manifest,
    normalize_strain_id,
    parse_dto_strain_id,
    rewrite_label_content,
)


def test_parse_dto_strain_id_extracts_filename_token() -> None:
    assert parse_dto_strain_id("DTO_478-C6_CYA_ob_sample.jpg") == "DTO_478-C6"


def test_normalize_strain_id_converts_underscores_to_mapping_format() -> None:
    assert normalize_strain_id("DTO_478-C6") == "DTO 478-C6"


def test_build_species_class_manifest_is_sorted_and_stable() -> None:
    manifest = build_species_class_manifest(
        [
            "Penicillium viridicatum",
            "Penicillium aurantiogriseum",
            "Penicillium viridicatum",
        ]
    )

    assert manifest == {
        "Penicillium aurantiogriseum": 0,
        "Penicillium viridicatum": 1,
    }


def test_build_dataset_yaml_uses_test_split_as_val() -> None:
    output_root = Path("/tmp/derived-dataset")
    yaml_text = build_dataset_yaml(
        output_root,
        {
            "Penicillium aurantiogriseum": 0,
            "Penicillium viridicatum": 1,
        },
    )

    assert "train: train/images" in yaml_text
    assert "val: test/images" in yaml_text
    assert "0: Penicillium aurantiogriseum" in yaml_text
    assert "1: Penicillium viridicatum" in yaml_text


def test_rewrite_label_content_preserves_geometry_and_replaces_class_id() -> None:
    label_text = "0 0.1 0.2 0.3 0.4 0.5 0.6\n0 0.7 0.8 0.9 1.0 0.2 0.3\n"

    rewritten = rewrite_label_content(label_text, class_id=5)

    assert rewritten.splitlines()[0].startswith("5 ")
    assert "0.1 0.2 0.3 0.4 0.5 0.6" in rewritten


def test_build_train_test_manifest_is_deterministic(tmp_path: Path) -> None:
    dataset_root = tmp_path / "species_dataset"
    for split_name in ["train", "test"]:
        image_dir = dataset_root / split_name / "images"
        image_dir.mkdir(parents=True)
        for index in range(3):
            (image_dir / f"sample_{split_name}_{index}.jpg").write_bytes(b"img")

    manifest_a = build_train_test_manifest(dataset_root, test_ratio=0.34, seed=123)
    manifest_b = build_train_test_manifest(dataset_root, test_ratio=0.34, seed=123)

    assert manifest_a == manifest_b
    assert set(manifest_a["train"]).isdisjoint(set(manifest_a["test"]))


def test_run_yolo_segmentation_writes_split_metadata(tmp_path: Path) -> None:
    dataset_root = tmp_path / "species_dataset"
    for split_name in ["train", "test"]:
        image_dir = dataset_root / split_name / "images"
        image_dir.mkdir(parents=True)
        for index in range(2):
            (image_dir / f"sample_{split_name}_{index}.jpg").write_bytes(b"img")

    (dataset_root / "dataset.yaml").write_text(
        build_dataset_yaml(dataset_root, {"Species A": 0, "Species B": 1})
    )

    summary = run_yolo_segmentation(dataset_root, train=False)
    metrics = json.loads(Path(summary["metrics_path"]).read_text())
    metadata = json.loads(Path(summary["metadata_path"]).read_text())

    assert metrics["has_validation_split"] is False
    assert metadata["dataset_root"] == str(dataset_root)
    assert set(metadata.keys()) >= {"train_count", "test_count", "split_seed"}
