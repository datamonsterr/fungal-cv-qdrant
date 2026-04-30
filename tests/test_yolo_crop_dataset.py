from pathlib import Path

import torch
from PIL import Image

from src.experiments.finetune_dl.crop_dataset import create_crop_dataset
from src.experiments.finetune_dl.train_yolo_crops import (
    build_model,
    export_backbone_weights,
)


def test_create_crop_dataset_preserves_parent_split_assignment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "species_dataset"
    train_image_dir = dataset_root / "train" / "images"
    train_label_dir = dataset_root / "train" / "labels"
    test_image_dir = dataset_root / "test" / "images"
    test_label_dir = dataset_root / "test" / "labels"
    train_image_dir.mkdir(parents=True)
    train_label_dir.mkdir(parents=True)
    test_image_dir.mkdir(parents=True)
    test_label_dir.mkdir(parents=True)

    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(
        train_image_dir / "DTO_100-A1_train.jpg"
    )
    (train_label_dir / "DTO_100-A1_train.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    Image.new("RGB", (100, 100), color=(80, 80, 80)).save(
        test_image_dir / "DTO_100-A1_test.jpg"
    )
    (test_label_dir / "DTO_100-A1_test.txt").write_text("1 0.5 0.5 0.2 0.2\n")

    output_root = tmp_path / "crop_dataset"
    summary = create_crop_dataset(dataset_root, output_root, crop_size=64)

    assert summary["train_count"] == 1
    assert summary["test_count"] == 1
    assert (output_root / "train" / "0").exists()
    assert (output_root / "test" / "1").exists()
    assert (output_root / "crop_assignment.csv").exists()


def test_export_backbone_weights_excludes_classifier_layers(tmp_path: Path) -> None:
    model = build_model("ResNet50", num_classes=2)
    weights_path = export_backbone_weights(
        model,
        model_name="ResNet50",
        output_path=tmp_path / "ResNet50_finetuned.pth",
    )

    state_dict = torch.load(weights_path, map_location="cpu")

    assert weights_path.exists()
    assert not any(key.startswith("fc.") for key in state_dict)
