from pathlib import Path

from PIL import Image

from src.experiments.yolo_cross_validation.visualize import build_visualization_index
from src.experiments.yolo_dataset.augmentation_preview import (
    build_augmentation_preview_summary,
)


def test_build_visualization_index_generates_plot(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "fold_id,species_name,test_strain_id,sample_count_train,sample_count_test,metric_accuracy,warning\n"
        "0,Species A,DTO 100-A1,6,1,0.8,\n"
        "1,Species A,DTO 100-A2,6,1,0.9,\n"
    )

    summary = build_visualization_index(tmp_path)

    assert summary["figure_count"] == 1
    assert summary["figures"][0].endswith("fold_accuracy.png")
    assert (tmp_path / "fold_accuracy.png").exists()


def test_augmentation_debug_grid_output_exists(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    output_path = tmp_path / "preview.jpg"
    Image.new("RGB", (64, 64), color=(120, 80, 60)).save(image_path)

    summary = build_augmentation_preview_summary(
        image_path=image_path,
        output_path=output_path,
        preview_count=4,
        columns=2,
        base_seed=7,
    )

    assert output_path.exists()
    preview = Image.open(output_path)
    assert preview.size == (128, 192)
    assert summary["preview_path"] == str(output_path)
