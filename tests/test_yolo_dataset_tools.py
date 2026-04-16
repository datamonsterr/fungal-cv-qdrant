from pathlib import Path

import numpy as np

from src.preprocessing.preprocess import center_crop_square, prepare_image
from tools.crop_yolo_segments import yolo_line_to_bbox
from tools.export_yolo_dataset import bbox_to_yolo_line, parse_source_metadata


def test_center_crop_square_uses_middle_region() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, :1] = 10
    image[:, 1:5] = 20
    image[:, 5:] = 30

    cropped = center_crop_square(image)

    assert cropped.shape == (4, 4, 3)
    assert np.all(cropped[:, 0] == 20)
    assert np.all(cropped[:, -1] == 20)


def test_prepare_image_resizes_and_falls_back_when_circle_missing() -> None:
    image = np.full((300, 500, 3), 180, dtype=np.uint8)

    artifacts = prepare_image(image)

    assert artifacts.square_image.shape == (300, 300, 3)
    assert artifacts.export_image.shape == (3000, 3000, 3)
    assert artifacts.masked_export_image.shape == (3000, 3000, 3)
    assert artifacts.working_image.shape[0] == artifacts.working_image.shape[1]
    assert artifacts.circle_detected is False


def test_parse_source_metadata_handles_contaminated_suffix() -> None:
    source_path = Path(
        "/tmp/DTO 478-C6 Penicillium viridicatum/DTO 478-C6 CYArev contaminated_edited.jpg"
    )

    metadata = parse_source_metadata(source_path, {})

    assert metadata == {
        "strain": "DTO 478-C6",
        "species": "Penicillium viridicatum",
        "environment": "CYA",
        "angle": "rev",
    }


def test_yolo_bbox_round_trip() -> None:
    bbox = {"xmin": 100, "ymin": 50, "xmax": 300, "ymax": 250}

    line = bbox_to_yolo_line(bbox, image_width=1000, image_height=500)
    restored = yolo_line_to_bbox(line, image_width=1000, image_height=500)

    assert restored == bbox
