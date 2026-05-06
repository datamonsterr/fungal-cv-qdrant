from pathlib import Path

import cv2
import numpy as np

from src.analysis.dataset_eda import build_dataset_eda_report


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(path), image)



def test_build_dataset_eda_report_summarizes_each_collection(monkeypatch, tmp_path: Path) -> None:
    dataset_root = tmp_path / "Dataset"
    curated_root = dataset_root / "curated_primary"
    incoming_root = dataset_root / "incoming_low_quality"
    mapping_path = dataset_root / "strain_to_specy.csv"

    _write_image(
        curated_root
        / "DTO 148-D1 Penicillium polonicum"
        / "DTO 148-D1 MEAob_edited.jpg"
    )
    _write_image(incoming_root / "Aspergillus section" / "unknown_file.jpg")
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
    monkeypatch.setattr(
        "src.analysis.dataset_eda.SOURCE_COLLECTIONS",
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

    report = build_dataset_eda_report(["curated", "incoming"])

    assert report.total_images == 2
    assert report.total_species == 2
    assert len(report.collections) == 2
    assert report.collections[0].collection_key == "curated"
    assert report.collections[0].species_image_counts == {"Penicillium polonicum": 1}
    assert report.collections[1].species_image_counts == {"Aspergillus section": 1}
