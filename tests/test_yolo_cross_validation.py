from pathlib import Path

from src.utils.yolo_cross_validation import (
    build_fold_summary_rows,
    build_metrics_rows,
    build_strict_cv_folds,
)


def test_build_strict_cv_folds_selects_unique_strains_per_species() -> None:
    csv_path = Path("/tmp/strain_to_specy_test.csv")
    csv_path.write_text(
        "Strain,Species,Test\n"
        "DTO 100-A1,Species A,False\n"
        "DTO 100-A2,Species A,False\n"
        "DTO 200-B1,Species B,False\n"
        "DTO 200-B2,Species B,False\n"
    )

    folds = build_strict_cv_folds(csv_path=csv_path, n_folds=2)

    assert len(folds) == 2
    assert folds[0]["Species A"] != folds[1]["Species A"]
    assert folds[0]["Species B"] != folds[1]["Species B"]


def test_build_strict_cv_folds_uses_round_robin_for_small_species() -> None:
    csv_path = Path("/tmp/strain_to_specy_round_robin.csv")
    csv_path.write_text(
        "Strain,Species,Test\n"
        "DTO 100-A1,Species A,False\n"
        "DTO 200-B1,Species B,False\n"
        "DTO 200-B2,Species B,False\n"
    )

    folds = build_strict_cv_folds(csv_path=csv_path, n_folds=3)

    assert len(folds) == 3
    assert folds[0]["Species A"] == "DTO 100-A1"
    assert folds[1]["Species A"] == "DTO 100-A1"
    assert folds[0]["Species B"] == "DTO 200-B1"
    assert folds[1]["Species B"] == "DTO 200-B2"
    assert folds[2]["Species B"] == "DTO 200-B1"


def test_fold_summary_and_metrics_rows_include_round_robin_warning() -> None:
    csv_path = Path("/tmp/strain_to_specy_summary.csv")
    csv_path.write_text(
        "Strain,Species,Test\n"
        "DTO 100-A1,Species A,False\n"
        "DTO 100-A2,Species A,False\n"
        "DTO 200-B1,Species B,False\n"
    )

    folds = build_strict_cv_folds(csv_path=csv_path, n_folds=2)
    summary_rows = build_fold_summary_rows(folds)
    metrics_rows = build_metrics_rows(folds)

    assert summary_rows
    assert metrics_rows
    assert any("Species B" in row["round_robin_species"] for row in summary_rows)
    assert any("Round-robin species:" in row["warning"] for row in metrics_rows)
