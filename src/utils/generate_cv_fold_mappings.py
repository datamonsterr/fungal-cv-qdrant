"""Generate 5 fold-specific strain/species mapping CSV files.

Each generated file preserves the full mapping schema:
- Strain
- Species
- Test

For fold i, exactly one strain per species is marked as Test=True using the
same deterministic round-robin rule as cross_validation.generate_cv_folds.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.config import DATASET_ROOT, STRAIN_SPECIES_MAPPING_PATH
from src.experiments.cross_validation.run import N_FOLDS, generate_cv_folds


def generate_fold_mapping_files(
    source_csv: Path = STRAIN_SPECIES_MAPPING_PATH,
    output_dir: Path = DATASET_ROOT,
    n_folds: int = N_FOLDS,
    filename_prefix: str = "strain_to_specy_fold",
) -> List[Path]:
    """Generate fold mapping CSV files and return created paths."""
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Source mapping CSV not found at {source_csv}. "
            "Run 'uv run python -m src.utils.generate_strain_mapping' first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(source_csv)
    if "Strain" not in base_df.columns or "Species" not in base_df.columns:
        raise ValueError(
            "Source mapping CSV must contain 'Strain' and 'Species' columns."
        )

    folds = generate_cv_folds(csv_path=source_csv, n_folds=n_folds)
    generated_files: List[Path] = []

    for fold_idx, fold_selection in enumerate(folds):
        fold_df = base_df.copy()
        fold_df["Test"] = False

        for row_idx, row in fold_df.iterrows():
            species = row["Species"]
            strain = row["Strain"]
            fold_df.at[row_idx, "Test"] = fold_selection.get(species) == strain

        out_path = output_dir / f"{filename_prefix}{fold_idx}.csv"
        fold_df.to_csv(out_path, index=False)
        generated_files.append(out_path)

        test_count = int(fold_df["Test"].sum())
        species_count = fold_df["Species"].nunique()
        print(
            f"Fold {fold_idx}: wrote {out_path} "
            f"(test strains={test_count}, species={species_count})"
        )

    return generated_files


def main() -> None:
    print("Generating fold-specific strain mapping files...")
    generated = generate_fold_mapping_files()
    print(f"Done. Generated {len(generated)} files.")


if __name__ == "__main__":
    main()
