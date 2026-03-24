"""
5-Fold Strain-Level Cross-Validation
=====================================
Rotates the test strain across all strains for each species (round-robin).
Fixed extractor: EfficientNetB1_finetuned
Env strategies : [None/E1, "all"/E2]
Agg strategies : ["uni", "avg" (weighted)]
K values       : [3, 5, 7, 9, 11]
Total runs     : 5 folds × 2 env × 2 agg × 5 K = 100

Results are appended to ``report/week_1_2/cv_results.csv`` immediately after
each run, so the job is safe to interrupt and resume — already-completed
(fold, env_strategy, agg_strategy, k) combinations are skipped automatically.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from qdrant_client import QdrantClient

from src.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    RESULTS_DIR,
    STRAIN_SPECIES_MAPPING_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_DIR = Path(__file__).parent.parent.parent / "report" / "week_1_2"
CV_RESULTS_CSV = REPORT_DIR / "cv_results.csv"
CV_SUMMARY_CSV = REPORT_DIR / "cv_summary_table.csv"

CV_RESULTS_FIELDS = [
    "fold",
    "species",
    "strain",
    "ground_truth",
    "predicted_specy",
    "correct",
    "test_set_index",
    "env_strategy",
    "agg_strategy",
    "k",
    "extractor",
    "collection",
]

N_FOLDS = 5
K_VALUES = [3, 5, 7, 9, 11]
ENV_STRATEGIES: List[Optional[str]] = [None, "all"]  # E1, E2
AGG_STRATEGIES = ["uni", "avg"]  # uni = uniform, avg = score-weighted


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def generate_cv_folds(
    csv_path: Path = STRAIN_SPECIES_MAPPING_PATH,
    n_folds: int = N_FOLDS,
) -> List[Dict[str, str]]:
    """
    Return a list of *n_folds* dicts, each mapping ``{species: strain}``.

    The strains for each species are sorted alphabetically and assigned to
    folds via round-robin (``strain[fold_idx % len(strains)]``).
    Species that have fewer strains than *n_folds* will repeat earlier strains
    in later folds — this matches the task specification ("fold 5 reuses fold
    1's test strain for 4-strain species").
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Strain mapping CSV not found at {csv_path}. "
            "Run 'uv run python src/main.py generate-mapping' first."
        )

    df = pd.read_csv(csv_path)
    species_to_strains: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        species_to_strains[row["Species"]].append(row["Strain"])

    # Sort each species' strains for deterministic fold assignment
    for sp in species_to_strains:
        species_to_strains[sp].sort()

    folds: List[Dict[str, str]] = []
    for fold_idx in range(n_folds):
        fold: Dict[str, str] = {}
        for species, strains in species_to_strains.items():
            fold[species] = strains[fold_idx % len(strains)]
        folds.append(fold)

    return folds


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_completed_runs(csv_path: Path) -> set:
    """Return completed run keys from existing CSV."""
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                int(row["fold"]),
                row["env_strategy"],
                row["agg_strategy"],
                int(row["k"]),
                row.get("extractor", ""),
                row.get("collection", ""),
            )
            completed.add(key)
    return completed


def _append_rows(csv_path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_cross_validation(
    collection_name: Optional[str] = None,
    extractor_key: str = "efficientnetb1_finetuned",
    use_fold_specific_assets: bool = False,
    collection_template: Optional[str] = None,
    weights_dir: str = "weights",
    n_folds: int = N_FOLDS,
    k_values: List[int] = K_VALUES,
    env_strategies: List[Optional[str]] = ENV_STRATEGIES,
    agg_strategies: List[str] = AGG_STRATEGIES,
    output_dir: Optional[Path] = None,
) -> None:
    from src.classification.evaluate_species import run_species_evaluation
    from src.feature_extraction.feature_extractors import (
        EfficientNetB1FinetunedExtractor,
        EfficientNetB1Extractor,
        EfficientNetB1TripletExtractor,
        ResNet50FinetunedExtractor,
        MobileNetV2FinetunedExtractor,
        ResNet50Extractor,
        MobileNetV2Extractor,
    )

    _extractor_map = {
        "efficientnetb1_finetuned": EfficientNetB1FinetunedExtractor,
        "efficientnetb1": EfficientNetB1Extractor,
        "efficientnetb1_triplet": EfficientNetB1TripletExtractor,
        "resnet50_finetuned": ResNet50FinetunedExtractor,
        "resnet50": ResNet50Extractor,
        "mobilenetv2_finetuned": MobileNetV2FinetunedExtractor,
        "mobilenetv2": MobileNetV2Extractor,
    }

    if extractor_key not in _extractor_map:
        raise ValueError(
            f"Unknown extractor '{extractor_key}'. "
            f"Choose from: {list(_extractor_map.keys())}"
        )

    client = QdrantClient(url=QDRANT_URL)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir or (RESULTS_DIR / "cross_validation")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    folds = generate_cv_folds(n_folds=n_folds)
    completed = _load_completed_runs(CV_RESULTS_CSV)

    total = n_folds * len(env_strategies) * len(agg_strategies) * len(k_values)
    run_num = 0

    for fold_idx, fold_strains in enumerate(folds):
        if use_fold_specific_assets and extractor_key == "efficientnetb1_finetuned":
            fold_weight_path = Path(weights_dir) / f"fold{fold_idx}_EfficientNetB1_finetuned.pth"
            if not fold_weight_path.exists():
                raise FileNotFoundError(
                    f"Missing fold-specific weight: {fold_weight_path}. "
                    "Copy fold weights from Drive before running CV."
                )
            extractor = _extractor_map[extractor_key](weights_path=str(fold_weight_path))
            extractor_id = f"{extractor_key}_fold{fold_idx}"
        else:
            extractor = _extractor_map[extractor_key]()
            extractor_id = extractor_key

        if use_fold_specific_assets:
            if collection_template:
                coll = collection_template.format(fold=fold_idx)
            elif collection_name:
                coll = f"{collection_name}_fold{fold_idx}"
            else:
                coll = f"{COLLECTION_NAME}_finetuned_fold{fold_idx}"
        else:
            coll = collection_name or COLLECTION_NAME

        for env_val in env_strategies:
            env_label = "E1" if env_val is None else "E2"
            for agg in agg_strategies:
                for k in k_values:
                    run_num += 1
                    key = (fold_idx, env_label, agg, k, extractor_id, coll)

                    if key in completed:
                        print(
                            f"[{run_num}/{total}] SKIP fold={fold_idx} env={env_label}"
                            f" agg={agg} k={k} (already done)"
                        )
                        continue

                    print(f"\n{'='*60}")
                    print(
                        f"[{run_num}/{total}] fold={fold_idx}  env={env_label}"
                        f"  agg={agg}  k={k}"
                    )
                    print(f"  Test strains: {fold_strains}")
                    print(f"{'='*60}")

                    fold_out = (
                        run_output_dir
                        / f"fold{fold_idx}_{env_label}_{agg}_k{k}"
                    )
                    fold_out.mkdir(parents=True, exist_ok=True)

                    results, _ = run_species_evaluation(
                        client=client,
                        collection_name=coll,
                        feature_extractor=extractor,
                        k=k,
                        without_siblings=True,
                        environment=env_val,
                        strategy=agg,
                        output_dir=str(fold_out),
                        generate_visualizations=True,
                        selected_strains=fold_strains,
                    )

                    # Build CSV rows — one per prediction entry
                    rows: List[dict] = []
                    for res in results:
                        rows.append(
                            {
                                "fold": fold_idx,
                                "species": res.get("ground_truth", ""),
                                "strain": res.get("strain", ""),
                                "ground_truth": res.get("ground_truth", ""),
                                "predicted_specy": res.get("predicted_specy", ""),
                                "correct": int(bool(res.get("correct"))),
                                "test_set_index": res.get("test_set_index", 0),
                                "env_strategy": env_label,
                                "agg_strategy": agg,
                                "k": k,
                                "extractor": extractor_id,
                                "collection": coll,
                            }
                        )

                    _append_rows(CV_RESULTS_CSV, rows)

                    # Compute per-fold accuracy and print
                    if results:
                        acc = sum(r.get("correct", False) for r in results) / len(results)
                    else:
                        acc = 0.0
                    print(
                        f"  → accuracy (fold {fold_idx}, {env_label}, {agg}, k={k}) = {acc:.4f}"
                        f"  ({len(results)} test samples)"
                    )
                    completed.add(key)

    print(f"\n{'='*60}")
    print(f"Cross-validation complete.  {run_num} runs processed.")
    print(f"Results: {CV_RESULTS_CSV}")
    print(f"{'='*60}")

    _generate_summary(CV_RESULTS_CSV, CV_SUMMARY_CSV)
    print(f"Summary table: {CV_SUMMARY_CSV}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _generate_summary(results_csv: Path, summary_csv: Path) -> None:
    """Aggregate per-prediction rows into mean/std accuracy per (env, agg, k)."""
    if not results_csv.exists():
        print("No results CSV found; skipping summary.")
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        print("Results CSV is empty; skipping summary.")
        return

    # Compute per-fold accuracy first, then aggregate across folds
    fold_acc = (
        df.groupby(
            [
                "fold",
                "env_strategy",
                "agg_strategy",
                "k",
                "extractor",
                "collection",
            ]
        )["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "fold_accuracy"})
    )

    summary = (
        fold_acc.groupby(
            ["env_strategy", "agg_strategy", "k", "extractor", "collection"]
        )["fold_accuracy"]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            min_accuracy="min",
            max_accuracy="max",
        )
        .reset_index()
    )
    summary = summary.sort_values("mean_accuracy", ascending=False)
    summary.to_csv(summary_csv, index=False)
    print(f"\nTop 10 configurations:")
    print(summary.head(10).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    collection: Optional[str] = None,
    extractor: str = "efficientnetb1_finetuned",
    use_fold_specific_assets: bool = False,
    collection_template: Optional[str] = None,
    weights_dir: str = "weights",
) -> None:
    run_cross_validation(
        collection_name=collection,
        extractor_key=extractor,
        use_fold_specific_assets=use_fold_specific_assets,
        collection_template=collection_template,
        weights_dir=weights_dir,
    )


if __name__ == "__main__":
    main()
