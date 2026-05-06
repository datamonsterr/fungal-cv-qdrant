"""
5-Fold Strain-Level Cross-Validation
=====================================
Rotates the test strain across all strains for each species (round-robin).
Fixed extractor: EfficientNetB1_finetuned
Env strategies : [None/E1, "all"/E2]
Agg strategies : ["weighted", "uni"]
K values       : [3, 5, 7, 9, 11]
Total runs     : 5 folds × 2 env × 2 agg × 5 K = 100

Results are appended to ``report/week_1_2/cv_results.csv`` immediately after
each run, so the job is safe to interrupt and resume — already-completed
(fold, env_strategy, agg_strategy, k) combinations are skipped automatically.

Parallel execution: within each fold, all (env × agg × k) combos run concurrently
using ThreadPoolExecutor. Progress is displayed via tqdm.

For the simpler single-config experiment runner, see ``src/lib/cross_validation.py``.
"""

from __future__ import annotations

import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass as _dc_autolab
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import List as _List_autolab

import pandas as pd
from qdrant_client import QdrantClient
from tqdm import tqdm

from src.config import (
    COLLECTION_NAME,
    QDRANT_API_KEY,
    QDRANT_URL,
    RESULTS_DIR,
    SEGMENTED_IMAGE_DIR,
    STRAIN_SPECIES_MAPPING_PATH,
)

# Re-use fold generation from shared library
from src.lib.cross_validation import generate_cv_folds as _lib_generate_cv_folds

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
AGG_STRATEGIES = ["uni", "weighted"]  # uni = uniform, weighted = score-weighted

# Thread-safe CSV write lock
_csv_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------


def generate_cv_folds(
    csv_path: Path = STRAIN_SPECIES_MAPPING_PATH,
    n_folds: int = N_FOLDS,
) -> List[Dict[str, str]]:
    """Delegate to shared library implementation."""
    return _lib_generate_cv_folds(csv_path=csv_path, n_folds=n_folds)


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
    with _csv_lock:
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CV_RESULTS_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Single-combo worker
# ---------------------------------------------------------------------------


def _run_fold_combo(
    fold_idx: int,
    fold_strains: Dict[str, str],
    env_val: Optional[str],
    agg: str,
    k: int,
    extractor: Any,
    extractor_id: str,
    coll: str,
    run_output_dir: Path,
    segmented_image_dir: str,
) -> Tuple[tuple, List[dict], float, str, str, int]:
    """
    Execute one (env, agg, k) combination for a fold.

    Each call creates its own QdrantClient to avoid connection sharing across
    threads. The feature extractor is shared (PyTorch inference in eval mode
    is thread-safe under torch.no_grad).

    Returns (key, rows, accuracy, env_label, agg, k).
    """
    from src.analysis.visualization.visualize_prediction import (
        batch_visualize_predictions,
    )
    from src.experiments.retrieval.run import run_species_evaluation

    env_label = "E1" if env_val is None else "E2"
    key = (fold_idx, env_label, agg, k, extractor_id, coll)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    fold_out = run_output_dir / f"fold{fold_idx}_{env_label}_{agg}_k{k}"
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
        generate_visualizations=False,
        selected_strains=fold_strains,
    )

    # Generate visualizations for ALL cases (correct + incorrect)
    if results:
        viz_dir = str(fold_out / "visualizations")
        batch_visualize_predictions(
            prediction_results=results,
            segmented_image_dir=segmented_image_dir,
            output_dir=viz_dir,
            k=k,
            filter_correct=None,  # all cases
        )

    rows = [
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
        for res in results
    ]

    acc = (
        sum(r.get("correct", False) for r in results) / len(results) if results else 0.0
    )
    return key, rows, acc, env_label, agg, k


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
    max_workers: int = 4,
) -> None:
    from src.experiments.feature_extraction.feature_extractors import (
        EfficientNetB1Extractor,
        EfficientNetB1FinetunedExtractor,
        EfficientNetB1TripletExtractor,
        MobileNetV2Extractor,
        MobileNetV2FinetunedExtractor,
        ResNet50Extractor,
        ResNet50FinetunedExtractor,
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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir or (RESULTS_DIR / "cross_validation")
    run_output_dir.mkdir(parents=True, exist_ok=True)

    folds = generate_cv_folds(n_folds=n_folds)
    completed = _load_completed_runs(CV_RESULTS_CSV)

    total = n_folds * len(env_strategies) * len(agg_strategies) * len(k_values)
    segmented_image_dir = str(SEGMENTED_IMAGE_DIR)

    print(f"Cross-validation: {total} total runs, max_workers={max_workers}")
    print(f"Results → {CV_RESULTS_CSV}")

    with tqdm(total=total, desc="CV runs", unit="run", dynamic_ncols=True) as pbar:
        for fold_idx, fold_strains in enumerate(folds):
            # ---- build extractor for this fold ----
            if use_fold_specific_assets and extractor_key == "efficientnetb1_finetuned":
                fold_weight_path = (
                    Path(weights_dir) / f"fold{fold_idx}_EfficientNetB1_finetuned.pth"
                )
                if not fold_weight_path.exists():
                    raise FileNotFoundError(
                        f"Missing fold-specific weight: {fold_weight_path}. "
                        "Copy fold weights before running CV."
                    )
                extractor = _extractor_map[extractor_key](
                    weights_path=str(fold_weight_path)
                )
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

            # ---- build list of pending combos for this fold ----
            pending: List[Tuple[Optional[str], str, int]] = []
            skip_count = 0
            for env_val in env_strategies:
                env_label = "E1" if env_val is None else "E2"
                for agg in agg_strategies:
                    for k in k_values:
                        key = (fold_idx, env_label, agg, k, extractor_id, coll)
                        if key in completed:
                            skip_count += 1
                        else:
                            pending.append((env_val, agg, k))

            if skip_count:
                pbar.update(skip_count)
                pbar.set_postfix({"fold": fold_idx, "skipped": skip_count})

            if not pending:
                continue

            # ---- run pending combos in parallel ----
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_fold_combo,
                        fold_idx,
                        fold_strains,
                        env_val,
                        agg,
                        k,
                        extractor,
                        extractor_id,
                        coll,
                        run_output_dir,
                        segmented_image_dir,
                    ): (env_val, agg, k)
                    for (env_val, agg, k) in pending
                }

                for future in as_completed(futures):
                    try:
                        key, rows, acc, env_label, agg, k = future.result()
                        _append_rows(CV_RESULTS_CSV, rows)
                        completed.add(key)
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "fold": fold_idx,
                                "env": env_label,
                                "agg": agg,
                                "k": k,
                                "acc": f"{acc:.3f}",
                            }
                        )
                    except Exception as exc:
                        env_val_f, agg_f, k_f = futures[future]
                        env_label_f = "E1" if env_val_f is None else "E2"
                        tqdm.write(
                            f"  ✗ fold={fold_idx} env={env_label_f} agg={agg_f} k={k_f} "
                            f"failed: {exc}"
                        )
                        pbar.update(1)

    print(f"\nCross-validation complete.  {total} runs processed.")
    print(f"Results: {CV_RESULTS_CSV}")

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
    print("\nTop 10 configurations:")
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
    max_workers: int = 4,
) -> None:
    run_cross_validation(
        collection_name=collection,
        extractor_key=extractor,
        use_fold_specific_assets=use_fold_specific_assets,
        collection_template=collection_template,
        weights_dir=weights_dir,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()


@_dc_autolab
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@_dc_autolab
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: _List_autolab[str]
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    """Uniform experiment contract wrapper. Scoped to params.output_root."""
    import json as _json_autolab
    from pathlib import Path as _Path_autolab
    output_root = _Path_autolab(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = params.description[:30] if params.description else "cross_validation"
    result_data = {"f1_score": 0.0, "strategy_name": strategy, "artifact_paths": [], "run_id": params.run_id}
    (output_root / "results.json").write_text(_json_autolab.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
