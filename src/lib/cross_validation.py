"""
Shared Cross-Validation Library
=================================
Reusable K-fold strain-level cross-validation logic for all retrieval experiments.

Usage:
    from src.lib.cross_validation import run_cross_validation, generate_cv_folds

    # Run 5-fold CV and get per-fold results with correct/incorrect cases
    results = run_cross_validation(
        collection_name="myco_fungi_features_full_finetuned",
        extractor_key="efficientnetb1_finetuned",
        k=11,
        environment=None,        # E1: same environment
        strategy="weighted",     # score-weighted
        n_folds=5,
    )

    # results is a dict with per-fold results, each containing:
    # {
    #   "fold": 0,
    #   "strain": "DTO 123-A1",
    #   "ground_truth": "Penicillium commune",
    #   "predicted_species": "Penicillium commune",
    #   "correct": True,
    #   "test_set_index": 0,
    #   "neighbors": [...],   # for debugging
    # }

For experiment runners, the main result is the mean accuracy:
    mean_acc = sum(r["correct"] for r in results) / len(results)
"""

from __future__ import annotations

import csv
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from qdrant_client import QdrantClient

from src.config import (
    QDRANT_API_KEY,
    QDRANT_URL,
    STRAIN_SPECIES_MAPPING_PATH,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FOLDS = 5
DEFAULT_K = 11
DEFAULT_STRATEGY = "weighted"  # score-weighted aggregation

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

# Thread-safe CSV write lock
_csv_lock = threading.Lock()


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
    in later folds.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Strain mapping CSV not found at {csv_path}. "
            "Run 'uv run python -m src.utils.generate_strain_mapping' first."
        )

    df = pd.read_csv(csv_path)
    species_to_strains: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        species_to_strains[row["Species"]].append(row["Strain"])

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
    with csv.DictReader(open(csv_path)) as reader:
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
# Retrieval helpers (imported from retrieval experiment)
# ---------------------------------------------------------------------------


def _get_all_images_for_strain(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all points for one strain (optionally filtered by environment)."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    conditions = [FieldCondition(key="strain", match=MatchValue(value=strain))]
    if environment and environment.lower() != "all":
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )
    search_filter = Filter(must=conditions)

    all_images: List[Dict[str, Any]] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            payload = point.payload
            all_images.append(
                {
                    "image_id": payload.get("image_id"),
                    "strain": payload.get("strain"),
                    "environment": payload.get("environment"),
                    "angle": payload.get("angle"),
                    "specy": payload.get("specy"),
                    "parent_id": payload.get("parent_id"),
                    "segment_index": payload.get("segment_index"),
                    "bbox": payload.get("bbox"),
                }
            )
        if next_offset is None:
            break
        offset = next_offset
    return all_images


def _filter_siblings(
    neighbors: List[Dict[str, Any]], query_parent_id: str
) -> List[Dict[str, Any]]:
    """Remove neighbors from the same parent image as the query segment."""
    return [n for n in neighbors if n.get("parent_id") != query_parent_id]


def _aggregate_predictions(
    raw_results: List[Dict[str, Any]],
    strain_to_specy: Dict[str, str],
    k: int,
    strategy: str = "weighted",
) -> List[Tuple[str, float]]:
    """Aggregate per-segment neighbours into species ranking."""
    from collections import Counter

    species_scores = Counter()
    species_counts = Counter()

    for result in raw_results:
        for neighbor in result["neighbors"]:
            specy = neighbor.get("specy")
            score = neighbor.get("score", 0.0)
            if not specy or specy == "unknown":
                strain = neighbor.get("strain")
                if strain:
                    specy = strain_to_specy.get(strain, "unknown")
            if specy and specy != "unknown":
                species_scores[specy] += score
                species_counts[specy] += 1

    total_neighbors = sum(species_counts.values())
    aggregated: List[Tuple[str, float]] = []
    for specy, total_score in species_scores.items():
        if strategy == "weighted":
            final_score = total_score / total_neighbors if total_neighbors > 0 else 0.0
        elif strategy == "uni":
            count = species_counts[specy]
            final_score = count / total_neighbors if total_neighbors > 0 else 0.0
        else:
            final_score = float(total_score)
        aggregated.append((specy, final_score))

    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated


def _collect_testset(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment: Optional[str],
) -> List[List[Dict[str, Any]]]:
    """Build per-strain test sets (one image per environment)."""
    strain_images = _get_all_images_for_strain(
        client=client,
        collection_name=collection_name,
        strain=strain,
        environment=(
            environment if environment and not environment.startswith("E") else None
        ),
    )

    if not strain_images:
        return []

    env_segment_angle_images: Dict[str, Dict[Any, Dict[Any, List[Dict[str, Any]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for img in strain_images:
        env = img.get("environment", "unknown")
        seg_idx = img.get("segment_index", 0)
        angle = img.get("angle", "unknown")
        env_segment_angle_images[env][seg_idx][angle].append(img)

    test_sets: List[List[Dict[str, Any]]] = []
    used_per_env: Dict[str, set] = defaultdict(set)
    test_configs = [
        (0, "ob"),
        (0, "rev"),
        (1, "ob"),
        (1, "rev"),
        (2, "ob"),
        (2, "rev"),
    ]

    for segment_idx, preferred_angle in test_configs:
        test_set: List[Dict[str, Any]] = []
        for env in sorted(env_segment_angle_images.keys()):
            seg_imgs = env_segment_angle_images[env]
            img_selected: Optional[Dict[str, Any]] = None

            if segment_idx in seg_imgs:
                angle_vars = {"ob": ["ob", "obverse"], "rev": ["rev", "reverse"]}
                for angle_var in angle_vars.get(preferred_angle, [preferred_angle]):
                    if (
                        angle_var in seg_imgs[segment_idx]
                        and seg_imgs[segment_idx][angle_var]
                    ):
                        for candidate in seg_imgs[segment_idx][angle_var]:
                            key = (
                                candidate["image_id"],
                                candidate.get("angle", "unknown"),
                            )
                            if key not in used_per_env[env]:
                                img_selected = candidate
                                break
                        if img_selected:
                            break
                if not img_selected:
                    for angle in sorted(seg_imgs[segment_idx].keys()):
                        for candidate in seg_imgs[segment_idx][angle]:
                            key = (
                                candidate["image_id"],
                                candidate.get("angle", "unknown"),
                            )
                            if key not in used_per_env[env]:
                                img_selected = candidate
                                break
                        if img_selected:
                            break

            if img_selected:
                test_set.append(img_selected)
                key = (img_selected["image_id"], img_selected.get("angle", "unknown"))
                used_per_env[env].add(key)
            else:
                break

        if test_set and len(test_set) == len(env_segment_angle_images):
            test_sets.append(test_set)

    return test_sets


def _predict_segment_group(
    client: QdrantClient,
    collection_name: str,
    test_group: List[Dict[str, Any]],
    strain: str,
    extractor_name: str,
    k: int,
    environment: Optional[str],
    strategy: str,
    strain_to_specy: Dict[str, str],
) -> Dict[str, Any]:
    """Predict species from a test segment group; returns per-result dict."""
    from src.utils.qdrant_query import find_nearest_neighbors_by_id

    ground_truth = strain_to_specy.get(strain, "unknown")
    raw_results: List[Dict[str, Any]] = []

    for query_img in test_group:
        image_id = query_img["image_id"]
        parent_id = query_img["parent_id"]
        img_env = query_img.get("environment", "unknown")

        search_env = (
            environment if environment and environment.lower() != "all" else None
        )

        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=extractor_name,
            num_neighbors=k * 10,
            environment=search_env,
            exclude_self=True,
            exclude_strain=strain,
        )
        neighbors = _filter_siblings(neighbors, parent_id)
        neighbors = neighbors[:k]
        raw_results.append(
            {
                "query_image_id": image_id,
                "query_environment": img_env,
                "neighbors": neighbors,
            }
        )

    aggregated = _aggregate_predictions(raw_results, strain_to_specy, k, strategy)

    predicted_species = aggregated[0][0] if aggregated else "unknown"
    confidence = aggregated[0][1] if aggregated else 0.0
    is_correct = predicted_species == ground_truth

    return {
        "strain": strain,
        "ground_truth": ground_truth,
        "predicted_species": predicted_species,
        "correct": is_correct,
        "confidence": confidence,
        "aggregated": [{"specy": s, "score": sc} for s, sc in aggregated],
        "raw_results": raw_results,
        "extractor": extractor_name,
        "strategy": strategy,
        "environment": environment,
    }


# ---------------------------------------------------------------------------
# Per-fold worker
# ---------------------------------------------------------------------------


def _run_fold(
    fold_idx: int,
    fold_strains: Dict[str, str],
    collection_name: str,
    extractor_key: str,
    k: int,
    environment: Optional[str],
    strategy: str,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run one CV fold and return per-strain result dicts with correct/incorrect.
    """
    from src.experiments.feature_extraction.feature_extractors import (
        EfficientNetB1FinetunedExtractor,
    )

    extractor_map = {
        "efficientnetb1_finetuned": EfficientNetB1FinetunedExtractor,
    }
    extractor_cls = extractor_map.get(extractor_key, EfficientNetB1FinetunedExtractor)
    extractor = extractor_cls()
    extractor_name = extractor.name

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    df = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    strain_to_specy = dict(zip(df["Strain"], df["Species"]))

    env_label = "E1" if environment is None else "E2"
    results: List[Dict[str, Any]] = []

    for species, strain in fold_strains.items():
        test_sets = _collect_testset(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=environment,
        )

        for i, test_group in enumerate(test_sets):
            res = _predict_segment_group(
                client=client,
                collection_name=collection_name,
                test_group=test_group,
                strain=strain,
                extractor_name=extractor_name,
                k=k,
                environment=environment,
                strategy=strategy,
                strain_to_specy=strain_to_specy,
            )
            res["fold"] = fold_idx
            res["species"] = species
            res["test_set_index"] = i
            res["env_strategy"] = env_label
            res["agg_strategy"] = strategy
            res["k"] = k
            res["extractor"] = extractor_name
            res["collection"] = collection_name
            results.append(res)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_cross_validation(
    collection_name: str = "myco_fungi_features_full_finetuned",
    extractor_key: str = "efficientnetb1_finetuned",
    k: int = DEFAULT_K,
    environment: Optional[str] = None,  # None = E1 (same env)
    strategy: str = DEFAULT_STRATEGY,  # "weighted" or "uni"
    n_folds: int = N_FOLDS,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Run K-fold cross-validation and return per-query result dicts.

    Each result dict contains:
      - fold, species, strain, ground_truth, predicted_species, correct,
        test_set_index, env_strategy, agg_strategy, k, extractor, collection

    Mean accuracy:
      >>> results = run_cross_validation(...)
      >>> mean_acc = sum(r["correct"] for r in results) / len(results)
    """
    folds = generate_cv_folds(n_folds=n_folds)
    all_results: List[Dict[str, Any]] = []

    for fold_idx, fold_strains in enumerate(folds):
        print(f"Running fold {fold_idx + 1}/{n_folds} ...")
        fold_results = _run_fold(
            fold_idx=fold_idx,
            fold_strains=fold_strains,
            collection_name=collection_name,
            extractor_key=extractor_key,
            k=k,
            environment=environment,
            strategy=strategy,
            max_workers=max_workers,
        )
        all_results.extend(fold_results)
        fold_acc = (
            sum(r["correct"] for r in fold_results) / len(fold_results)
            if fold_results
            else 0
        )
        print(
            f"  Fold {fold_idx}: {fold_acc:.3f} ({len([r for r in fold_results if r['correct']])}/{len(fold_results)})"
        )

    return all_results


def compute_mean_accuracy(results: List[Dict[str, Any]]) -> float:
    """Compute mean accuracy from per-query result list."""
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_cv_results(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Append cross-validation results to a CSV file."""
    if not results:
        return
    write_header = not output_path.exists()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CV_RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "fold": r["fold"],
                    "species": r.get("species", ""),
                    "strain": r.get("strain", ""),
                    "ground_truth": r["ground_truth"],
                    "predicted_specy": r["predicted_species"],
                    "correct": int(r["correct"]),
                    "test_set_index": r["test_set_index"],
                    "env_strategy": r.get("env_strategy", "E1"),
                    "agg_strategy": r.get("agg_strategy", "weighted"),
                    "k": r.get("k", DEFAULT_K),
                    "extractor": r.get("extractor", ""),
                    "collection": r.get("collection", ""),
                }
            )
