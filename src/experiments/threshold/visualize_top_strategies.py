"""
Visualize threshold results for top strategies using visualize_prediction_by_environment.

For each top strategy, build prediction_result dicts per sample and generate
per-environment neighbor visualizations showing TP/FP/TN/FN classification.

Usage:
    uv run python -m src.experiments.threshold.visualize_top_strategies
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_ROOT, RESULTS_DIR  # noqa: E402

INPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"
OUTPUT_DIR = RESULTS_DIR / "threshold" / "strategy_viz"
SEGMENTED_IMAGE_DIR = DATASET_ROOT / "diverse_data" / "images"

# Top strategies from expanded analysis
TOP_STRATEGIES = [
    ("geom_mean_top3", "roc_opt", 0.302755, 0.163934),
    ("random_lin_058", "f1_grid", 0.194752, 0.142857),
    ("harm_mean_top3", "f1_grid", 0.273732, 0.128205),
]


def load_csv() -> List[Dict[str, Any]]:
    with open(INPUT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def compute_gm3(scores: np.ndarray) -> float:
    """Geometric mean of top-3 scores (works on 1D or 2D)."""
    eps = 1e-12
    if scores.ndim == 2:
        return np.power(scores[:, 0] * scores[:, 1] * scores[:, 2] + eps, 1.0 / 3.0)
    return float(np.power(scores[0] * scores[1] * scores[2] + eps, 1.0 / 3.0))


def compute_harm3(scores: np.ndarray) -> float:
    """Harmonic mean of top-3 with floor (works on 1D or 2D)."""
    eps = 1e-12
    if scores.ndim == 2:
        return 3.0 / (
            1.0 / (scores[:, 0] + eps)
            + 1.0 / (np.maximum(scores[:, 1], eps))
            + 1.0 / (np.maximum(scores[:, 2], eps))
        )
    s = scores
    return 3.0 / (1.0 / (s[0] + eps) + 1.0 / max(s[1], eps) + 1.0 / max(s[2], eps))


def compute_random_lin_058(scores: np.ndarray) -> float:
    """Fixed random linear weights from expanded analysis (1D or 2D)."""
    rng = np.random.RandomState(58)
    raw = rng.random(5)
    total = raw.sum() or 1.0
    w = raw / total
    if scores.ndim == 2:
        return np.sum(scores * w, axis=1)
    return float(np.dot(scores, w))


def classify_sample(scores_1d: np.ndarray, strategy: str, threshold: float) -> int:
    """Return binary known/unknown prediction for a single sample."""
    if strategy == "geom_mean_top3":
        s = compute_gm3(scores_1d)
    elif strategy == "harm_mean_top3":
        s = compute_harm3(scores_1d)
    elif strategy.startswith("random_lin"):
        s = compute_random_lin_058(scores_1d)
    else:
        s = scores_1d[0]
    return int(s >= threshold)


def build_prediction_result(
    row: Dict[str, Any],
    strategy: str,
    threshold: float,
) -> Dict[str, Any]:
    """Build a prediction_result dict for one sample."""
    scores = np.array(
        [
            float(row["s0_score"]) if row["s0_score"] else 0.0,
            float(row["s1_score"]) if row["s1_score"] else 0.0,
            float(row["s2_score"]) if row["s2_score"] else 0.0,
            float(row["s3_score"]) if row["s3_score"] else 0.0,
            float(row["s4_score"]) if row["s4_score"] else 0.0,
        ]
    )

    gm3 = compute_gm3(scores)

    # Neighbors: top-5 from the CSV columns
    neighbors = []
    for i in range(5):
        sp = row.get(f"s{i}_species", "")
        sc = row.get(f"s{i}_score", "")
        if sp:
            neighbors.append(
                {
                    "specy": sp,
                    "score": float(sc) if sc else 0.0,
                }
            )

    ground_truth = row["species_label"]
    # If known: correct if predicted == correct_species
    is_known = int(row["is_known"]) == 1
    correct_species = row.get("correct_species", "")
    predicted = row["predicted_species"] or "unknown"

    correct = is_known and predicted == correct_species

    # Threshold decision
    if strategy == "geom_mean_top3":
        score_val = gm3
    else:
        score_val = scores[0]

    return {
        "ground_truth": ground_truth,
        "predicted_specy": predicted,
        "correct": correct,
        "predicted_confidence": score_val,
        "feature_extractor": "EfficientNetB1_finetuned",
        "strategy": strategy,
        "environment": row["environment"],
        "strain": row["species_label"],
        "sample_id": row["sample_id"],
        "is_known": is_known,
        "threshold": threshold,
        "raw_results": [
            {
                "query_image_id": row["sample_id"],
                "query_environment": row["environment"],
                "neighbors": neighbors,
            }
        ],
        "aggregated_results": [
            {"specy": row[f"s{i}_species"], "score": float(row[f"s{i}_score"])}
            for i in range(5)
            if row.get(f"s{i}_species")
        ],
    }


def run_visualization():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_csv()
    n = len(rows)

    # Pre-compute score matrix
    all_scores = np.zeros((n, 5))
    for i, r in enumerate(rows):
        for j in range(5):
            v = r.get(f"s{j}_score", "")
            all_scores[i, j] = float(v) if v else 0.0

    from src.analysis.visualization.visualize_prediction import (
        visualize_prediction_by_environment,
    )

    for strategy, algo, threshold, f1 in TOP_STRATEGIES:
        print(f"\nVisualizing: {strategy}_{algo} (F1={f1:.4f})")

        # Build prediction results for all samples
        pred_results = []
        for i, row in enumerate(rows):
            scores_1d = all_scores[i]
            pr = build_prediction_result(row, strategy, threshold)
            pr["threshold_decision"] = classify_sample(scores_1d, strategy, threshold)
            pred_results.append(pr)

        # Group by environment and create per-env visualizations
        from collections import defaultdict

        by_env = defaultdict(list)
        for pr in pred_results:
            by_env[pr["environment"]].append(pr)

        # Create per-environment charts
        for env, env_results in sorted(by_env.items()):
            if not env_results:
                continue

            # Filter to unique samples per env (avoid duplicates)
            seen = set()
            unique_results = []
            for pr in env_results:
                if pr["sample_id"] not in seen:
                    seen.add(pr["sample_id"])
                    unique_results.append(pr)

            # Create environment-level prediction result
            env_pred_result = {
                "ground_truth": unique_results[0]["ground_truth"],
                "predicted_specy": unique_results[0]["predicted_specy"],
                "correct": unique_results[0]["correct"],
                "predicted_confidence": np.mean(
                    [r["predicted_confidence"] for r in unique_results]
                ),
                "feature_extractor": "EfficientNetB1_finetuned",
                "strategy": f"{strategy}_{env}",
                "environment": env,
                "strain": unique_results[0]["strain"],
                "raw_results": [
                    {
                        "query_image_id": r["sample_id"],
                        "query_environment": env,
                        "neighbors": r["raw_results"][0]["neighbors"],
                    }
                    for r in unique_results
                ],
                "aggregated_results": unique_results[0]["aggregated_results"],
            }

            output_path = OUTPUT_DIR / f"{strategy}_{algo}_{env}.jpg"
            try:
                visualize_prediction_by_environment(
                    prediction_result=env_pred_result,
                    segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                    output_path=str(output_path),
                    k=min(7, 5),
                )
            except Exception as e:
                print(f"  Warning: could not visualize {env}: {e}")

        print(f"  Saved visualizations for {strategy}_{algo}")

    print(f"\nAll visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_visualization()
