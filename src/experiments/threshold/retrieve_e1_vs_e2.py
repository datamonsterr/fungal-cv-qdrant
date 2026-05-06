"""
Experiment attempt 4: E2 (all environments) + different k values

Hypothesis: Some known species may score higher cross-environment.
k=3 may give better top-3 gm than k=11 (less dilution from weaker matches).

Strategy: E2, k=3 and k=5, weighted aggregation.
If any known samples improve their top-3 gm significantly, the threshold ceiling rises.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os  # noqa: E402

import cv2  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from src.config import (  # noqa: E402
    DATASET_ROOT,
    QDRANT_API_KEY,
    QDRANT_URL,
    RESULTS_DIR,
    WORKSPACE_ROOT,
)

_qdrant_url = QDRANT_URL
if not os.getenv("QDRANT_URL") or "cloud.qdrant" in _qdrant_url:
    _qdrant_url = "http://localhost:6333"

from src.experiments.feature_extraction.feature_extractors import (  # noqa: E402
    EfficientNetB1FinetunedExtractor,
)

DIVERSE_METADATA_PATH = DATASET_ROOT / "diverse_data" / "diverse_data_metadata.json"
COLLECTION = "myco_fungi_features_full_finetuned"
EXTRACTOR_KEY = "EfficientNetB1_finetuned"

KNOWN_SPECIES_MAP = {
    "commune": "Penicillium commune",
    "crustosum": "Penicillium crustosum",
    "expansum": "Penicillium expansum",
    "chrysogenum": "Penicillium chrysogenum",
    "citreonigrum": "Penicillium citreonigrum",
    "Penicillium commune": "Penicillium commune",
    "Penicillium crustosum": "Penicillium crustosum",
    "Penicillium expansum": "Penicillium expansum",
    "Penicillium chrysogenum": "Penicillium chrysogenum",
    "Penicillium citreonigrum": "Penicillium citreonigrum",
}


def is_known(species_label: str) -> bool:
    label = species_label.strip().lower()
    for key in KNOWN_SPECIES_MAP:
        if label == key.lower():
            return True
    return False


def map_to_db(species_label: str):
    label = species_label.strip()
    for key, val in KNOWN_SPECIES_MAP.items():
        if label.lower() == key.lower():
            return val
    return None


def aggregate_weighted(neighbors: List[Dict]) -> List[Dict]:
    totals: Dict[str, float] = {}
    for n in neighbors:
        sp = n.get("specy") or n.get("species", "unknown")
        score = float(n.get("score", 0.0))
        totals[sp] = totals.get(sp, 0.0) + score
    total_weight = sum(totals.values()) or 1.0
    return sorted(
        [{"species": sp, "score": s / total_weight} for sp, s in totals.items()],
        key=lambda x: x["score"],
        reverse=True,
    )


def retrieve_with_k(client, extractor, img_path, k, env_filter=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    features = extractor.extract(img)
    results = client.query_points(
        collection_name=COLLECTION,
        query=features.tolist(),
        using=EXTRACTOR_KEY,
        query_filter=env_filter,
        limit=k,
        with_payload=True,
    )
    neighbors = [
        {"specy": h.payload.get("specy", "unknown"), "score": h.score}
        for h in results.points
    ]
    return neighbors


def run():
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    with open(DIVERSE_METADATA_PATH) as f:
        metadata = json.load(f)
    images = metadata.get("images", [])

    # Filter to known species only (we only care about checking their scores)
    known_images = [
        img for img in images if is_known(img.get("data", {}).get("species", ""))
    ]
    print(f"Known samples: {len(known_images)}")

    client = QdrantClient(url=_qdrant_url, api_key=QDRANT_API_KEY, timeout=30)
    extractor = EfficientNetB1FinetunedExtractor()

    # For each known sample, compare E1 vs E2 at k=3, k=5, k=11
    results_rows = []
    for img_info in known_images:
        data = img_info.get("data", {})
        species = data.get("species", "")
        environment = data.get("environment", "")
        img_path_rel = img_info.get("step_images", {}).get(
            "preprocessed"
        ) or img_info.get("file_path", "")
        if not img_path_rel:
            continue
        img_path = WORKSPACE_ROOT / img_path_rel
        if not img_path.exists():
            continue

        db_species = map_to_db(species)
        step_images = img_info.get("step_images", {})
        original_rel = step_images.get("original") or img_info.get("file_path", "")
        original_path = WORKSPACE_ROOT / original_rel
        if not original_path.exists():
            continue

        # E1 filter
        env_filter = None
        if environment not in ("UNKNOWN", ""):
            env_filter = Filter(
                must=[
                    FieldCondition(
                        key="environment", match=MatchValue(value=environment)
                    )
                ]
            )

        # Retrieve with E1 (same env) at k=3, k=5, k=11
        for k in [3, 5, 11]:
            neighbors_e1 = retrieve_with_k(
                client, extractor, img_path, k=k, env_filter=env_filter
            )
            if neighbors_e1 is None:
                continue
            ranked_e1 = aggregate_weighted(neighbors_e1)

            # Retrieve with E2 (no filter) at k=3, k=5, k=11
            neighbors_e2 = retrieve_with_k(
                client, extractor, img_path, k=k, env_filter=None
            )
            ranked_e2 = aggregate_weighted(neighbors_e2)

            # Geometric mean top-3 for both
            gm3_e1 = 0.0
            if len(ranked_e1) >= 3:
                s = [ranked_e1[i]["score"] for i in range(3)]
                gm3_e1 = (s[0] * s[1] * s[2]) ** (1 / 3)
            elif len(ranked_e1) >= 2:
                gm3_e1 = (ranked_e1[0]["score"] * ranked_e1[1]["score"]) ** 0.5

            gm3_e2 = 0.0
            if len(ranked_e2) >= 3:
                s = [ranked_e2[i]["score"] for i in range(3)]
                gm3_e2 = (s[0] * s[1] * s[2]) ** (1 / 3)
            elif len(ranked_e2) >= 2:
                gm3_e2 = (ranked_e2[0]["score"] * ranked_e2[1]["score"]) ** 0.5

            pred_e1 = ranked_e1[0]["species"] if ranked_e1 else "unknown"
            pred_e2 = ranked_e2[0]["species"] if ranked_e2 else "unknown"

            correct_e1 = pred_e1 == db_species
            correct_e2 = pred_e2 == db_species

            results_rows.append(
                {
                    "species": species,
                    "environment": environment,
                    "db_species": db_species,
                    "k": k,
                    "pred_e1": pred_e1,
                    "pred_e2": pred_e2,
                    "correct_e1": correct_e1,
                    "correct_e2": correct_e2,
                    "gm3_e1": gm3_e1,
                    "gm3_e2": gm3_e2,
                    "s0_e1": ranked_e1[0]["score"] if ranked_e1 else 0.0,
                    "s0_e2": ranked_e2[0]["score"] if ranked_e2 else 0.0,
                }
            )

    # Save full comparison
    output_path = RESULTS_DIR / "threshold" / "e1_vs_e2_comparison.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        writer.writerows(results_rows)

    # Summary
    print(f"\nSaved comparison to {output_path}")
    print()
    print(
        f"{'species':12s} {'env':6s} {'k':3s} {'correct_E1':10s} {'correct_E2':10s} {'gm3_E1':8s} {'gm3_E2':8s} {'s0_E1':8s} {'s0_E2':8s}"
    )
    print("-" * 90)
    for r in results_rows:
        print(
            f"{r['species']:12s} {r['environment']:6s} {r['k']:3d} {str(r['correct_e1']):10s} {str(r['correct_e2']):10s} {r['gm3_e1']:8.4f} {r['gm3_e2']:8.4f} {r['s0_e1']:8.4f} {r['s0_e2']:8.4f}"
        )

    # Count improvements
    e1_correct = sum(1 for r in results_rows if r["correct_e1"])
    e2_correct = sum(1 for r in results_rows if r["correct_e2"])
    print(f"\nE1 correct (k=11): {e1_correct}/{len(known_images)}")
    print(f"E2 correct (k=11): {e2_correct}/{len(known_images)}")

    return output_path


if __name__ == "__main__":
    run()
