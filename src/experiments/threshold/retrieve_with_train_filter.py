"""
Retrieve diverse_data images from Qdrant, EXCLUDING test strains (train=False).

Uses preprocessed images from diverse_data, queries Qdrant with train=True filter,
and writes results with known=True for test strains and known=False for others.

Usage:
    uv run python -m src.experiments.threshold.retrieve_with_train_filter
    uv run python -m src.experiments.threshold.retrieve_with_train_filter --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os  # noqa: E402

import cv2  # noqa: E402

from src.config import QDRANT_API_KEY, QDRANT_URL, RESULTS_DIR, WORKSPACE_ROOT  # noqa: E402
from src.experiments.feature_extraction.feature_extractors import (  # noqa: E402
    EfficientNetB1FinetunedExtractor,
)

_qdrant_url = QDRANT_URL
if not os.getenv("QDRANT_URL") or "cloud.qdrant" in _qdrant_url:
    _qdrant_url = "http://localhost:6333"


try:
    from qdrant_client import QdrantClient  # noqa: E402
    from qdrant_client.models import (  # noqa: E402
        FieldCondition,
        Filter,
        MatchValue,
    )
except ImportError:
    print("ERROR: qdrant_client not installed.")
    sys.exit(1)

COLLECTION = "myco_fungi_features_full_finetuned"
EXTRACTOR_KEY = "EfficientNetB1_finetuned"
K = 11
TOP_N = 5

RETRIEVAL_LIST = RESULTS_DIR / "threshold" / "test_strain_retrieval_list.json"
OUTPUT_CSV = RESULTS_DIR / "threshold" / "diverse_retrieval_results.csv"

# Test strains to EXCLUDE from Qdrant neighbors
TEST_STRAINS = {
    "DTO 217-D9",
    "DTO 470-I9",
    "DTO 158-D1",
    "DTO 148-D1",
    "DTO 469-I5",
    "DTO 469-I4",
    "DTO 163-I2",
}

CSV_FIELDS = (
    ["sample_id", "species_label", "is_known", "environment", "angle"]
    + [f"s{i}_score" for i in range(TOP_N)]
    + [f"s{i}_species" for i in range(TOP_N)]
    + ["predicted_species", "correct_species", "image_path"]
)


def aggregate_weighted(neighbors):
    totals = {}
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


def load_done_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {row["sample_id"] for row in csv.DictReader(f)}


def run_retrieval(resume: bool = False, limit: int | None = None):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load retrieval list
    with open(RETRIEVAL_LIST) as f:
        retrieval_list = json.load(f)
    print(f"Loaded {len(retrieval_list)} images for retrieval")

    done_ids = load_done_ids(OUTPUT_CSV) if resume else set()
    if resume and done_ids:
        print(f"Resuming: {len(done_ids)} already done, skipping")

    # Connect Qdrant
    client = QdrantClient(url=_qdrant_url, api_key=QDRANT_API_KEY, timeout=30)
    try:
        info = client.get_collection(COLLECTION)
        print(f"Qdrant collection '{COLLECTION}': {info.points_count} points")
    except Exception as e:
        print(f"ERROR accessing Qdrant: {e}")
        sys.exit(1)

    extractor = EfficientNetB1FinetunedExtractor()

    mode = "a" if resume and OUTPUT_CSV.exists() else "w"
    with open(OUTPUT_CSV, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()

        processed = 0
        skipped = 0

        for entry in retrieval_list:
            if limit and processed >= limit:
                break

            img_id = entry["id"]
            if resume and img_id in done_ids:
                skipped += 1
                continue

            data = entry.get("data", {})
            species_label = data.get("species", "UNKNOWN")
            environment = data.get("environment", "UNKNOWN")
            angle = data.get("angle", "UNKNOWN")
            is_known = int(entry.get("is_known", 0))

            # Image path: use step_images.preprocessed or file_path
            step_images = entry.get("step_images", {})
            img_path_rel = step_images.get("preprocessed") or entry.get("file_path")
            if not img_path_rel:
                print(f"  SKIP {img_id}: no image path")
                continue
            img_path = WORKSPACE_ROOT / img_path_rel
            if not img_path.exists():
                print(f"  SKIP {img_id}: image not found: {img_path}")
                continue

            # Load and extract features
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  SKIP {img_id}: cannot read image")
                continue
            try:
                features = extractor.extract(img_bgr)
            except Exception as exc:
                print(f"  SKIP {img_id}: extraction failed: {exc}")
                continue

            # E1 filter: same environment
            env_filter = None
            if environment not in ("UNKNOWN", ""):
                env_filter = Filter(
                    must=[
                        FieldCondition(
                            key="environment",
                            match=MatchValue(value=environment),
                        )
                    ]
                )

            # Query Qdrant
            try:
                results = client.query_points(
                    collection_name=COLLECTION,
                    query=features.tolist(),
                    using=EXTRACTOR_KEY,
                    query_filter=env_filter,
                    limit=K,
                    with_payload=True,
                )
            except Exception as exc:
                print(f"  SKIP {img_id}: Qdrant query failed: {exc}")
                continue

            # Build neighbor list, filtering out test strains
            neighbors = []
            for hit in results.points:
                payload = hit.payload or {}
                neigh_strain = payload.get("strain", "")
                if neigh_strain in TEST_STRAINS:
                    continue  # Exclude test strains from neighbors
                neighbors.append(
                    {
                        "specy": payload.get("specy", "unknown"),
                        "score": hit.score,
                        "strain": neigh_strain,
                        "environment": payload.get("environment", ""),
                    }
                )

            ranked = aggregate_weighted(neighbors)

            row = {
                "sample_id": img_id,
                "species_label": species_label,
                "is_known": is_known,
                "environment": environment,
                "angle": angle,
                "predicted_species": ranked[0]["species"] if ranked else "unknown",
                "correct_species": species_label,
                "image_path": str(img_path_rel),
            }
            for i in range(TOP_N):
                if i < len(ranked):
                    row[f"s{i}_score"] = f"{ranked[i]['score']:.6f}"
                    row[f"s{i}_species"] = ranked[i]["species"]
                else:
                    row[f"s{i}_score"] = ""
                    row[f"s{i}_species"] = ""

            writer.writerow(row)
            csvfile.flush()

            status = "KNOWN" if is_known else "unknown"
            correct_mark = (
                " OK"
                if (is_known and ranked and ranked[0]["species"] == species_label)
                else ""
            )
            s0 = float(row["s0_score"]) if row["s0_score"] else 0.0
            print(
                f"  [{status}]{correct_mark} {species_label}/{environment} "
                f"-> {row['predicted_species']} (s0={s0:.4f}, {len(neighbors)} filtered neighbors)"
            )
            processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Results saved to: {OUTPUT_CSV}")
    return OUTPUT_CSV


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve diverse data excluding test strains"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-processed IDs"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    args = parser.parse_args()
    run_retrieval(resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
