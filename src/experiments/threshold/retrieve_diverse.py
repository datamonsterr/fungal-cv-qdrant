"""
Step 1: Retrieve diverse dataset images against Qdrant and record similarity scores.

Reads Dataset/diverse_data/diverse_data_metadata.json, runs each preprocessed image
through EfficientNetB1_finetuned retrieval (E1, weighted, k=11), and writes:

    results/threshold/diverse_retrieval_results.csv

Each row = one image from the diverse dataset with columns:
    sample_id, species_label, is_known, environment, angle,
    s0_score, s0_species, s1_score, s1_species, ..., s4_score, s4_species,
    predicted_species, correct_species

Usage:
    uv run python -m src.experiments.threshold.retrieve_diverse
    uv run python -m src.experiments.threshold.retrieve_diverse --limit 50
    uv run python -m src.experiments.threshold.retrieve_diverse --resume   # skip already done
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.config import DATASET_ROOT, QDRANT_URL, QDRANT_API_KEY
from src.experiments.feature_extraction.feature_extractors import EfficientNetB1FinetunedExtractor

# Override: use local Docker Qdrant when env var is unset (defaults to cloud URL)
_qdrant_url = QDRANT_URL
# Use localhost when QDRANT_URL is the cloud default (env var is unset)
if not os.getenv("QDRANT_URL") or "cloud.qdrant" in _qdrant_url:
    _qdrant_url = "http://localhost:6333"

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("ERROR: qdrant_client not installed. Run: uv add qdrant-client")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVERSE_METADATA_PATH = DATASET_ROOT / "diverse_data" / "diverse_data_metadata.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "threshold"
OUTPUT_CSV = OUTPUT_DIR / "diverse_retrieval_results.csv"

COLLECTION = "myco_fungi_features_full_finetuned"
EXTRACTOR_KEY = "EfficientNetB1_finetuned"
K = 11
TOP_N_SCORES = 5  # Record s0..s4

# Species in diverse data (without "Penicillium" prefix) that are in our DB
KNOWN_SPECIES_MAP: Dict[str, str] = {
    "commune": "Penicillium commune",
    "crustosum": "Penicillium crustosum",
    "expansum": "Penicillium expansum",
    "chrysogenum": "Penicillium chrysogenum",
    "citreonigrum": "Penicillium citreonigrum",
    # also handle full names if present
    "Penicillium commune": "Penicillium commune",
    "Penicillium crustosum": "Penicillium crustosum",
    "Penicillium expansum": "Penicillium expansum",
    "Penicillium chrysogenum": "Penicillium chrysogenum",
    "Penicillium citreonigrum": "Penicillium citreonigrum",
}

CSV_FIELDS = (
    ["sample_id", "species_label", "is_known", "environment", "angle"]
    + [f"s{i}_score" for i in range(TOP_N_SCORES)]
    + [f"s{i}_species" for i in range(TOP_N_SCORES)]
    + ["predicted_species", "correct_species", "image_path"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_known_species(species_label: str) -> bool:
    """Return True if the species is one of the 5 known Penicillium species."""
    label = species_label.strip().lower()
    for key in KNOWN_SPECIES_MAP:
        if label == key.lower():
            return True
    return False


def map_to_db_species(species_label: str) -> Optional[str]:
    """Map diverse-data species name to the DB species name, or None if unknown."""
    label = species_label.strip()
    for key, val in KNOWN_SPECIES_MAP.items():
        if label.lower() == key.lower():
            return val
    return None


def aggregate_weighted(neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate neighbour scores into a ranked species list (weighted by score).
    Returns list of {"species": ..., "score": ...} sorted descending.
    """
    totals: Dict[str, float] = {}
    for n in neighbors:
        sp = n.get("specy") or n.get("species", "unknown")
        score = float(n.get("score", 0.0))
        totals[sp] = totals.get(sp, 0.0) + score

    # Normalise
    total_weight = sum(totals.values()) or 1.0
    ranked = sorted(
        [{"species": sp, "score": s / total_weight} for sp, s in totals.items()],
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked


def check_qdrant(client: QdrantClient) -> bool:
    """Check if the collection exists and has data."""
    try:
        info = client.get_collection(COLLECTION)
        count = info.points_count
        print(f"  Qdrant collection '{COLLECTION}': {count} points")
        return count > 0
    except Exception as exc:
        print(f"  ERROR: Cannot access Qdrant collection: {exc}")
        return False


def load_done_ids(csv_path: Path) -> set:
    """Load already-processed sample_ids from existing CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return {row["sample_id"] for row in reader}


# ---------------------------------------------------------------------------
# Main retrieval
# ---------------------------------------------------------------------------


def retrieve_diverse(limit: Optional[int] = None, resume: bool = False) -> Path:
    """
    Run retrieval for all diverse-data images. Returns path to output CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    if not DIVERSE_METADATA_PATH.exists():
        print(f"ERROR: Diverse metadata not found: {DIVERSE_METADATA_PATH}")
        print("Run: uv run python src/prepare/reorganize_diverse_data.py --mode pipeline")
        sys.exit(1)

    with open(DIVERSE_METADATA_PATH) as f:
        metadata = json.load(f)

    images = metadata.get("images", [])
    print(f"Loaded {len(images)} images from diverse metadata")

    # Resume: skip already-processed
    done_ids: set = set()
    if resume and OUTPUT_CSV.exists():
        done_ids = load_done_ids(OUTPUT_CSV)
        print(f"Resuming: {len(done_ids)} already processed, skipping")

    if limit:
        images = images[:limit]

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant: {_qdrant_url}")
    client = QdrantClient(url=_qdrant_url, api_key=QDRANT_API_KEY, timeout=30)
    if not check_qdrant(client):
        print("ERROR: Qdrant not available or collection empty.")
        print("Run: docker compose up -d  (and ensure collection is populated)")
        sys.exit(1)

    # Load feature extractor
    print(f"\nLoading extractor: {EXTRACTOR_KEY}")
    extractor = EfficientNetB1FinetunedExtractor()

    # Open CSV for writing (append if resuming)
    mode = "a" if resume and OUTPUT_CSV.exists() else "w"
    with open(OUTPUT_CSV, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        if mode == "w":
            writer.writeheader()

        processed = 0
        skipped = 0

        for entry in images:
            img_id = entry["id"]

            if img_id in done_ids:
                skipped += 1
                continue

            data = entry.get("data", {})
            species_label = data.get("species", "UNKNOWN")
            environment = data.get("environment", "UNKNOWN")
            angle = data.get("angle", "UNKNOWN")
            is_known = is_known_species(species_label)
            db_species = map_to_db_species(species_label)

            # Get the preprocessed image path (step_images.preprocessed)
            step_images = entry.get("step_images", {})
            img_path_rel = step_images.get("preprocessed") or entry.get("file_path")
            if not img_path_rel:
                print(f"  SKIP {img_id}: no image path in metadata")
                continue

            img_path = PROJECT_ROOT / img_path_rel
            if not img_path.exists():
                print(f"  SKIP {img_id}: image not found at {img_path}")
                continue

            # Load image and extract features
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  SKIP {img_id}: cannot read image at {img_path}")
                continue
            try:
                features = extractor.extract(img_bgr)
            except Exception as exc:
                print(f"  SKIP {img_id}: feature extraction failed: {exc}")
                continue

            # E1 strategy: filter by same environment
            from qdrant_client.models import Filter, FieldCondition, MatchValue

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

            # KNN search
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
                print(f"  SKIP {img_id}: Qdrant search failed: {exc}")
                continue

            neighbors = []
            for hit in results.points:
                payload = hit.payload or {}
                neighbors.append(
                    {
                        "specy": payload.get("specy", "unknown"),
                        "score": hit.score,
                        "strain": payload.get("strain", ""),
                        "environment": payload.get("environment", ""),
                    }
                )

            ranked = aggregate_weighted(neighbors)

            row: Dict[str, Any] = {
                "sample_id": img_id,
                "species_label": species_label,
                "is_known": int(is_known),
                "environment": environment,
                "angle": angle,
                "predicted_species": ranked[0]["species"] if ranked else "unknown",
                "correct_species": db_species or "",
                "image_path": str(img_path_rel),
            }

            for i in range(TOP_N_SCORES):
                if i < len(ranked):
                    row[f"s{i}_score"] = f"{ranked[i]['score']:.6f}"
                    row[f"s{i}_species"] = ranked[i]["species"]
                else:
                    row[f"s{i}_score"] = ""
                    row[f"s{i}_species"] = ""

            writer.writerow(row)
            csvfile.flush()

            status = "KNOWN" if is_known else "unknown"
            correct_mark = ""
            if is_known and db_species:
                pred = ranked[0]["species"] if ranked else ""
                correct_mark = " OK" if pred == db_species else " WRONG"
            s0 = float(row["s0_score"]) if row["s0_score"] else 0.0
            print(
                f"  [{status}]{correct_mark} {species_label}/{environment} "
                f"-> {row['predicted_species']} (s0={s0:.4f})"
            )
            processed += 1

    print(f"\nDone. Processed: {processed}, Skipped (resume): {skipped}")
    print(f"Results saved to: {OUTPUT_CSV}")
    return OUTPUT_CSV


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve diverse-data images against Qdrant and record scores."
    )
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed IDs")
    args = parser.parse_args()

    retrieve_diverse(limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
