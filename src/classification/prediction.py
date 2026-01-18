import csv
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import RESULTS_DIR, SEGMENTED_IMAGE_DIR, STRAIN_SPECIES_MAPPING_PATH
from src.database.query_utils import (
    find_nearest_neighbors_by_id,
    get_image_metadata,
    visualize_neighbors,
)
from src.feature_extraction.feature_extractors import FeatureExtractor


def load_strain_to_species_mapping(
    csv_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
) -> Dict[str, str]:
    """
    Load strain to species mapping from CSV file.
    """
    strain_to_specy = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strain_to_specy[row["Strain"]] = row["Species"]
    return strain_to_specy


def get_all_images_for_strain(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get all images for a specific strain from the collection.
    """
    conditions = [FieldCondition(key="strain", match=MatchValue(value=strain))]

    if environment and environment.lower() != "all":
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )

    search_filter = Filter(must=conditions)

    all_images = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )

        points, next_offset = result

        for point in points:
            payload = point.payload
            image_data = {
                "image_id": payload.get("image_id"),
                "strain": payload.get("strain"),
                "environment": payload.get("environment"),
                "angle": payload.get("angle"),
                "specy": payload.get("specy"),
                "parent_id": payload.get("parent_id"),
                "segment_index": payload.get("segment_index"),
                "bbox": payload.get("bbox"),
            }
            all_images.append(image_data)

        if next_offset is None:
            break
        offset = next_offset

    return all_images


def get_available_strains(client: QdrantClient, collection_name: str) -> List[str]:
    """
    Get all unique strain names available in the collection.
    """
    all_strains = set()
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name, limit=100, offset=offset, with_payload=True
        )

        points, next_offset = result

        for point in points:
            strain = point.payload.get("strain")
            if strain:
                all_strains.add(strain)

        if next_offset is None:
            break
        offset = next_offset

    return sorted(list(all_strains))


def filter_siblings(
    neighbors: List[Dict[str, Any]], query_parent_id: str
) -> List[Dict[str, Any]]:
    """
    Filter out neighbors that come from the same parent image (siblings).
    """
    filtered = []
    for neighbor in neighbors:
        if neighbor.get("parent_id") != query_parent_id:
            filtered.append(neighbor)
    return filtered


def aggregate_predictions(
    all_results: List[Dict[str, Any]],
    strain_to_specy: Dict[str, str],
    k: int,
    min_samples: Optional[int] = None,
    strategy: str = "avg",
) -> List[Tuple[str, float]]:
    """
    Aggregate predictions from multiple query images (segments).
    """
    species_scores = Counter()
    species_counts = Counter()

    for result in all_results:
        neighbors = result["neighbors"]

        for neighbor in neighbors:
            specy = neighbor.get("specy")
            score = neighbor.get("score", 0.0)

            if not specy or specy == "unknown":
                strain = neighbor.get("strain")
                if strain:
                    specy = strain_to_specy.get(strain, "unknown")

            if specy and specy != "unknown":
                species_scores[specy] += score
                species_counts[specy] += 1

    aggregated = []
    total_neighbors = sum(species_counts.values())

    for specy, total_score in species_scores.items():
        if strategy == "avg":
            # Weighted sum (sum of scores) divided by total neighbors count
            final_score = total_score / total_neighbors if total_neighbors > 0 else 0
        elif strategy == "uni":
            # Uniform weight (count) divided by total neighbors count
            count = species_counts[specy]
            final_score = count / total_neighbors if total_neighbors > 0 else 0
        else:
            final_score = total_score
        aggregated.append((specy, final_score))

    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated


def predict(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: Optional[int] = None,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strategy: str = "avg",
    strain_to_specy_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
    segmented_image_dir: str = str(SEGMENTED_IMAGE_DIR),
    output_dir: str = str(RESULTS_DIR),
) -> Dict[str, Any]:
    """
    Predict species for a given strain.
    """
    strain_to_specy = load_strain_to_species_mapping(strain_to_specy_path)
    ground_truth_specy = strain_to_specy.get(strain, "unknown")

    query_images = get_all_images_for_strain(
        client, collection_name, strain, environment
    )

    if not query_images:
        return {
            "strain": strain,
            "ground_truth": ground_truth_specy,
            "predicted_specy": "unknown",
            "correct": False,
            "predicted_confidence": 0.0,
            "error": "No images found",
        }

    raw_results = []

    for query_img in query_images:
        image_id = query_img["image_id"]
        parent_id = query_img["parent_id"]

        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_extractor.name,
            num_neighbors=k
            * 10,  # Fetch significantly more to ensure enough non-siblings remain
            environment=(
                environment if environment and environment.lower() != "all" else None
            ),
            exclude_self=True,
        )

        if without_siblings:
            neighbors = filter_siblings(neighbors, parent_id)

        neighbors = neighbors[:k]

        raw_results.append(
            {
                "query_image_id": image_id,
                "query_environment": query_img.get("environment"),
                "neighbors": neighbors,
            }
        )

    aggregated = aggregate_predictions(
        raw_results, strain_to_specy, k, min_samples, strategy
    )

    if not aggregated:
        predicted_specy = "unknown"
        confidence = 0.0
    else:
        predicted_specy = aggregated[0][0]
        confidence = aggregated[0][1]

    is_correct = predicted_specy == ground_truth_specy

    return {
        "strain": strain,
        "ground_truth": ground_truth_specy,
        "predicted_specy": predicted_specy,
        "correct": is_correct,
        "predicted_confidence": confidence,
        "aggregated_results": [{"specy": s, "score": sc} for s, sc in aggregated],
        "raw_results": raw_results,
        "feature_extractor": feature_extractor.name,
        "strategy": strategy,
        "environment": environment,
    }


def visualize_false_predictions(
    client: QdrantClient,
    collection_name: str,
    result: Dict[str, Any],
    segmented_image_dir: str,
    output_dir: str,
    feature_extractor: FeatureExtractor,
    max_visualizations: int = 3,
) -> None:
    """
    Visualize false predictions.
    """
    # Implementation moved to visualization module or kept here?
    # The user asked to move visualization scripts.
    # But this function uses `visualize_false_prediction` from `query_utils` (or wherever it is).
    # I'll keep it here but it should import from visualization module if I move `visualize_false_prediction`.
    # For now, `visualize_false_prediction` is in `src.database.query_utils`.
    # Wait, I should probably move `visualize_false_prediction` to `src.classification.visualization.utils`.
    # But I left it in `query_utils`.

    from src.database.query_utils import visualize_false_prediction as viz_false

    if result["correct"]:
        return

    os.makedirs(output_dir, exist_ok=True)

    raw_results = result["raw_results"]
    # Sort by confidence of wrong prediction? Or just take first few.

    count = 0
    for raw in raw_results:
        if count >= max_visualizations:
            break

        query_id = raw["query_image_id"]
        neighbors = raw["neighbors"]

        # Check if this specific query led to wrong prediction (majority vote)
        # Simple check: is the top neighbor wrong?
        if not neighbors:
            continue

        top_neighbor = neighbors[0]
        if top_neighbor.get("specy") != result["ground_truth"]:
            # Visualize this one
            img_path = os.path.join(segmented_image_dir, f"{query_id}.jpg")
            output_path = os.path.join(
                output_dir, f"false_pred_{result['strain']}_{query_id}.jpg"
            )

            viz_false(
                query_image_path=img_path,
                neighbors=neighbors,
                segmented_image_dir=segmented_image_dir,
                output_path=output_path,
                ground_truth_species=result["ground_truth"],
                predicted_species=result["predicted_specy"],
            )
            count += 1


def draw_confusion_matrix(
    predictions: List[Dict[str, Any]],
    output_path: str = str(RESULTS_DIR / "confusion_matrix.png"),
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true = [p["ground_truth"] for p in predictions]
    y_pred = [p["predicted_specy"] for p in predictions]

    # Calculate accuracy
    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = (correct_count / len(predictions) * 100) if predictions else 0

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        f"Confusion Matrix - Accuracy: {accuracy:.2f}% ({correct_count}/{len(predictions)})"
    )
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def batch_predict(
    client: QdrantClient,
    collection_name: str,
    strains: List[str],
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: Optional[int] = None,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strain_to_specy_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
    segmented_image_dir: str = str(SEGMENTED_IMAGE_DIR),
    output_dir: str = str(RESULTS_DIR),
) -> List[Dict[str, Any]]:

    results = []
    for i, strain in enumerate(strains):
        print(f"Predicting for strain {strain} ({i+1}/{len(strains)})...")
        res = predict(
            client,
            collection_name,
            strain,
            feature_extractor,
            k,
            min_samples,
            without_siblings,
            environment,
            "avg",
            strain_to_specy_path,
            segmented_image_dir,
            output_dir,
        )
        results.append(res)

    return results


if __name__ == "__main__":
    # Example usage
    pass
