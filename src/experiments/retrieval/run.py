"""Unified evaluation module.

This file is the canonical path for prediction and species evaluation logic.
It consolidates functionality previously split across classification modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import (
    RESULTS_DIR,
    SEGMENTED_IMAGE_DIR,
    STRAIN_SPECIES_MAPPING_PATH,
)
from src.experiments.feature_extraction.feature_extractors import FeatureExtractor
from src.utils.qdrant_query import find_nearest_neighbors_by_id


def get_extractor_by_name(name: str) -> Optional[FeatureExtractor]:
    """Build a feature extractor instance from a normalized short name."""
    from src.experiments.feature_extraction.feature_extractors import (
        ColorHistogramExtractor,
        EfficientNetB1Extractor,
        GaborExtractor,
        HOGExtractor,
        MobileNetV2Extractor,
        ResNet50Extractor,
    )

    normalized = name.lower()
    if normalized == "resnet50":
        return ResNet50Extractor()
    if normalized == "mobilenetv2":
        return MobileNetV2Extractor()
    if normalized == "efficientnetv2":
        return EfficientNetB1Extractor()
    if normalized == "hog":
        return HOGExtractor()
    if normalized == "gabor":
        return GaborExtractor()
    if normalized == "colorhistogram":
        return ColorHistogramExtractor()
    return None


def load_strain_to_species_mapping(
    csv_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
) -> Dict[str, str]:
    """Load a ``strain -> species`` mapping from CSV."""
    strain_to_specy: Dict[str, str] = {}
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
    """Fetch all points for one strain (optionally filtered by environment)."""
    conditions: list[FieldCondition | Filter] = [
        FieldCondition(key="strain", match=MatchValue(value=strain))
    ]

    if environment and environment.lower() != "all":
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )

    search_filter = Filter(must=list(conditions))

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
            payload = point.payload or {}
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


def filter_siblings(
    neighbors: List[Dict[str, Any]], query_parent_id: str
) -> List[Dict[str, Any]]:
    """Remove neighbors from the same parent image as the query segment."""
    return [n for n in neighbors if n.get("parent_id") != query_parent_id]


def aggregate_predictions(
    all_results: List[Dict[str, Any]],
    strain_to_specy: Dict[str, str],
    k: int,
    min_samples: Optional[int] = None,
    strategy: str = "weighted",
) -> List[Tuple[str, float]]:
    """Aggregate per-segment neighbors into strain-level species ranking."""
    del k
    del min_samples

    species_scores: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()

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

    aggregated: List[Tuple[str, float]] = []
    total_neighbors = sum(species_counts.values())

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


def find_neighbors(
    client: QdrantClient,
    collection_name: str,
    query_image_id: str,
    feature_extractor: FeatureExtractor,
    k: int = 10,
    environment: Optional[str] = None,
    exclude_self: bool = True,
    exclude_environment: Optional[str] = None,
    exclude_strain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Compatibility helper to query neighbors for one image ID."""
    return find_nearest_neighbors_by_id(
        client=client,
        collection_name=collection_name,
        query_image_id=query_image_id,
        feature_type=feature_extractor.name,
        num_neighbors=k,
        environment=environment,
        exclude_self=exclude_self,
        exclude_environment=exclude_environment,
        exclude_strain=exclude_strain,
    )


def predict(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: Optional[int] = None,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strategy: str = "weighted",
    strain_to_specy_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
    segmented_image_dir: str = str(SEGMENTED_IMAGE_DIR),
    output_dir: str = str(RESULTS_DIR),
) -> Dict[str, Any]:
    """Predict species for one strain by aggregating over its segments."""
    del segmented_image_dir
    del output_dir

    strain_to_specy = load_strain_to_species_mapping(strain_to_specy_path)
    ground_truth_specy = strain_to_specy.get(strain, "unknown")

    query_images = get_all_images_for_strain(
        client=client,
        collection_name=collection_name,
        strain=strain,
        environment=environment,
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

    raw_results: List[Dict[str, Any]] = []

    for query_img in query_images:
        image_id = query_img["image_id"]
        parent_id = query_img["parent_id"]

        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_extractor.name,
            num_neighbors=k * 10,
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
        raw_results,
        strain_to_specy,
        k,
        min_samples,
        strategy,
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


def draw_confusion_matrix(
    predictions: List[Dict[str, Any]],
    output_path: str = str(RESULTS_DIR / "confusion_matrix.png"),
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Render and save confusion matrix for species predictions."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true = [p["ground_truth"] for p in predictions]
    y_pred = [p["predicted_specy"] for p in predictions]

    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = (correct_count / len(predictions) * 100.0) if predictions else 0.0

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
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


def print_selection_report(selected: Dict[str, str], output_dir: str) -> str:
    """Write selected strain-per-species report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"strain_selection_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"Total species: {len(selected)}\\n")
        for species, strain in selected.items():
            f.write(f"{species}: {strain}\\n")
    return report_path


def print_prediction_results(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Write human-readable and JSON prediction summaries."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"prediction_report_{timestamp}.txt")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0.0

    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\\n")
        for r in results:
            f.write(
                f"{r['strain']} ({r['ground_truth']}) -> {r['predicted_specy']} "
                f"[{'Correct' if r['correct'] else 'Wrong'}]\\n"
            )

    json_path = os.path.join(output_dir, "evaluation_results.json")
    summary = {
        "overall_accuracy": accuracy,
        "correct_predictions": correct_count,
        "total_strains": len(results),
        "timestamp": timestamp,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return report_path


def write_evaluation_csv(
    results: List[Dict[str, Any]],
    output_dir: str,
    feature_extractor: FeatureExtractor,
    environment: Optional[str],
    strategy: str,
) -> Optional[str]:
    """Write ranked species predictions to CSV."""
    if not results:
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if environment is None:
        media_label = "same"
    elif isinstance(environment, str) and environment.lower() == "all":
        media_label = "all"
    else:
        media_label = str(environment)

    agg_label = "weighted" if strategy == "weighted" else strategy

    max_rank = max(len(r.get("aggregated_results", [])) for r in results)

    base_fields = [
        "strain",
        "test_set_index",
        "ground_truth",
        "feature_extractor",
        "media",
        "aggregation",
    ]

    rank_fields: List[str] = []
    for rank in range(1, max_rank + 1):
        rank_fields.append(f"rank{rank}_species")
        rank_fields.append(f"rank{rank}_score")

    fieldnames = base_fields + rank_fields
    csv_name = f"{output_path.name}.csv"
    csv_path = output_path / csv_name

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row: Dict[str, Any] = {
                "strain": result.get("strain"),
                "test_set_index": result.get("test_set_index"),
                "ground_truth": result.get("ground_truth"),
                "feature_extractor": feature_extractor.name,
                "media": media_label,
                "aggregation": agg_label,
            }

            aggregated = result.get("aggregated_results", [])
            for idx, entry in enumerate(aggregated, start=1):
                row[f"rank{idx}_species"] = entry.get("specy")
                row[f"rank{idx}_score"] = entry.get("score")

            writer.writerow(row)

    return str(csv_path)


def collect_testset(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment_strategy: str,
) -> List[List[Dict[str, Any]]]:
    """Build per-strain test sets according to E1/E2/E3/E4 strategy."""
    if environment_strategy.startswith("E3_"):
        is_e3 = True
        is_e4 = False
        target_env = environment_strategy[3:]
        exclude_env = None
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=target_env,
        )
    elif environment_strategy.startswith("E4_"):
        is_e3 = False
        is_e4 = True
        target_env = None
        exclude_env = environment_strategy[3:]
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=None,
        )
    else:
        is_e3 = False
        is_e4 = False
        target_env = None
        exclude_env = None
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=None,
        )

    del target_env

    if not strain_images:
        return []

    if is_e3:
        test_sets: List[List[Dict[str, Any]]] = []
        for img in strain_images[:6]:
            test_sets.append([img])
        return test_sets

    env_segment_angle_images: Dict[str, Dict[Any, Dict[Any, List[Dict[str, Any]]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for img in strain_images:
        env = img.get("environment", "unknown")
        if is_e4 and env == exclude_env:
            continue
        segment_idx = img.get("segment_index", 0)
        angle = img.get("angle", "unknown")
        env_segment_angle_images[env][segment_idx][angle].append(img)

    test_sets = []
    test_configs = [
        (0, "ob"),
        (0, "rev"),
        (1, "ob"),
        (1, "rev"),
        (2, "ob"),
        (2, "rev"),
    ]

    used_image_angle_per_env: Dict[str, set] = defaultdict(set)

    for segment_idx, preferred_angle in test_configs:
        test_set: List[Dict[str, Any]] = []

        for env in sorted(env_segment_angle_images.keys()):
            segment_images = env_segment_angle_images[env]
            img_selected: Optional[Dict[str, Any]] = None

            if segment_idx in segment_images:
                angle_variations = {
                    "ob": ["ob", "obverse"],
                    "rev": ["rev", "reverse"],
                }
                for angle_var in angle_variations.get(
                    preferred_angle, [preferred_angle]
                ):
                    if (
                        angle_var in segment_images[segment_idx]
                        and segment_images[segment_idx][angle_var]
                    ):
                        candidates = segment_images[segment_idx][angle_var]
                        for candidate in candidates:
                            combo_key = (
                                candidate["image_id"],
                                candidate.get("angle", "unknown"),
                            )
                            if combo_key not in used_image_angle_per_env[env]:
                                img_selected = candidate
                                break
                        if img_selected is None and candidates:
                            img_selected = candidates[0]
                        break

                if img_selected is None:
                    for angle in sorted(segment_images[segment_idx].keys()):
                        candidates = segment_images[segment_idx][angle]
                        if candidates:
                            for candidate in candidates:
                                combo_key = (
                                    candidate["image_id"],
                                    candidate.get("angle", "unknown"),
                                )
                                if combo_key not in used_image_angle_per_env[env]:
                                    img_selected = candidate
                                    break
                            if img_selected is None:
                                img_selected = candidates[0]
                            break

            if img_selected is None:
                for seg_idx in sorted(segment_images.keys()):
                    for angle in sorted(segment_images[seg_idx].keys()):
                        candidates = segment_images[seg_idx][angle]
                        if candidates:
                            for candidate in candidates:
                                combo_key = (
                                    candidate["image_id"],
                                    candidate.get("angle", "unknown"),
                                )
                                if combo_key not in used_image_angle_per_env[env]:
                                    img_selected = candidate
                                    break
                            if img_selected is not None:
                                break
                    if img_selected is not None:
                        break

            if img_selected is not None:
                test_set.append(img_selected)
                combo_key = (
                    img_selected["image_id"],
                    img_selected.get("angle", "unknown"),
                )
                used_image_angle_per_env[env].add(combo_key)
            else:
                break

        if test_set and len(test_set) == len(env_segment_angle_images):
            test_sets.append(test_set)

    return test_sets


def predict_segment_group(
    client: QdrantClient,
    collection_name: str,
    test_group: List[Dict[str, Any]],
    strain: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: Optional[int] = None,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strategy: str = "weighted",
    strain_to_specy_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
) -> Dict[str, Any]:
    """Predict species from a provided segment group (one strain test split)."""
    import pandas as pd

    df = pd.read_csv(strain_to_specy_path)
    strain_to_specy = dict(zip(df["Strain"], df["Species"]))
    ground_truth_specy = strain_to_specy.get(strain, "unknown")

    raw_results: List[Dict[str, Any]] = []
    for query_img in test_group:
        image_id = query_img["image_id"]
        parent_id = query_img["parent_id"]
        img_environment = query_img.get("environment", "unknown")

        search_environment = None
        exclude_environment = None

        if environment is None:
            search_environment = img_environment
        elif environment.lower() == "all":
            search_environment = None
        elif environment.startswith("E4_"):
            exclude_environment = environment[3:]
            search_environment = None
        elif environment.startswith("E3_"):
            search_environment = environment[3:]
        else:
            search_environment = environment

        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_extractor.name,
            num_neighbors=k * 10,
            environment=search_environment,
            exclude_self=True,
            exclude_environment=exclude_environment,
            exclude_strain=strain,
        )

        if without_siblings:
            neighbors = filter_siblings(neighbors, parent_id)

        neighbors = neighbors[:k]

        raw_results.append(
            {
                "query_image_id": image_id,
                "query_environment": img_environment,
                "neighbors": neighbors,
            }
        )

    aggregated = aggregate_predictions(
        raw_results,
        strain_to_specy,
        k,
        min_samples,
        strategy,
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


def run_species_evaluation(
    client: QdrantClient,
    collection_name: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: Optional[int] = None,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strategy: str = "weighted",
    output_dir: str = str(RESULTS_DIR),
    generate_visualizations: bool = False,
    selected_strains: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Run evaluation over selected test strains and return predictions + report path."""
    import pandas as pd

    from src.analysis.visualization.visualize_prediction import (
        batch_visualize_predictions,
    )

    if selected_strains is None:
        if not STRAIN_SPECIES_MAPPING_PATH.exists():
            print(
                f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found. "
                "Please run 'uv run python -m src.utils.generate_strain_mapping' first."
            )
            return [], ""

        df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
        if "Test" not in df_mapping.columns:
            print(
                "Error: 'Test' column not found in mapping CSV. Please regenerate mapping."
            )
            return [], ""

        test_df = df_mapping[df_mapping["Test"]]
        selected_strains = {}
        for _, row in test_df.iterrows():
            selected_strains[row["Species"]] = row["Strain"]

    selection_report_path = print_selection_report(selected_strains, output_dir)
    del selection_report_path

    results: List[Dict[str, Any]] = []
    for species, strain in selected_strains.items():
        print(f"Evaluating {species} (Strain: {strain})...")
        env_strategy = environment if environment else "E1"

        test_sets = collect_testset(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment_strategy=env_strategy,
        )

        if not test_sets:
            print(f"  No test sets found for {strain} with strategy {env_strategy}")
            continue

        print(f"  Found {len(test_sets)} test sets for {strain}")
        for i, test_group in enumerate(test_sets):
            res = predict_segment_group(
                client=client,
                collection_name=collection_name,
                test_group=test_group,
                strain=strain,
                feature_extractor=feature_extractor,
                k=k,
                min_samples=min_samples,
                without_siblings=without_siblings,
                environment=environment,
                strategy=strategy,
            )
            res["test_set_index"] = i
            res["environment_strategy"] = env_strategy
            results.append(res)

    report_path = print_prediction_results(results, output_dir)
    draw_confusion_matrix(results, os.path.join(output_dir, "confusion_matrix.png"))

    csv_path = write_evaluation_csv(
        results=results,
        output_dir=output_dir,
        feature_extractor=feature_extractor,
        environment=environment,
        strategy=strategy,
    )
    if csv_path:
        print(f"CSV saved to: {csv_path}")

    if generate_visualizations and results:
        print("\nGenerating visualizations...")

        correct_dir = os.path.join(output_dir, "visualizations", "correct")
        os.makedirs(correct_dir, exist_ok=True)
        correct_results = [r for r in results if r["correct"]]
        if correct_results:
            print(f"  Visualizing {len(correct_results)} correct predictions...")
            batch_visualize_predictions(
                prediction_results=correct_results,
                segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                output_dir=correct_dir,
                k=k,
                filter_correct=True,
                max_visualizations=None,
            )

        incorrect_dir = os.path.join(output_dir, "visualizations", "incorrect")
        os.makedirs(incorrect_dir, exist_ok=True)
        incorrect_results = [r for r in results if not r["correct"]]
        if incorrect_results:
            print(f"  Visualizing {len(incorrect_results)} incorrect predictions...")
            batch_visualize_predictions(
                prediction_results=incorrect_results,
                segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                output_dir=incorrect_dir,
                k=k,
                filter_correct=False,
                max_visualizations=None,
            )

        print(
            f"  Visualizations saved to: {os.path.join(output_dir, 'visualizations')}"
        )

    return results, report_path


def run_comprehensive_report(
    identifier: str,
    extractors: List[str],
    env_strategies: List[str],
    agg_strategies: List[str],
    k: int = 5,
    max_visualizations: int = 20,
    visualize_correct: bool = True,
    visualize_incorrect: bool = True,
) -> None:
    """Run a multi-configuration retrieval benchmark and optional visualization."""
    from src.analysis.visualization.visualize_prediction import (
        batch_visualize_predictions,
    )
    from src.config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL

    print(f"Starting Comprehensive Report: {identifier}")
    print(f"Extractors: {extractors}")
    print(f"Environment Strategies: {env_strategies}")
    print(f"Aggregation Strategies: {agg_strategies}")
    print(f"K: {k}")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    base_output_dir = RESULTS_DIR / identifier
    base_output_dir.mkdir(parents=True, exist_ok=True)

    for ext_name in extractors:
        extractor = get_extractor_by_name(ext_name)
        if not extractor:
            print(f"Warning: Unknown extractor {ext_name}, skipping.")
            continue

        for env_strat in env_strategies:
            for agg_strat in agg_strategies:
                subfolder_name = f"{ext_name}_{env_strat}_{agg_strat}"
                output_dir = base_output_dir / subfolder_name
                output_dir.mkdir(parents=True, exist_ok=True)

                if env_strat == "E1":
                    environment_param = None
                elif env_strat == "E2":
                    environment_param = "all"
                else:
                    environment_param = env_strat

                print(f"\\nRunning evaluation for: {subfolder_name}")
                results, _ = run_species_evaluation(
                    client=client,
                    collection_name=COLLECTION_NAME,
                    feature_extractor=extractor,
                    k=k,
                    without_siblings=True,
                    environment=environment_param,
                    strategy=agg_strat,
                    output_dir=str(output_dir),
                )

                if visualize_correct:
                    batch_visualize_predictions(
                        prediction_results=results,
                        segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                        output_dir=str(output_dir / "visualizations" / "correct"),
                        k=k,
                        filter_correct=True,
                        max_visualizations=max_visualizations,
                    )

                if visualize_incorrect:
                    batch_visualize_predictions(
                        prediction_results=results,
                        segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                        output_dir=str(output_dir / "visualizations" / "incorrect"),
                        k=k,
                        filter_correct=False,
                        max_visualizations=max_visualizations,
                    )


def run_ensemble_analysis() -> None:
    """Execute ensemble analysis workflow from retrieval experiment."""
    from src.experiments.retrieval import ensemble_analysis

    ensemble_analysis.main()


def run_ensemble_report(strategy: str = "weighted") -> None:
    """Generate detailed ensemble report for one strategy."""
    from src.analysis.retrieval.ensemble_report import (
        create_detailed_comparison_chart,
        load_ensemble_results,
        print_detailed_report,
    )

    results_dir = RESULTS_DIR / "ensemble_analysis"
    json_path = str(results_dir / f"ensemble_results_{strategy}.json")
    output_path = str(results_dir / f"detailed_comparison_{strategy}.png")

    print(f"Loading ensemble results for strategy: {strategy}")
    results = load_ensemble_results(json_path)
    print_detailed_report(results)
    create_detailed_comparison_chart(results, output_path)


def run_compare_ensemble_strategies() -> None:
    """Compare weighted and simple-average ensemble predictions."""
    from src.analysis.retrieval.compare_ensemble_strategies import compare_strategies

    compare_strategies()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieval experiment runner")
    subparsers = parser.add_subparsers(dest="command")

    comprehensive = subparsers.add_parser(
        "comprehensive", help="Run multi-extractor comprehensive retrieval report"
    )
    comprehensive.add_argument(
        "--identifier",
        type=str,
        default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Unique identifier for this run",
    )
    comprehensive.add_argument(
        "--extractors",
        nargs="+",
        default=[
            "resnet50",
            "mobilenetv2",
            "efficientnetv2",
            "hog",
            "gabor",
            "colorhistogram",
        ],
    )
    comprehensive.add_argument("--env_strategies", nargs="+", default=["E1", "E2"])
    comprehensive.add_argument(
        "--agg_strategies", nargs="+", default=["weighted", "uni"]
    )
    comprehensive.add_argument("--k", type=int, default=5)
    comprehensive.add_argument("--max_visualizations", type=int, default=20)
    comprehensive.add_argument(
        "--no-viz-correct",
        action="store_false",
        dest="visualize_correct",
    )
    comprehensive.add_argument(
        "--no-viz-incorrect",
        action="store_false",
        dest="visualize_incorrect",
    )
    comprehensive.set_defaults(visualize_correct=True, visualize_incorrect=True)

    subparsers.add_parser("ensemble-analysis")

    ensemble_report = subparsers.add_parser("ensemble-report")
    ensemble_report.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        choices=["weighted", "simple_avg", "manual_weighted"],
    )

    subparsers.add_parser("ensemble-compare")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "comprehensive":
        run_comprehensive_report(
            identifier=args.identifier,
            extractors=args.extractors,
            env_strategies=args.env_strategies,
            agg_strategies=args.agg_strategies,
            k=args.k,
            max_visualizations=args.max_visualizations,
            visualize_correct=args.visualize_correct,
            visualize_incorrect=args.visualize_incorrect,
        )
        return

    if args.command == "ensemble-analysis":
        run_ensemble_analysis()
        return

    if args.command == "ensemble-report":
        run_ensemble_report(strategy=args.strategy)
        return

    if args.command == "ensemble-compare":
        run_compare_ensemble_strategies()
        return

    parser.print_help()


__all__ = [
    "aggregate_predictions",
    "collect_testset",
    "draw_confusion_matrix",
    "filter_siblings",
    "find_neighbors",
    "get_all_images_for_strain",
    "get_extractor_by_name",
    "load_strain_to_species_mapping",
    "predict",
    "predict_segment_group",
    "run_compare_ensemble_strategies",
    "run_comprehensive_report",
    "run_ensemble_analysis",
    "run_ensemble_report",
    "run_species_evaluation",
    "write_evaluation_csv",
    "ExperimentParams",
    "ExperimentResult",
    "run",
]


if __name__ == "__main__":
    main()


@dataclass
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@dataclass
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: list
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    """Execute one retrieval experiment run scoped to params.output_root."""
    import json
    from pathlib import Path as _Path

    output_root = _Path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "log"
    log_dir.mkdir(exist_ok=True)

    try:
        from src.config import COLLECTION_NAME, QDRANT_API_KEY, QDRANT_URL
        from qdrant_client import QdrantClient as _QC

        client = _QC(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        extractor = get_extractor_by_name("resnet50")
        if extractor is None:
            raise ValueError("resnet50 extractor unavailable")
        results, report_path = run_species_evaluation(
            client=client,
            collection_name=COLLECTION_NAME,
            feature_extractor=extractor,
            output_dir=str(output_root / "artifacts"),
        )
        correct = sum(1 for r in results if r.get("correct", False))
        total = len(results)
        f1 = correct / total if total > 0 else 0.0
        strategy_name = params.description[:30] if params.description else "retrieval"
        artifact_paths = [str(output_root / "artifacts")]
    except Exception as exc:
        error_log = log_dir / "run.log"
        error_log.write_text(str(exc))
        raise

    result_data = {
        "f1_score": f1,
        "strategy_name": strategy_name,
        "artifact_paths": artifact_paths,
        "run_id": params.run_id,
    }
    results_json = output_root / "results.json"
    results_json.write_text(json.dumps(result_data, indent=2))
    artifact_paths = [str(p) for p in artifact_paths] + [str(results_json)]

    return ExperimentResult(
        f1_score=f1,
        strategy_name=strategy_name,
        artifact_paths=artifact_paths,
        run_id=params.run_id,
    )
