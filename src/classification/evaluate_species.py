import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient

from src.classification.prediction import (
    draw_confusion_matrix,
    get_all_images_for_strain,
)
from src.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    RESULTS_DIR,
    STRAIN_SPECIES_MAPPING_PATH,
)
from src.feature_extraction.feature_extractors import (
    ColorHistogramExtractor,
    ColorHistogramHSExtractor,
    EfficientNetB1Extractor,
    FeatureExtractor,
    GaborExtractor,
    HOGExtractor,
    MobileNetV2Extractor,
    ResNet50Extractor,
)

# Original evaluate_species.py had predict_segment_group defined inside it?
# Let me check evaluate_species.py again.
# Yes, predict_segment_group was in evaluate_species.py.
# I should probably move it to prediction.py or keep it here.
# It seems to be a variant of predict.


def print_selection_report(selected: Dict[str, str], output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"strain_selection_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(f"Total species: {len(selected)}\n")
        for species, strain in selected.items():
            f.write(f"{species}: {strain}\n")
    return report_path


def print_prediction_results(results: List[Dict[str, Any]], output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"prediction_report_{timestamp}.txt")

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / len(results) if results else 0

    with open(report_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        for r in results:
            f.write(
                f"{r['strain']} ({r['ground_truth']}) -> {r['predicted_specy']} [{'Correct' if r['correct'] else 'Wrong'}]\n"
            )

    # Also save JSON summary
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


def collect_testset(
    client: QdrantClient, collection_name: str, strain: str, environment_strategy: str
) -> List[List[Dict[str, Any]]]:
    """
    Collect test sets for a strain based on environment strategy.
    Creates up to 6 test sets, where each test set contains one image per environment.
    """
    # Determine strategy type and extract environment name
    if environment_strategy.startswith("E3_"):
        is_e3 = True
        is_e4 = False
        target_env = environment_strategy[3:]  # Extract environment name after "E3_"
        exclude_env = None
        # Get images from specific environment only
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=target_env,
        )
    elif environment_strategy.startswith("E4_"):
        # E4: Get ALL images but will exclude specific environment from test sets
        is_e3 = False
        is_e4 = True
        target_env = None
        exclude_env = environment_strategy[3:]  # Extract environment name after "E4_"
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=None,
        )
    else:
        # E1 or E2: Get ALL images from ALL environments
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

    if not strain_images:
        return []

    if is_e3:
        # E3: Create test sets with one image each from the specific environment
        # Group by segment_index to get different "views"
        test_sets = []
        for img in strain_images[:6]:  # Take up to 6 different images
            test_sets.append([img])
        return test_sets
    else:
        # E1/E2/E4: Create test sets with one image per environment
        # Group images by environment, segment_index, and angle
        env_segment_angle_images = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for img in strain_images:
            env = img.get("environment", "unknown")
            # Skip excluded environment for E4 strategy
            if is_e4 and env == exclude_env:
                continue
            segment_idx = img.get("segment_index", 0)
            angle = img.get("angle", "unknown")
            env_segment_angle_images[env][segment_idx][angle].append(img)

        # Create test sets based on segment_index and angle combinations
        # With 3 segments (0,1,2) and 2 angles (ob/obverse, rev/reverse), we get 6 unique test sets
        test_sets = []

        # Define test set configurations: (segment_index, preferred_angle)
        # This ensures each test set uses a unique combination of segment and angle
        test_configs = [
            (0, "ob"),
            (0, "rev"),
            (1, "ob"),
            (1, "rev"),
            (2, "ob"),
            (2, "rev"),
        ]

        # Track which (image_id, angle) combinations have been used to ensure diversity
        used_image_angle_per_env = defaultdict(set)

        for test_idx, (segment_idx, preferred_angle) in enumerate(test_configs):
            test_set = []

            # For each environment, pick the image with this segment_index and angle
            for env in sorted(env_segment_angle_images.keys()):
                segment_images = env_segment_angle_images[env]

                # Try to get image with current segment_index and preferred angle
                img_selected = None
                if segment_idx in segment_images:
                    # Try preferred angle first (supporting both 'ob'/'rev' and 'obverse'/'reverse')
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
                            # Try to find an unused (image_id, angle) combination
                            for candidate in candidates:
                                combo_key = (
                                    candidate["image_id"],
                                    candidate.get("angle", "unknown"),
                                )
                                if combo_key not in used_image_angle_per_env[env]:
                                    img_selected = candidate
                                    break
                            # If all are used, take the first one anyway
                            if img_selected is None and candidates:
                                img_selected = candidates[0]
                            break

                    # If preferred angle not found, use any angle from this segment
                    if img_selected is None:
                        for angle in sorted(segment_images[segment_idx].keys()):
                            candidates = segment_images[segment_idx][angle]
                            if candidates:
                                # Try to find an unused (image_id, angle) combination
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

                # Fallback: use any available image from this environment, preferring unused (image_id, angle) combinations
                if img_selected is None:
                    for seg_idx in sorted(segment_images.keys()):
                        for angle in sorted(segment_images[seg_idx].keys()):
                            candidates = segment_images[seg_idx][angle]
                            if candidates:
                                # Try to find an unused (image_id, angle) combination
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

                # Only add to test set if we found a unique image
                if img_selected is not None:
                    test_set.append(img_selected)
                    combo_key = (
                        img_selected["image_id"],
                        img_selected.get("angle", "unknown"),
                    )
                    used_image_angle_per_env[env].add(combo_key)
                else:
                    # Can't find unique image for this environment, skip this test set
                    break

            # Only add test set if it has images from all environments
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
    strategy: str = "avg",
    strain_to_specy_path: str = str(STRAIN_SPECIES_MAPPING_PATH),
) -> Dict[str, Any]:
    # This is similar to predict but takes a pre-fetched group of images
    import pandas as pd

    from src.classification.prediction import (
        aggregate_predictions,
        filter_siblings,
        find_nearest_neighbors_by_id,
    )

    df = pd.read_csv(strain_to_specy_path)
    strain_to_specy = dict(zip(df["Strain"], df["Species"]))
    ground_truth_specy = strain_to_specy.get(strain, "unknown")

    raw_results = []
    for query_img in test_group:
        image_id = query_img["image_id"]
        parent_id = query_img["parent_id"]
        img_environment = query_img.get("environment", "unknown")

        # Determine environment filter logic
        search_environment = None
        exclude_environment = None

        if environment is None:
            # E1: Use same environment as query image
            search_environment = img_environment
        elif environment.lower() == "all":
            # E2: No environment filter
            search_environment = None
        elif environment.startswith("E4_"):
            # E4: Exclude specific environment
            exclude_environment = environment[3:]
            search_environment = None
        elif environment.startswith("E3_"):
            # E3: Use specific environment
            search_environment = environment[3:]
        else:
            # E3 (legacy) or other: Use specified environment
            search_environment = environment

        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_extractor.name.lower(),
            num_neighbors=k
            * 10,  # Fetch significantly more to ensure enough non-siblings remain
            environment=search_environment,
            exclude_self=True,
            exclude_environment=exclude_environment,
            exclude_strain=strain,  # Exclude the query strain from results
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


def run_species_evaluation(
    client: QdrantClient,
    collection_name: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: int = None,
    without_siblings: bool = True,
    environment: str = None,
    strategy: str = "avg",
    output_dir: str = str(RESULTS_DIR),
    generate_visualizations: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:

    import pandas as pd

    from src.classification.visualization.visualize_prediction import (
        batch_visualize_predictions,
    )

    if not STRAIN_SPECIES_MAPPING_PATH.exists():
        print(
            f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found. Please run 'python src/main.py generate-mapping' first."
        )
        return [], ""

    df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    if "Test" not in df_mapping.columns:
        print(
            "Error: 'Test' column not found in mapping CSV. Please regenerate mapping."
        )
        return [], ""

    # Select strains where Test is True
    test_df = df_mapping[df_mapping["Test"] == True]
    selected_strains = {}
    for _, row in test_df.iterrows():
        selected_strains[row["Species"]] = row["Strain"]

    selection_report_path = print_selection_report(selected_strains, output_dir)

    results = []
    # Iterate over the selected strains (one per species as per mapping logic)
    for species, strain in selected_strains.items():
        print(f"Evaluating {species} (Strain: {strain})...")

        # Use collect_testset to get test sets based on strategy
        # If environment is None, default to "E1" (all images)
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
                environment=environment,  # Pass the original environment argument
                strategy=strategy,
            )
            # Add metadata
            res["test_set_index"] = i
            res["environment_strategy"] = env_strategy
            results.append(res)

    report_path = print_prediction_results(results, output_dir)
    draw_confusion_matrix(results, os.path.join(output_dir, "confusion_matrix.png"))

    # Generate visualizations if requested
    if generate_visualizations and results:
        from src.config import SEGMENTED_IMAGE_DIR

        print("\nGenerating visualizations...")

        # Visualize correct predictions
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
                max_visualizations=None,  # Visualize all
            )

        # Visualize incorrect predictions
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
                max_visualizations=None,  # Visualize all
            )

        print(
            f"  Visualizations saved to: {os.path.join(output_dir, 'visualizations')}"
        )

    return results, report_path


if __name__ == "__main__":
    pass
