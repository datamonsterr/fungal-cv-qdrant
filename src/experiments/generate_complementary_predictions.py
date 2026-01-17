"""
Generate detailed predictions with raw_results for visualization.
Re-runs predictions for specific cases to get full neighbor information.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from prediction import predict_segment_group
from query_utils import QueryClient

from config import QDRANT_COLLECTION_FEATURES, SEGMENTED_IMAGE_DIR


def load_results_from_folder(folder_path: str) -> Tuple[List[Dict[str, Any]], Dict]:
    """Load prediction results and metadata from a feature extractor folder."""
    results_files = list(Path(folder_path).glob("results_*.json"))
    if not results_files:
        raise FileNotFoundError(f"No results_*.json found in {folder_path}")

    # Use the most recent file
    results_file = sorted(results_files)[-1]
    print(f"Loading {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    metadata = {
        "feature_extractor": data["metadata"]["feature_extractor"],
        "k": data["metadata"]["k"],
        "environment": data["metadata"].get("environment", "all"),
        "strategy": data["metadata"]["strategy"],
    }

    return data["predictions"], metadata


def find_complementary_cases(
    colorhistogram_predictions: List[Dict[str, Any]],
    resnet_predictions: List[Dict[str, Any]],
    efficient_predictions: List[Dict[str, Any]],
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Find cases where ColorHistogramHS is wrong but other models are correct.
    Returns (strain, test_set_index, ground_truth) tuples.
    """

    # Create lookup dictionaries by (strain, test_set_index)
    def create_lookup(predictions: List[Dict]) -> Dict[Tuple[str, int], Dict]:
        lookup = {}
        for pred in predictions:
            key = (pred["strain"], pred["test_set_index"])
            lookup[key] = pred
        return lookup

    colorhist_lookup = create_lookup(colorhistogram_predictions)
    resnet_lookup = create_lookup(resnet_predictions)
    efficient_lookup = create_lookup(efficient_predictions)

    resnet_only_cases = []
    efficient_only_cases = []
    all_wrong_cases = []

    # Find cases where ColorHistogramHS is wrong
    for key, colorhist_pred in colorhist_lookup.items():
        if not colorhist_pred["correct"]:
            strain, test_set_idx = key
            ground_truth = colorhist_pred["ground_truth"]

            # Add to all_wrong_cases
            all_wrong_cases.append(
                (strain, test_set_idx, ground_truth, "ColorHistogramHS")
            )

            # Check if ResNet50 is correct
            resnet_pred = resnet_lookup.get(key)
            if resnet_pred and resnet_pred["correct"]:
                resnet_only_cases.append(
                    (strain, test_set_idx, ground_truth, "ResNet50")
                )
                print(f"✓ ResNet50 corrects: {strain} test_set_{test_set_idx}")

            # Check if EfficientNetV2B0 is correct
            efficient_pred = efficient_lookup.get(key)
            if efficient_pred and efficient_pred["correct"]:
                efficient_only_cases.append(
                    (strain, test_set_idx, ground_truth, "EfficientNetV2B0")
                )
                print(f"✓ EfficientNetV2B0 corrects: {strain} test_set_{test_set_idx}")

    return resnet_only_cases, efficient_only_cases, all_wrong_cases


def regenerate_prediction_with_details(
    strain: str,
    test_set_index: int,
    ground_truth: str,
    feature_extractor: str,
    k: int,
    environment: str,
    strategy: str,
    query_client: QueryClient,
    strain_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Regenerate a prediction with full raw_results for visualization.
    """
    print(
        f"  Regenerating prediction for {strain} test_set_{test_set_index} with {feature_extractor}..."
    )

    # Get the test images for this strain and test set
    strain_data = strain_df[strain_df["strain"] == strain]
    if len(strain_data) == 0:
        raise ValueError(f"Strain {strain} not found in dataset")

    # Get test images for this specific test set index
    test_images = strain_data[strain_data["test_set"] == test_set_index][
        "image_id"
    ].tolist()

    if len(test_images) == 0:
        raise ValueError(f"No test images found for {strain} test_set_{test_set_index}")

    # Run prediction with full details
    result = predict_segment_group(
        test_segment_ids=test_images,
        strain_name=strain,
        ground_truth_specy=ground_truth,
        query_client=query_client,
        k=k,
        environment=environment if environment != "all" else None,
        min_samples=None,
        without_siblings=True,
        feature_extractor=feature_extractor,
        aggregation_strategy=strategy,
    )

    return result


def main():
    """Main function to generate detailed predictions for complementary cases."""
    # Configuration
    results_base_dir = "./results/comprehensive_k7_NoSib_6"
    output_base_dir = "./results/complementary_visualizations_detailed"
    k = 7

    print("=" * 80)
    print("GENERATE DETAILED PREDICTIONS FOR COMPLEMENTARY CASES")
    print("=" * 80)
    print()

    # Load predictions from each feature extractor
    print("Loading predictions...")
    colorhistogram_preds, colorhist_meta = load_results_from_folder(
        os.path.join(results_base_dir, "ColorHistogramHS_E2_AVG")
    )
    resnet_preds, resnet_meta = load_results_from_folder(
        os.path.join(results_base_dir, "ResNet50_E2_AVG")
    )
    efficient_preds, efficient_meta = load_results_from_folder(
        os.path.join(results_base_dir, "EfficientNetV2B0_E2_AVG")
    )
    print()

    # Find complementary cases
    print("Finding complementary cases...")
    resnet_only, efficient_only, all_wrong = find_complementary_cases(
        colorhistogram_preds, resnet_preds, efficient_preds
    )
    print()

    print("SUMMARY:")
    print(f"  Cases where ColorHistogramHS wrong: {len(all_wrong)}")
    print(f"  Cases where ResNet50 correct: {len(resnet_only)}")
    print(f"  Cases where EfficientNetV2B0 correct: {len(efficient_only)}")
    print()

    if len(resnet_only) == 0 and len(efficient_only) == 0:
        print("⚠ No complementary cases found!")
        return

    # Load strain data for test set lookup
    print("Loading strain data...")
    strain_csv_path = "../Dataset/strain_to_specy.csv"
    strain_df = pd.read_csv(strain_csv_path)
    print(f"  Loaded {len(strain_df)} records")
    print()

    # Initialize query clients for each feature extractor
    print("Initializing Qdrant clients...")
    clients = {}
    for extractor in ["ColorHistogramHS", "ResNet50", "EfficientNetV2B0"]:
        collection_name = QDRANT_COLLECTION_FEATURES.get(extractor)
        if collection_name:
            clients[extractor] = QueryClient(collection_name=collection_name)
            print(f"  ✓ {extractor} -> {collection_name}")
    print()

    # Generate detailed predictions for each category
    os.makedirs(output_base_dir, exist_ok=True)

    # 1. ResNet50 corrections
    if resnet_only:
        print(f"[1/3] Generating {len(resnet_only)} ResNet50 correction predictions...")
        resnet_output_dir = os.path.join(output_base_dir, "resnet_only")
        os.makedirs(resnet_output_dir, exist_ok=True)

        resnet_detailed_predictions = []
        for strain, test_idx, ground_truth, model in resnet_only:
            try:
                result = regenerate_prediction_with_details(
                    strain=strain,
                    test_set_index=test_idx,
                    ground_truth=ground_truth,
                    feature_extractor="ResNet50",
                    k=k,
                    environment=resnet_meta["environment"],
                    strategy=resnet_meta["strategy"],
                    query_client=clients["ResNet50"],
                    strain_df=strain_df,
                )
                resnet_detailed_predictions.append(result)
            except Exception as e:
                print(f"    Error: {e}")

        # Save to JSON
        output_file = os.path.join(resnet_output_dir, "detailed_predictions.json")
        with open(output_file, "w") as f:
            json.dump(resnet_detailed_predictions, f, indent=2)
        print(
            f"  ✓ Saved {len(resnet_detailed_predictions)} predictions to {output_file}"
        )
        print()

    # 2. EfficientNetV2B0 corrections
    if efficient_only:
        print(
            f"[2/3] Generating {len(efficient_only)} EfficientNetV2B0 correction predictions..."
        )
        efficient_output_dir = os.path.join(output_base_dir, "efficient_only")
        os.makedirs(efficient_output_dir, exist_ok=True)

        efficient_detailed_predictions = []
        for strain, test_idx, ground_truth, model in efficient_only:
            try:
                result = regenerate_prediction_with_details(
                    strain=strain,
                    test_set_index=test_idx,
                    ground_truth=ground_truth,
                    feature_extractor="EfficientNetV2B0",
                    k=k,
                    environment=efficient_meta["environment"],
                    strategy=efficient_meta["strategy"],
                    query_client=clients["EfficientNetV2B0"],
                    strain_df=strain_df,
                )
                efficient_detailed_predictions.append(result)
            except Exception as e:
                print(f"    Error: {e}")

        # Save to JSON
        output_file = os.path.join(efficient_output_dir, "detailed_predictions.json")
        with open(output_file, "w") as f:
            json.dump(efficient_detailed_predictions, f, indent=2)
        print(
            f"  ✓ Saved {len(efficient_detailed_predictions)} predictions to {output_file}"
        )
        print()

    # 3. All ColorHistogramHS wrong cases
    print(f"[3/3] Generating {len(all_wrong)} ColorHistogramHS wrong predictions...")
    wrong_output_dir = os.path.join(output_base_dir, "wrong_colorhistogramhs")
    os.makedirs(wrong_output_dir, exist_ok=True)

    wrong_detailed_predictions = []
    for strain, test_idx, ground_truth, model in all_wrong:
        try:
            result = regenerate_prediction_with_details(
                strain=strain,
                test_set_index=test_idx,
                ground_truth=ground_truth,
                feature_extractor="ColorHistogramHS",
                k=k,
                environment=colorhist_meta["environment"],
                strategy=colorhist_meta["strategy"],
                query_client=clients["ColorHistogramHS"],
                strain_df=strain_df,
            )
            wrong_detailed_predictions.append(result)
        except Exception as e:
            print(f"    Error: {e}")

    # Save to JSON
    output_file = os.path.join(wrong_output_dir, "detailed_predictions.json")
    with open(output_file, "w") as f:
        json.dump(wrong_detailed_predictions, f, indent=2)
    print(f"  ✓ Saved {len(wrong_detailed_predictions)} predictions to {output_file}")
    print()

    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(
        f"\nNext step: Run visualization script to create images from these predictions"
    )
    print()


if __name__ == "__main__":
    main()
