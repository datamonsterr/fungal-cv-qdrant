"""
Visualize complementary cases where ColorHistogramHS fails but other models succeed.
Creates separate visualizations for:
- resnet_only: Cases where only ResNet50 is correct
- efficient_only: Cases where only EfficientNetV2B0 is correct
- wrong_colorhistogramhs: All cases where ColorHistogramHS is wrong (for comparison)
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.analysis.visualization.visualize_prediction import (
    visualize_prediction_by_environment,
)
from src.config import RESULTS_DIR, SEGMENTED_IMAGE_DIR
from src.experiments.retrieval.run import predict


def load_results_from_folder(folder_path: str) -> Tuple[List[Dict[str, Any]], Dict]:
    """Load prediction results from a feature extractor folder."""
    results_files = list(Path(folder_path).glob("results_*.json"))
    if not results_files:
        raise FileNotFoundError(f"No results_*.json found in {folder_path}")

    # Use the most recent file
    results_file = sorted(results_files)[-1]
    print(f"Loading {results_file}")

    with open(results_file, "r") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    return data["predictions"], metadata


def find_complementary_cases(
    colorhistogram_predictions: List[Dict[str, Any]],
    resnet_predictions: List[Dict[str, Any]],
    efficient_predictions: List[Dict[str, Any]],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Find cases where ColorHistogramHS is wrong but other models are correct.

    Returns:
        Tuple of (resnet_only_cases, efficient_only_cases, all_colorhistogram_wrong_cases)
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
            # ColorHistogramHS is wrong - add to all_wrong_cases
            all_wrong_cases.append(colorhist_pred)

            # Check if ResNet50 is correct
            resnet_pred = resnet_lookup.get(key)
            if resnet_pred and resnet_pred["correct"]:
                resnet_only_cases.append(resnet_pred)
                print(
                    f"✓ ResNet50 corrects ColorHistogramHS: {key[0]} test_set_{key[1]}"
                )
                print(f"  Ground truth: {colorhist_pred['ground_truth']}")
                print(
                    f"  ColorHistogramHS predicted: {colorhist_pred['predicted_specy']} (wrong)"
                )
                print(
                    f"  ResNet50 predicted: {resnet_pred['predicted_specy']} (correct)"
                )

            # Check if EfficientNetV2B0 is correct
            efficient_pred = efficient_lookup.get(key)
            if efficient_pred and efficient_pred["correct"]:
                efficient_only_cases.append(efficient_pred)
                print(
                    f"✓ EfficientNetV2B0 corrects ColorHistogramHS: {key[0]} test_set_{key[1]}"
                )
                print(f"  Ground truth: {colorhist_pred['ground_truth']}")
                print(
                    f"  ColorHistogramHS predicted: {colorhist_pred['predicted_specy']} (wrong)"
                )
                print(
                    f"  EfficientNetV2B0 predicted: {efficient_pred['predicted_specy']} (correct)"
                )

    return resnet_only_cases, efficient_only_cases, all_wrong_cases


def regenerate_prediction_with_details(
    strain: str, test_set_index: int, feature_extractor: str, metadata: Dict
) -> Dict[str, Any]:
    """
    Regenerate a prediction with full raw_results for visualization.
    Uses the predict() function to get detailed neighbor information.
    """
    from qdrant_client import QdrantClient

    from src.config import QDRANT_COLLECTION_FEATURES, QDRANT_HOST, QDRANT_PORT

    collection_name = QDRANT_COLLECTION_FEATURES.get(feature_extractor)
    if not collection_name:
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    # Create Qdrant client
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Run prediction with full details
    k = metadata.get("k", 7)
    environment = metadata.get("environment", "all")
    strategy = metadata.get("strategy", "weighted")

    result = predict(
        query_strain=strain,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        k=k,
        environment=environment if environment != "all" else None,
        min_samples=None,
        without_siblings=True,
        feature_extractor=feature_extractor,
        aggregation_strategy=strategy,
        test_set_index=test_set_index,
    )

    return result


def main():
    """Main function to create complementary case visualizations."""
    # Configuration
    results_base_dir = RESULTS_DIR / "comprehensive_k7_NoSib_6"
    output_base_dir = RESULTS_DIR / "complementary_visualizations"
    k = 7  # Number of neighbors to show

    print("=" * 80)
    print("COMPLEMENTARY CASE VISUALIZATION")
    print("=" * 80)
    print(f"\nResults directory: {results_base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Segmented images: {SEGMENTED_IMAGE_DIR}")
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

    print(f"  ColorHistogramHS: {len(colorhistogram_preds)} predictions")
    print(f"  ResNet50: {len(resnet_preds)} predictions")
    print(f"  EfficientNetV2B0: {len(efficient_preds)} predictions")
    print()

    # Find complementary cases
    print("Finding complementary cases...")
    print("-" * 80)
    resnet_only, efficient_only, all_wrong = find_complementary_cases(
        colorhistogram_preds, resnet_preds, efficient_preds
    )
    print("-" * 80)
    print()

    # Print summary
    print("SUMMARY:")
    print(f"  Cases where ColorHistogramHS wrong: {len(all_wrong)}")
    print(
        f"  Cases where ResNet50 correct (ColorHistogramHS wrong): {len(resnet_only)}"
    )
    print(
        f"  Cases where EfficientNetV2B0 correct (ColorHistogramHS wrong): {len(efficient_only)}"
    )
    print()

    if len(resnet_only) == 0 and len(efficient_only) == 0:
        print("⚠ No complementary cases found!")
        print(
            "This explains why ensemble doesn't improve - models make the same mistakes."
        )
        return

    # Create visualizations for each category
    print("Creating visualizations...")
    print()

    os.makedirs(output_base_dir, exist_ok=True)

    # 1. ResNet50 only corrections
    if resnet_only:
        print(f"[1/3] ResNet50 corrections ({len(resnet_only)} cases)...")
        resnet_output_dir = os.path.join(output_base_dir, "resnet_only")
        os.makedirs(resnet_output_dir, exist_ok=True)

        for idx, case in enumerate(resnet_only, 1):
            strain = case["strain"]
            test_idx = case["test_set_index"]
            print(
                f"  [{idx}/{len(resnet_only)}] Regenerating prediction for {strain} test_set_{test_idx}..."
            )
            try:
                result = regenerate_prediction_with_details(
                    strain, test_idx, "ResNet50", resnet_meta
                )

                output_path = os.path.join(
                    resnet_output_dir,
                    f"{idx:03d}_{strain.replace(' ', '_')}_correct.jpg",
                )
                visualize_prediction_by_environment(
                    prediction_result=result,
                    segmented_image_dir=SEGMENTED_IMAGE_DIR,
                    output_path=output_path,
                    k=k,
                )
                print(f"    ✓ Saved to {output_path}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        print()
    else:
        print("[1/3] No ResNet50 corrections found - skipping")
        print()

    # 2. EfficientNetV2B0 only corrections
    if efficient_only:
        print(f"[2/3] EfficientNetV2B0 corrections ({len(efficient_only)} cases)...")
        efficient_output_dir = os.path.join(output_base_dir, "efficient_only")
        os.makedirs(efficient_output_dir, exist_ok=True)

        for idx, case in enumerate(efficient_only, 1):
            strain = case["strain"]
            test_idx = case["test_set_index"]
            print(
                f"  [{idx}/{len(efficient_only)}] Regenerating prediction for {strain} test_set_{test_idx}..."
            )
            try:
                result = regenerate_prediction_with_details(
                    strain, test_idx, "EfficientNetV2B0", efficient_meta
                )

                output_path = os.path.join(
                    efficient_output_dir,
                    f"{idx:03d}_{strain.replace(' ', '_')}_correct.jpg",
                )
                visualize_prediction_by_environment(
                    prediction_result=result,
                    segmented_image_dir=SEGMENTED_IMAGE_DIR,
                    output_path=output_path,
                    k=k,
                )
                print(f"    ✓ Saved to {output_path}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        print()
    else:
        print("[2/3] No EfficientNetV2B0 corrections found - skipping")
        print()

    # 3. All ColorHistogramHS wrong cases (for comparison)
    print(f"[3/3] All ColorHistogramHS wrong cases ({len(all_wrong)} cases)...")
    wrong_output_dir = os.path.join(output_base_dir, "wrong_colorhistogramhs")
    os.makedirs(wrong_output_dir, exist_ok=True)

    for idx, case in enumerate(all_wrong, 1):
        strain = case["strain"]
        test_idx = case["test_set_index"]
        print(
            f"  [{idx}/{len(all_wrong)}] Regenerating prediction for {strain} test_set_{test_idx}..."
        )
        try:
            result = regenerate_prediction_with_details(
                strain, test_idx, "ColorHistogramHS", colorhist_meta
            )

            output_path = os.path.join(
                wrong_output_dir, f"{idx:03d}_{strain.replace(' ', '_')}_false.jpg"
            )
            visualize_prediction_by_environment(
                prediction_result=result,
                segmented_image_dir=SEGMENTED_IMAGE_DIR,
                output_path=output_path,
                k=k,
            )
            print(f"    ✓ Saved to {output_path}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    print()

    # Final summary
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print("\nOutput directories:")
    if resnet_only:
        print(f"  - {os.path.join(output_base_dir, 'resnet_only')}/")
        print(
            f"    ({len(resnet_only)} cases where ResNet50 correct, ColorHistogramHS wrong)"
        )
    if efficient_only:
        print(f"  - {os.path.join(output_base_dir, 'efficient_only')}/")
        print(
            f"    ({len(efficient_only)} cases where EfficientNetV2B0 correct, ColorHistogramHS wrong)"
        )
    print(f"  - {os.path.join(output_base_dir, 'wrong_colorhistogramhs')}/")
    print(f"    ({len(all_wrong)} cases where ColorHistogramHS wrong)")
    print()

    # Analysis insight
    if all_wrong:
        correction_rate = (
            (len(resnet_only) + len(efficient_only)) / len(all_wrong) * 100
        )
        print(
            f"Correction rate: {correction_rate:.1f}% of ColorHistogramHS errors were corrected by other models"
        )
        print(
            "This low rate explains why ensemble (66.67%) performs worse than ColorHistogramHS alone (75%)"
        )
    print()


if __name__ == "__main__":
    main()
