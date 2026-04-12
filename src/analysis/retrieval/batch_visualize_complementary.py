"""
Batch script to generate visualizations for complementary cases.
Uses run_species_evaluation to get results with raw_results, then visualizes them.
"""

import json
import os

from qdrant_client import QdrantClient

from src.analysis.visualization.visualize_prediction import (
    visualize_prediction_by_environment,
)
from src.config import RESULTS_DIR, SEGMENTED_IMAGE_DIR
from src.experiments.feature_extraction.feature_extractors import (
    ColorHistogramHSExtractor,
    EfficientNetB1Extractor,
    ResNet50Extractor,
)
from src.experiments.retrieval.run import (
    collect_testset,
    predict_segment_group,
)


def load_complementary_cases():
    """Load the list of complementary cases from the analysis."""
    cases_file = (
        RESULTS_DIR
        / "ensemble_analysis"
        / "complementary_cases"
        / "complementary_cases_list.json"
    )

    with open(cases_file, "r") as f:
        data = json.load(f)

    return (data["resnet_only"], data["efficient_only"], data["all_colorhist_wrong"])


def visualize_strain_with_extractor(
    client,
    collection_name,
    strain,
    feature_extractor,
    extractor_name,
    test_set_index,
    output_dir,
    k=7,
):
    """
    Run evaluation for a single strain and visualize the specific test_set_index.
    """
    print(f"  Collecting test sets for {strain}...")

    # Collect test sets (6 test sets with one image per environment each)
    test_sets = collect_testset(
        client=client,
        collection_name=collection_name,
        strain=strain,
        environment_strategy="E2",  # E2 strategy - create test sets with images from all environments
    )

    if not test_sets:
        print(f"    ✗ No test sets could be created for {strain}")
        return None

    # Check if the requested test_set_index exists
    result_idx = test_set_index - 1  # Convert to 0-based index

    if result_idx >= len(test_sets) or result_idx < 0:
        print(
            f"    ✗ Test set {test_set_index} not found (only {len(test_sets)} test sets available)"
        )
        return None

    test_group = test_sets[result_idx]

    print(
        f"  Running prediction for test set {test_set_index} with {extractor_name}..."
    )

    # Run prediction on this specific test group
    result = predict_segment_group(
        client=client,
        collection_name=collection_name,
        test_group=test_group,
        strain=strain,
        feature_extractor=feature_extractor,
        k=k,
        min_samples=None,
        without_siblings=True,
        environment="all",  # E2 strategy
        strategy="weighted",
    )

    # Create visualization
    safe_strain = strain.replace(" ", "_").replace("/", "-")
    filename = f"{safe_strain}_test{test_set_index}_{extractor_name}.jpg"
    output_path = os.path.join(output_dir, filename)

    print("  Creating visualization...")
    visualize_prediction_by_environment(
        prediction_result=result,
        segmented_image_dir=SEGMENTED_IMAGE_DIR,
        output_path=output_path,
        k=k,
    )

    print(f"  ✓ Saved: {output_path}")
    return output_path


def main():
    """Main function to generate all complementary case visualizations."""

    print("=" * 80)
    print("BATCH VISUALIZE COMPLEMENTARY CASES")
    print("=" * 80)
    print()

    # Configuration
    CLIENT = QdrantClient(host="localhost", port=6333)
    COLLECTION_NAME = "myco_fungi_features_full"
    OUTPUT_BASE_DIR = RESULTS_DIR / "ensemble_analysis" / "complementary_cases"
    K = 7

    # Load complementary cases
    print("Loading complementary cases...")
    resnet_only, efficient_only, all_colorhist_wrong = load_complementary_cases()

    print(f"  ResNet50 corrections: {len(resnet_only)}")
    print(f"  EfficientNetV2B0 corrections: {len(efficient_only)}")
    print(f"  ColorHistogramHS wrong: {len(all_colorhist_wrong)}")
    print()

    # Create output directories
    resnet_dir = os.path.join(str(OUTPUT_BASE_DIR), "resnet_only")
    efficient_dir = os.path.join(str(OUTPUT_BASE_DIR), "efficient_only")
    wrong_dir = os.path.join(str(OUTPUT_BASE_DIR), "wrong_colorhistogramhs")

    os.makedirs(resnet_dir, exist_ok=True)
    os.makedirs(efficient_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)

    success_count = 0
    total_count = 0

    # 1. Generate ResNet50 corrections
    if resnet_only:
        print(
            f"[1/3] Generating {len(resnet_only)} ResNet50 correction visualizations..."
        )
        print("-" * 80)
        for idx, case in enumerate(resnet_only, 1):
            print(
                f"Case {idx}/{len(resnet_only)}: {case['strain']} test_set_{case['test_set_index']}"
            )
            total_count += 1
            try:
                result = visualize_strain_with_extractor(
                    client=CLIENT,
                    collection_name=COLLECTION_NAME,
                    strain=case["strain"],
                    feature_extractor=ResNet50Extractor(),
                    extractor_name="ResNet50",
                    test_set_index=case["test_set_index"],
                    output_dir=resnet_dir,
                    k=K,
                )
                if result:
                    success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
        print()
    else:
        print("[1/3] No ResNet50 corrections to visualize")
        print()

    # 2. Generate EfficientNetV2B0 corrections
    if efficient_only:
        print(
            f"[2/3] Generating {len(efficient_only)} EfficientNetV2B0 correction visualizations..."
        )
        print("-" * 80)
        for idx, case in enumerate(efficient_only, 1):
            print(
                f"Case {idx}/{len(efficient_only)}: {case['strain']} test_set_{case['test_set_index']}"
            )
            total_count += 1
            try:
                result = visualize_strain_with_extractor(
                    client=CLIENT,
                    collection_name=COLLECTION_NAME,
                    strain=case["strain"],
                    feature_extractor=EfficientNetB1Extractor(),
                    extractor_name="EfficientNetV2B0",
                    test_set_index=case["test_set_index"],
                    output_dir=efficient_dir,
                    k=K,
                )
                if result:
                    success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
        print()
    else:
        print("[2/3] No EfficientNetV2B0 corrections to visualize")
        print()

    # 3. Generate ColorHistogramHS wrong cases
    print(
        f"[3/3] Generating {len(all_colorhist_wrong)} ColorHistogramHS wrong visualizations..."
    )
    print("-" * 80)
    for idx, case in enumerate(all_colorhist_wrong, 1):
        print(
            f"Case {idx}/{len(all_colorhist_wrong)}: {case['strain']} test_set_{case['test_set_index']}"
        )
        total_count += 1
        try:
            result = visualize_strain_with_extractor(
                client=CLIENT,
                collection_name=COLLECTION_NAME,
                strain=case["strain"],
                feature_extractor=ColorHistogramHSExtractor(),
                extractor_name="ColorHistogramHS",
                test_set_index=case["test_set_index"],
                output_dir=wrong_dir,
                k=K,
            )
            if result:
                success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
    print()

    # Summary
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nSuccess: {success_count}/{total_count} visualizations created")
    print("\nGenerated visualizations in:")
    if resnet_only:
        print(f"  - {resnet_dir}/")
        print(f"    ({len(resnet_only)} ResNet50 correction cases)")
    if efficient_only:
        print(f"  - {efficient_dir}/")
        print(f"    ({len(efficient_only)} EfficientNetV2B0 correction cases)")
    print(f"  - {wrong_dir}/")
    print(f"    ({len(all_colorhist_wrong)} ColorHistogramHS wrong cases)")
    print()


if __name__ == "__main__":
    main()
