"""
Example script to run comprehensive evaluation with all strategy combinations.

This script demonstrates how to:
1. Test multiple feature extractors
2. Test both aggregation strategies (S1: avg, S2: uni)
3. Test environment strategies (E1: same, E2: all, E3: specific, E4: exclude)
4. Generate a summary table with all results
"""

from qdrant_client import QdrantClient
from evaluate_species import run_comprehensive_evaluation
from list_env import get_environment_list
from feature_extractors import (
    ResNet50Extractor, 
    ColorHistogramExtractor,
    ColorHistogramHSExtractor,
    HOGExtractor,
    GaborExtractor,
    MobileNetV2Extractor,
    EfficientNetV2B0Extractor,
    ColorHistogramHSconcatResnet50
)

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "myco_fungi_concat_features_manual"

# Configuration
K = 3  # Fixed K=3 as specified
MIN_SAMPLES = None  # Set to an integer to enable minimum sample constraint
WITHOUT_SIBLINGS = True

# Paths to fine-tuned model weights
# Set to None to use default ImageNet weights instead
RESNET50_WEIGHTS = "./weights/resnet50/20251229_214513/best_model.h5"
MOBILENETV2_WEIGHTS = "./weights/mobilenetv2/20251229_223112/best_model.h5"
EFFICIENTNETV2B0_WEIGHTS = "./weights/efficientnetv2b0/20251229_225347/best_model.h5"

# List of feature extractors to test
# You can comment out extractors you don't want to test
FEATURE_EXTRACTORS = [
    ColorHistogramHSconcatResnet50(hist_weight=3.0, bins=32),
    ColorHistogramHSExtractor(),
    ColorHistogramExtractor(),
    ResNet50Extractor(weights_path=RESNET50_WEIGHTS),  # Using fine-tuned weights
    MobileNetV2Extractor(weights_path=MOBILENETV2_WEIGHTS),  # Using fine-tuned weights
    EfficientNetV2B0Extractor(weights_path=EFFICIENTNETV2B0_WEIGHTS),  # Using fine-tuned weights
    HOGExtractor(),
    GaborExtractor()
]

# Get available environments from the database
print("Fetching available environments from database...")
TEST_ENVIRONMENTS = get_environment_list(client, collection_name)
print(f"Found environments: {TEST_ENVIRONMENTS}\n")

# Output directory
# Include K, WITHOUT_SIBLINGS, and MIN_SAMPLES in the folder name
sibling_suffix = "NoSib" if WITHOUT_SIBLINGS else "WithSib"
min_samples_suffix = f"_M{MIN_SAMPLES}" if MIN_SAMPLES is not None else ""
OUTPUT_DIR = f"./results/new_all_finetuned_k{K}_{sibling_suffix}{min_samples_suffix}_4"

print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION - ALL STRATEGY COMBINATIONS")
print("="*80)
print("\nThis will test:")
print("  Feature Extractors:")
for fe in FEATURE_EXTRACTORS:
    print(f"    - {fe.name}")
print("\n  Environment Strategies:")
print("    - E1: Same as query image")
print("    - E2: All environments")
if TEST_ENVIRONMENTS:
    print(f"    - E3: Specific environments ({', '.join(TEST_ENVIRONMENTS)})")
    print(f"    - E4: Exclude environments ({', '.join(TEST_ENVIRONMENTS)})")
else:
    print("    - E3: No specific environments found")
    print("    - E4: No environments to exclude")
print("\n  Aggregation Strategies:")
print("    - S1 (avg): Weighted by similarity score")
print("    - S2 (uni): Uniform voting (each match = 1)")
print("\n  Fixed Parameters:")
print(f"    - K: {K}")
print(f"    - Min Samples: {MIN_SAMPLES}")
print(f"    - Without Siblings: {WITHOUT_SIBLINGS}")
print("="*80 + "\n")

total_combinations = len(FEATURE_EXTRACTORS) * 2 * (2 + len(TEST_ENVIRONMENTS) * 2)  # E1, E2, E3 per env, E4 per env
print(f"Total combinations to test: {total_combinations}")
print("This may take a while...\n")

# Run comprehensive evaluation
print("Starting comprehensive evaluation...")
comprehensive_results = run_comprehensive_evaluation(
    client=client,
    collection_name=collection_name,
    feature_extractors=FEATURE_EXTRACTORS,  # Pass list of extractors
    k=K,
    min_samples=MIN_SAMPLES,
    without_siblings=WITHOUT_SIBLINGS,
    test_environments=TEST_ENVIRONMENTS,
    output_dir=OUTPUT_DIR
)

# Print summary
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION COMPLETED!")
print("="*80)
print(f"\nTotal combinations tested: {len(comprehensive_results['summary_data'])}")
print(f"Summary saved to: {comprehensive_results['summary_path']}")

# Find best performing combination
summary_data = comprehensive_results['summary_data']
if summary_data:
    best = max(summary_data, key=lambda x: float(x['Accuracy'].rstrip('%')))
    print(f"\nBest performing combination:")
    print(f"  Feature Extractor: {best['Feature Extractor']}")
    print(f"  Environment: {best['Environment Strategy']}")
    print(f"  Aggregation: {best['Aggregation Strategy']}")
    print(f"  Accuracy: {best['Accuracy']}")
    print(f"  Correct/Total: {best['Correct/Total']}")
    
    # Show top 5 combinations
    print(f"\nTop 5 combinations:")
    sorted_results = sorted(summary_data, key=lambda x: float(x['Accuracy'].rstrip('%')), reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"  {i}. {result['Feature Extractor']:<20} {result['Environment Strategy']:<10} "
              f"{result['Aggregation Strategy']:<10} {result['Accuracy']:<10} ({result['Correct/Total']})")

print("\n")
