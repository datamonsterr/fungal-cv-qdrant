"""
Test script for E4_CREA strategy - excludes CREA environment from search.
"""

from qdrant_client import QdrantClient
from evaluate_species import run_species_evaluation
from feature_extractors import ColorHistogramHSconcatResnet50

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "myco_fungi_concat_features"

# Configuration
K = 7
MIN_SAMPLES = None
WITHOUT_SIBLINGS = True
OUTPUT_DIR = "./results/test_e4_crea"

print("\n" + "="*80)
print("TESTING E4_CREA STRATEGY (Exclude CREA Environment)")
print("="*80)
print("\nThis will test:")
print("  - Feature Extractor: ColorHistogramHSconcatResnet50")
print("  - Strategy: E4_CREA (search all environments EXCEPT CREA)")
print("  - Aggregation: avg (weighted by score)")
print(f"  - K: {K}")
print(f"  - Without Siblings: {WITHOUT_SIBLINGS}")
print("="*80 + "\n")

# Test E4_CREA with avg strategy
print("Running evaluation with E4_CREA strategy...")
results_avg, report_paths_avg = run_species_evaluation(
    client=client,
    collection_name=collection_name,
    feature_extractor=ColorHistogramHSconcatResnet50(hist_weight=3.0, bins=32),
    k=K,
    min_samples=MIN_SAMPLES,
    without_siblings=WITHOUT_SIBLINGS,
    environment="E4_CREA",  # This will exclude CREA from search
    strategy="avg",
    output_dir=OUTPUT_DIR
)

print("\n" + "="*80)
print("E4_CREA TEST COMPLETED!")
print("="*80)
correct_count = sum(1 for r in results_avg if r['correct'])
accuracy = (correct_count / len(results_avg) * 100) if results_avg else 0
print(f"Accuracy: {correct_count}/{len(results_avg)} = {accuracy:.2f}%")
print(f"Results saved to: {OUTPUT_DIR}/")
print("="*80 + "\n")
