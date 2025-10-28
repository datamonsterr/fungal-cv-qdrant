"""
Quick species evaluation - simplified version.
"""
from qdrant_client import QdrantClient
from feature_extractors import ResNet50Extractor
from evaluate_species import run_species_evaluation, compare_extractors


def main():
    """Run quick species evaluation."""
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    print("\n" + "="*80)
    print("QUICK SPECIES EVALUATION")
    print("="*80 + "\n")
    
    # Run evaluation with ResNet50
    print("Running evaluation with ResNet50 (one strain per species)...\n")
    
    results, report_paths = run_species_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractor=ResNet50Extractor(),
        k=5,
        without_siblings=True,
        environment=None,  # Use same environment as query
        output_dir="./results/species_evaluation"
    )
    
    # Print quick summary
    if results:
        correct = sum(1 for r in results if r.get('correct', False))
        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0
        
        print("\n" + "="*80)
        print("QUICK SUMMARY")
        print("="*80)
        print(f"\nTotal species tested: {total}")
        print(f"Correct predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"\nGenerated {len(report_paths)} report files:")
        for path in report_paths:
            print(f"  - {path}")
        print("\n")


if __name__ == "__main__":
    main()
