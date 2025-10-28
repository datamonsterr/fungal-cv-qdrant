"""
Example usage of the prediction system for myco fungi classification.
"""
from qdrant_client import QdrantClient
from feature_extractors import (
    ResNet50Extractor, 
    HOGExtractor, 
    ColorHistogramExtractor, 
    GaborExtractor
)
from prediction import (
    predict,
    batch_predict,
    load_strain_to_species_mapping,
    draw_confusion_matrix
)


def example_single_prediction():
    """Example: Predict a single strain."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Strain Prediction")
    print("="*80 + "\n")
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Choose feature extractor
    extractor = ResNet50Extractor()
    
    # Predict for a specific strain
    result = predict(
        client=client,
        collection_name=collection_name,
        strain="DTO 148-C8",  # Change to any strain in your database
        feature_extractor=extractor,
        k=5,
        without_siblings=True,
        environment=None  # Use same environment as query image
    )
    
    print(f"\nPrediction result:")
    print(f"  Strain: {result['strain']}")
    print(f"  Ground truth: {result['ground_truth']}")
    print(f"  Predicted: {result['predicted_specy']}")
    print(f"  Confidence: {result['predicted_confidence']:.4f}")
    print(f"  Correct: {result['correct']}")


def example_single_prediction_all_environments():
    """Example: Predict using all environments (no environment filter)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Single Strain Prediction (All Environments)")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    extractor = HOGExtractor()
    
    result = predict(
        client=client,
        collection_name=collection_name,
        strain="DTO 148-C9",  # Change to any strain in your database
        feature_extractor=extractor,
        k=7,
        without_siblings=True,
        environment="all"  # Search across all environments
    )


def example_single_prediction_with_siblings():
    """Example: Predict including siblings (same parent_id)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Single Strain Prediction (With Siblings)")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    extractor = GaborExtractor()
    
    result = predict(
        client=client,
        collection_name=collection_name,
        strain="DTO 148-D1",  # Change to any strain in your database
        feature_extractor=extractor,
        k=5,
        without_siblings=False,  # Include siblings
        environment=None
    )


def example_batch_prediction_subset():
    """Example: Batch prediction on a subset of strains."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Prediction (Subset of Strains)")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Load strain mapping
    strain_to_specy = load_strain_to_species_mapping()
    all_strains = list(strain_to_specy.keys())
    
    # Select first 10 strains for testing
    test_strains = all_strains[:10]
    
    extractor = ResNet50Extractor()
    
    results = batch_predict(
        client=client,
        collection_name=collection_name,
        strains=test_strains,
        feature_extractor=extractor,
        k=5,
        without_siblings=True,
        environment=None
    )
    
    # Print summary
    correct = sum(1 for r in results if r.get('correct', False))
    print(f"\nBatch Summary:")
    print(f"  Total strains: {len(results)}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/len(results):.2%}")


def example_batch_prediction_all_strains():
    """Example: Batch prediction on all strains."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Prediction (All Strains)")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Load all strains
    strain_to_specy = load_strain_to_species_mapping()
    all_strains = list(strain_to_specy.keys())
    
    print(f"Running prediction on {len(all_strains)} strains...")
    
    extractor = ResNet50Extractor()
    
    results = batch_predict(
        client=client,
        collection_name=collection_name,
        strains=all_strains,
        feature_extractor=extractor,
        k=5,
        without_siblings=True,
        environment=None
    )


def example_compare_extractors():
    """Example: Compare different feature extractors on the same strains."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Compare Feature Extractors")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Load strains
    strain_to_specy = load_strain_to_species_mapping()
    test_strains = list(strain_to_specy.keys())[:5]  # Test on 5 strains
    
    # Test different extractors
    extractors = [
        ResNet50Extractor(),
        HOGExtractor(),
        ColorHistogramExtractor(),
        GaborExtractor()
    ]
    
    results_by_extractor = {}
    
    for extractor in extractors:
        print(f"\n{'='*80}")
        print(f"Testing {extractor.name}")
        print(f"{'='*80}")
        
        results = batch_predict(
            client=client,
            collection_name=collection_name,
            strains=test_strains,
            feature_extractor=extractor,
            k=5,
            without_siblings=True,
            environment=None
        )
        
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / len(results) if results else 0
        
        results_by_extractor[extractor.name] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results)
        }
    
    # Print comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    print(f"{'Extractor':<25} {'Accuracy':<15} {'Correct/Total'}")
    print("-" * 60)
    for name, stats in results_by_extractor.items():
        print(f"{name:<25} {stats['accuracy']:<14.2%} {stats['correct']}/{stats['total']}")


def example_specific_environment():
    """Example: Predict for a specific environment only."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Prediction with Specific Environment Filter")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    extractor = ResNet50Extractor()
    
    # Predict using only CYA environment
    result = predict(
        client=client,
        collection_name=collection_name,
        strain="DTO 148-D2",  # Change to any strain in your database
        feature_extractor=extractor,
        k=5,
        without_siblings=True,
        environment="CYA"  # Only search in CYA environment
    )


def example_different_k_values():
    """Example: Test different k values."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Compare Different K Values")
    print("="*80 + "\n")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    extractor = ResNet50Extractor()
    test_strain = "DTO 148-D3"  # Change to any strain in your database
    
    k_values = [1, 3, 5, 7, 10]
    
    print(f"Testing strain: {test_strain}")
    print(f"{'K':<10} {'Predicted':<30} {'Confidence':<15} {'Correct'}")
    print("-" * 70)
    
    for k in k_values:
        result = predict(
            client=client,
            collection_name=collection_name,
            strain=test_strain,
            feature_extractor=extractor,
            k=k,
            without_siblings=True,
            environment=None
        )
        
        print(f"{k:<10} {result['predicted_specy']:<30} "
              f"{result['predicted_confidence']:<14.4f} {result['correct']}")


if __name__ == "__main__":
    # Uncomment the example you want to run
    
    # Basic examples
    example_single_prediction()
    # example_single_prediction_all_environments()
    # example_single_prediction_with_siblings()
    
    # Batch examples
    # example_batch_prediction_subset()
    # example_batch_prediction_all_strains()
    
    # Comparison examples
    # example_compare_extractors()
    # example_specific_environment()
    # example_different_k_values()
