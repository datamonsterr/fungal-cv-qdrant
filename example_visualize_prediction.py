"""
Example script showing how to use visualize_prediction.py with evaluation results.
"""
from qdrant_client import QdrantClient
from evaluate_species import run_species_evaluation
from feature_extractors import ResNet50Extractor, HOGExtractor
from visualize_prediction import visualize_prediction_by_environment, batch_visualize_predictions
import os


def example_single_evaluation_with_visualization():
    """
    Example: Run a single evaluation and visualize results.
    """
    # Setup
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "myco_features"
    segmented_image_dir = "../Dataset/myco_segmented"
    
    # Run evaluation with E1 strategy (same environment)
    print("Running E1 evaluation with ResNet50...")
    results, report_paths = run_species_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractor=ResNet50Extractor(),
        k=7,
        min_samples=None,
        without_siblings=True,
        environment=None,  # E1: same environment
        strategy="avg",
        output_dir="./results"
    )
    
    # Create visualizations for false predictions only
    print("\nCreating visualizations for false predictions...")
    false_predictions = [r for r in results if not r['correct']]
    
    if false_predictions:
        output_dir = "./results/visualizations_E1_false"
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, result in enumerate(false_predictions[:5], 1):  # Visualize first 5 false predictions
            output_path = os.path.join(
                output_dir, 
                f"false_{idx}_{result['strain'].replace(' ', '_')}.jpg"
            )
            
            visualize_prediction_by_environment(
                prediction_result=result,
                segmented_image_dir=segmented_image_dir,
                output_path=output_path,
                k=7
            )
    else:
        print("No false predictions found!")


def example_batch_visualization():
    """
    Example: Use batch visualization on existing results.
    """
    # Setup
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "myco_features"
    segmented_image_dir = "../Dataset/myco_segmented"
    
    # Run evaluation
    print("Running E2 evaluation with HOG...")
    results, report_paths = run_species_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractor=HOGExtractor(),
        k=7,
        min_samples=None,
        without_siblings=True,
        environment="all",  # E2: all environments
        strategy="uni",
        output_dir="./results"
    )
    
    # Create visualizations using batch function
    print("\nCreating batch visualizations...")
    
    # Visualize all false predictions
    output_paths = batch_visualize_predictions(
        prediction_results=results,
        segmented_image_dir=segmented_image_dir,
        output_dir="./results/visualizations_E2_false",
        k=7,
        filter_correct=False,  # Only false predictions
        max_visualizations=10  # Limit to 10 visualizations
    )
    
    print(f"\nCreated {len(output_paths)} visualizations")


def example_compare_strategies():
    """
    Example: Compare E1, E2, and E3 strategies with visualizations.
    """
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "myco_features"
    segmented_image_dir = "../Dataset/myco_segmented"
    extractor = ResNet50Extractor()
    
    strategies = [
        ("E1", None),
        ("E2", "all"),
        ("E3_CYA", "CYA"),
    ]
    
    for strategy_name, environment in strategies:
        print(f"\n{'='*80}")
        print(f"Running {strategy_name} evaluation...")
        print(f"{'='*80}")
        
        results, _ = run_species_evaluation(
            client=client,
            collection_name=collection_name,
            feature_extractor=extractor,
            k=7,
            min_samples=None,
            without_siblings=True,
            environment=environment,
            strategy="avg",
            output_dir=f"./results/{strategy_name}"
        )
        
        # Visualize false predictions for each strategy
        output_dir = f"./results/visualizations_{strategy_name}"
        batch_visualize_predictions(
            prediction_results=results,
            segmented_image_dir=segmented_image_dir,
            output_dir=output_dir,
            k=7,
            filter_correct=False,
            max_visualizations=5
        )


def example_visualize_specific_strain():
    """
    Example: Visualize prediction for a specific strain.
    """
    from evaluate_species import predict_segment_group, collect_testset
    
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "myco_features"
    segmented_image_dir = "../Dataset/myco_segmented"
    
    # Predict for a specific strain
    strain = "DTO 123-A1"  # Replace with actual strain
    test_sets = collect_testset(
        client=client,
        collection_name=collection_name,
        strain=strain,
        environment_strategy="E1"
    )
    
    if test_sets:
        # Use first test set
        result = predict_segment_group(
            client=client,
            collection_name=collection_name,
            test_group=test_sets[0],
            strain=strain,
            feature_extractor=ResNet50Extractor(),
            k=7,
            environment=None,  # E1
            strategy="avg"
        )
        
        # Visualize
        output_path = f"./results/viz_{strain.replace(' ', '_')}.jpg"
        visualize_prediction_by_environment(
            prediction_result=result,
            segmented_image_dir=segmented_image_dir,
            output_path=output_path,
            k=7
        )
        
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples for visualization")
    parser.add_argument(
        "--example",
        choices=["single", "batch", "compare", "strain"],
        default="single",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "single":
        example_single_evaluation_with_visualization()
    elif args.example == "batch":
        example_batch_visualization()
    elif args.example == "compare":
        example_compare_strategies()
    elif args.example == "strain":
        example_visualize_specific_strain()
