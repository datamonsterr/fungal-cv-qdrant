import os
import argparse
from datetime import datetime
from typing import List, Optional
from qdrant_client import QdrantClient

from src.config import (
    QDRANT_URL, COLLECTION_NAME, RESULTS_DIR
)
from src.feature_extraction.feature_extractors import (
    ResNet50Extractor, MobileNetV2Extractor, EfficientNetV2B0Extractor,
    HOGExtractor, GaborExtractor, ColorHistogramExtractor,
    FeatureExtractor
)
from src.classification.evaluate_species import run_species_evaluation
from src.classification.visualization.visualize_prediction import batch_visualize_predictions

def get_extractor_by_name(name: str) -> Optional[FeatureExtractor]:
    name = name.lower()
    if name == "resnet50":
        return ResNet50Extractor()
    elif name == "mobilenetv2":
        return MobileNetV2Extractor()
    elif name == "efficientnetv2":
        return EfficientNetV2B0Extractor()
    elif name == "hog":
        return HOGExtractor()
    elif name == "gabor":
        return GaborExtractor()
    elif name == "colorhistogram":
        return ColorHistogramExtractor()
    else:
        return None

def run_comprehensive_report(
    identifier: str,
    extractors: List[str],
    env_strategies: List[str],
    agg_strategies: List[str],
    k: int = 5,
    max_visualizations: int = 20,
    visualize_correct: bool = True,
    visualize_incorrect: bool = True
):
    print(f"Starting Comprehensive Report: {identifier}")
    print(f"Extractors: {extractors}")
    print(f"Environment Strategies: {env_strategies}")
    print(f"Aggregation Strategies: {agg_strategies}")
    print(f"K: {k}")
    print(f"Max Visualizations: {max_visualizations}")
    print(f"Visualize Correct: {visualize_correct}")
    print(f"Visualize Incorrect: {visualize_incorrect}")
    
    client = QdrantClient(url=QDRANT_URL)
    
    base_output_dir = RESULTS_DIR / identifier
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for ext_name in extractors:
        extractor = get_extractor_by_name(ext_name)
        if not extractor:
            print(f"Warning: Unknown extractor {ext_name}, skipping.")
            continue
            
        for env_strat in env_strategies:
            for agg_strat in agg_strategies:
                # Construct subfolder name
                # {feature_extractor}_{strategy environemnt}_{aggregate strategy}
                subfolder_name = f"{ext_name}_{env_strat}_{agg_strat}"
                output_dir = base_output_dir / subfolder_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"\nRunning evaluation for: {subfolder_name}")
                
                # Map env_strategy to environment parameter
                if env_strat == "E1":
                    environment_param = None
                elif env_strat == "E2":
                    environment_param = "all"
                else:
                    environment_param = env_strat
                
                try:
                    results, report_path = run_species_evaluation(
                        client=client,
                        collection_name=COLLECTION_NAME,
                        feature_extractor=extractor,
                        k=k,
                        without_siblings=True, # Default assumption
                        environment=environment_param,
                        strategy=agg_strat,
                        output_dir=str(output_dir)
                    )
                    
                    # Generate Visualizations
                    print("Generating visualizations...")
                    
                    if visualize_correct:
                        print("  Visualizing correct predictions...")
                        batch_visualize_predictions(
                            prediction_results=results,
                            segmented_image_dir=str(os.path.join(os.getcwd(), "Dataset/segmented_image")),
                            output_dir=str(output_dir / "visualizations" / "correct"),
                            k=k,
                            filter_correct=True,
                            max_visualizations=max_visualizations
                        )

                    if visualize_incorrect:
                        print("  Visualizing incorrect predictions...")
                        batch_visualize_predictions(
                            prediction_results=results,
                            segmented_image_dir=str(os.path.join(os.getcwd(), "Dataset/segmented_image")),
                            output_dir=str(output_dir / "visualizations" / "incorrect"),
                            k=k,
                            filter_correct=False,
                            max_visualizations=max_visualizations
                        )
                    
                except Exception as e:
                    print(f"Error running evaluation for {subfolder_name}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive evaluation report")
    parser.add_argument("--identifier", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Unique identifier for this run")
    
    # Default to all extractors
    all_extractors = ["resnet50", "mobilenetv2", "efficientnetv2", "hog", "gabor", "colorhistogram"]
    parser.add_argument("--extractors", nargs="+", default=all_extractors, help="List of feature extractors to use")
    
    parser.add_argument("--env_strategies", nargs="+", default=["E1","E2"], help="List of environment strategies (E1, E2, E3_MEA, E4_CYA, etc.)")
    parser.add_argument("--agg_strategies", nargs="+", default=["avg", "uni"], help="List of aggregation strategies (avg, uni)")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors (k)")
    parser.add_argument("--max_visualizations", type=int, default=20, help="Max number of visualizations per run")
    parser.add_argument("--no-viz-correct", action="store_false", dest="visualize_correct", help="Disable visualization of correct predictions")
    parser.add_argument("--no-viz-incorrect", action="store_false", dest="visualize_incorrect", help="Disable visualization of incorrect predictions")
    parser.set_defaults(visualize_correct=True, visualize_incorrect=True)
    
    args = parser.parse_args()
    
    run_comprehensive_report(
        identifier=args.identifier,
        extractors=args.extractors,
        env_strategies=args.env_strategies,
        agg_strategies=args.agg_strategies,
        k=args.k,
        max_visualizations=args.max_visualizations,
        visualize_correct=args.visualize_correct,
        visualize_incorrect=args.visualize_incorrect
    )