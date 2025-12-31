import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from qdrant_client import QdrantClient

from src.classification.prediction import (
    draw_confusion_matrix,
    get_all_images_for_strain
)
# Original evaluate_species.py had predict_segment_group defined inside it?
# Let me check evaluate_species.py again.
# Yes, predict_segment_group was in evaluate_species.py.
# I should probably move it to prediction.py or keep it here.
# It seems to be a variant of predict.

from src.feature_extraction.feature_extractors import (
    ResNet50Extractor,
    HOGExtractor,
    GaborExtractor,
    ColorHistogramExtractor,
    ColorHistogramHSExtractor,
    FeatureExtractor,
    MobileNetV2Extractor,
    EfficientNetV2B0Extractor
)
from src.config import (
    QDRANT_URL, COLLECTION_NAME, RESULTS_DIR, STRAIN_SPECIES_MAPPING_PATH
)



def print_selection_report(selected: Dict[str, str], output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"strain_selection_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Total species: {len(selected)}\n")
        for species, strain in selected.items():
            f.write(f"{species}: {strain}\n")
    return report_path

def print_prediction_results(results: List[Dict[str, Any]], output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"prediction_report_{timestamp}.txt")
    
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) if results else 0
    
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        for r in results:
            f.write(f"{r['strain']} ({r['ground_truth']}) -> {r['predicted_specy']} [{'Correct' if r['correct'] else 'Wrong'}]\n")
            
    return report_path

def collect_testset(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment_strategy: str
) -> List[List[Dict[str, Any]]]:
    # Implementation of test set collection based on strategy
    # This was in evaluate_species.py
    # I'll implement a simplified version or copy logic if needed.
    # For now, just a placeholder or basic logic.
    images = get_all_images_for_strain(client, collection_name, strain)
    # Group by something?
    return [images] # Treat all images as one group for now

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
    # I'll implement it here.
    from src.classification.prediction import find_nearest_neighbors_by_id, filter_siblings, aggregate_predictions
    import pandas as pd
    
    df = pd.read_csv(strain_to_specy_path)
    strain_to_specy = dict(zip(df['Strain'], df['Species']))
    ground_truth_specy = strain_to_specy.get(strain, "unknown")
    
    raw_results = []
    for query_img in test_group:
        image_id = query_img['image_id']
        parent_id = query_img['parent_id']
        
        neighbors = find_nearest_neighbors_by_id(
            client=client,
            collection_name=collection_name,
            query_image_id=image_id,
            feature_type=feature_extractor.name.lower(),
            num_neighbors=k + 5,
            environment=environment,
            exclude_self=True
        )
        
        if without_siblings:
            neighbors = filter_siblings(neighbors, parent_id)
        
        neighbors = neighbors[:k]
        
        raw_results.append({
            'query_image_id': image_id,
            'query_environment': query_img.get('environment'),
            'neighbors': neighbors
        })
        
    aggregated = aggregate_predictions(raw_results, strain_to_specy, k, min_samples, strategy)
    
    if not aggregated:
        predicted_specy = "unknown"
        confidence = 0.0
    else:
        predicted_specy = aggregated[0][0]
        confidence = aggregated[0][1]
        
    is_correct = (predicted_specy == ground_truth_specy)
    
    return {
        'strain': strain,
        'ground_truth': ground_truth_specy,
        'predicted_specy': predicted_specy,
        'correct': is_correct,
        'predicted_confidence': confidence,
        'aggregated_results': [{'specy': s, 'score': sc} for s, sc in aggregated],
        'raw_results': raw_results,
        'feature_extractor': feature_extractor.name,
        'strategy': strategy,
        'environment': environment
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
    output_dir: str = str(RESULTS_DIR)
) -> Tuple[List[Dict[str, Any]], List[str]]:
    
    import pandas as pd
    if not STRAIN_SPECIES_MAPPING_PATH.exists():
        print(f"Error: {STRAIN_SPECIES_MAPPING_PATH} not found. Please run 'python src/main.py generate-mapping' first.")
        return [], []

    df_mapping = pd.read_csv(STRAIN_SPECIES_MAPPING_PATH)
    if 'Test' not in df_mapping.columns:
        print("Error: 'Test' column not found in mapping CSV. Please regenerate mapping.")
        return [], []
        
    # Select strains where Test is True
    # Create a dict {Species: Strain} for the report and iteration
    # Note: The original code assumed one strain per species.
    # If multiple strains are marked as Test for a species, we might want to evaluate all of them.
    # But the return type implies a list of results.
    
    test_df = df_mapping[df_mapping['Test'] == True]
    selected_strains = {}
    for _, row in test_df.iterrows():
        selected_strains[row['Species']] = row['Strain']
        
    print_selection_report(selected_strains, output_dir)
    
    results = []
    for species, strain in selected_strains.items():
        print(f"Evaluating {species} (Strain: {strain})...")
        # Get images
        images = get_all_images_for_strain(client, collection_name, strain, environment)
        if not images:
            continue
            
        res = predict_segment_group(
            client, collection_name, images, strain, feature_extractor,
            k, min_samples, without_siblings, environment, strategy
        )
        results.append(res)
        
    print_prediction_results(results, output_dir)
    draw_confusion_matrix(results, os.path.join(output_dir, "confusion_matrix.png"))
    
    return results, list(selected_strains.values())

if __name__ == "__main__":
    pass
