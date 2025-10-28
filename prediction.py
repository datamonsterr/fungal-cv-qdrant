"""
Prediction system for myco fungi species classification using Qdrant vector database.
"""
import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import numpy as np
import cv2
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from feature_extractors import FeatureExtractor
from query_utils import find_nearest_neighbors_by_id, get_image_metadata, visualize_neighbors


def load_strain_to_species_mapping(csv_path: str = "../Dataset/strain_to_specy.csv") -> Dict[str, str]:
    """
    Load strain to species mapping from CSV file.
    
    Args:
        csv_path: Path to the strain_to_specy.csv file
        
    Returns:
        Dictionary mapping strain names to species names
    """
    strain_to_specy = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strain_to_specy[row['Strain']] = row['Species']
    return strain_to_specy


def get_all_images_for_strain(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all images for a specific strain from the collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        strain: Strain name to query
        environment: Optional environment filter
        
    Returns:
        List of image metadata dictionaries
    """
    conditions = [FieldCondition(key="strain", match=MatchValue(value=strain))]
    
    if environment and environment.lower() != "all":
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )
    
    search_filter = Filter(must=conditions)
    
    # Scroll through all matching images
    all_images = []
    offset = None
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=100,
            offset=offset,
            with_payload=True
        )
        
        points, next_offset = result
        
        for point in points:
            payload = point.payload  # type: ignore
            image_data = {
                'image_id': payload.get('image_id'),
                'strain': payload.get('strain'),
                'environment': payload.get('environment'),
                'angle': payload.get('angle'),
                'specy': payload.get('specy'),
                'parent_id': payload.get('parent_id'),
                'segment_index': payload.get('segment_index'),
                'bbox': payload.get('bbox'),
            }
            all_images.append(image_data)
        
        if next_offset is None:
            break
        offset = next_offset
    
    return all_images


def get_available_strains(
    client: QdrantClient,
    collection_name: str
) -> List[str]:
    """
    Get all unique strain names available in the collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        
    Returns:
        Sorted list of unique strain names
    """
    all_strains = set()
    offset = None
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True
        )
        
        points, next_offset = result
        
        for point in points:
            strain = point.payload.get('strain')  # type: ignore
            if strain:
                all_strains.add(strain)
        
        if next_offset is None:
            break
        offset = next_offset
    
    return sorted(list(all_strains))


def filter_siblings(
    neighbors: List[Dict[str, Any]],
    query_parent_id: str
) -> List[Dict[str, Any]]:
    """
    Filter out neighbors that have the same parent_id as the query image.
    
    Args:
        neighbors: List of neighbor dictionaries
        query_parent_id: Parent ID of the query image
        
    Returns:
        Filtered list of neighbors
    """
    return [n for n in neighbors if n.get('parent_id') != query_parent_id]


def aggregate_predictions(
    all_results: List[Dict[str, Any]],
    strain_to_specy: Dict[str, str],
    k: int
) -> List[Tuple[str, float]]:
    """
    Aggregate predictions using voting strategy.
    
    Args:
        all_results: List of all neighbor results from all query images
        strain_to_specy: Mapping from strain to species
        k: Number of top neighbors to consider per query
        
    Returns:
        List of (species, score) tuples sorted by score in descending order
    """
    # Count species votes from top-k neighbors
    species_votes: Counter = Counter()
    
    for result in all_results:
        # Get the strain from the neighbor
        neighbor_strain = result.get('strain', 'unknown')
        # Map strain to species
        neighbor_specy = strain_to_specy.get(neighbor_strain, 'unknown')
        
        # Weight by similarity score (optional: could use uniform voting)
        score = result.get('score', 0.0)
        species_votes[neighbor_specy] += score
    
    # Normalize scores
    total_score = sum(species_votes.values())
    if total_score > 0:
        normalized_results = [
            (specy, count / total_score) 
            for specy, count in species_votes.items()
        ]
    else:
        normalized_results = [(specy, 0.0) for specy in species_votes.keys()]
    
    # Sort by score in descending order
    normalized_results.sort(key=lambda x: x[1], reverse=True)
    
    return normalized_results


def predict(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strain_to_specy_path: str = "../Dataset/strain_to_specy.csv",
    segmented_image_dir: str = "../Dataset/segmented_image",
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Predict species for a given strain using k-nearest neighbors.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        strain: Strain name to predict
        feature_extractor: Feature extractor instance
        k: Number of nearest neighbors to consider
        without_siblings: Whether to exclude images with same parent_id
        environment: Environment filter (None for same as query, "all" for no filter)
        strain_to_specy_path: Path to strain-to-species mapping CSV
        segmented_image_dir: Directory containing segmented images
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing raw results, aggregated results, and ground truth
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load strain to species mapping
    strain_to_specy = load_strain_to_species_mapping(strain_to_specy_path)
    
    # Get ground truth species for the query strain
    ground_truth_specy = strain_to_specy.get(strain, 'unknown')
    
    print(f"\n{'='*80}")
    print(f"Predicting species for strain: {strain}")
    print(f"Ground truth: {ground_truth_specy}")
    print(f"Feature extractor: {feature_extractor.name}")
    print(f"K: {k}, Without siblings: {without_siblings}, Environment: {environment}")
    print(f"{'='*80}\n")
    
    # Get all images for this strain
    strain_images = get_all_images_for_strain(
        client=client,
        collection_name=collection_name,
        strain=strain,
        environment=environment
    )
    
    if not strain_images:
        print(f"No images found for strain {strain}")
        return {
            'strain': strain,
            'ground_truth': ground_truth_specy,
            'predicted_specy': 'unknown',
            'predicted_confidence': 0.0,
            'correct': False,
            'num_query_images': 0,
            'num_neighbors_total': 0,
            'raw_results': [],
            'aggregated_results': [],
            'feature_extractor': feature_extractor.name,
            'k': k,
            'without_siblings': without_siblings,
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'error': 'No images found'
        }
    
    print(f"Found {len(strain_images)} images for strain {strain}")
    
    # Query for each image and collect results
    all_neighbors = []
    raw_results_per_image = []
    
    for idx, image_data in enumerate(strain_images):
        image_id = image_data['image_id']
        parent_id = image_data.get('parent_id', 'unknown')
        img_environment = image_data.get('environment', 'unknown')
        
        print(f"  Querying image {idx+1}/{len(strain_images)}: {image_id}")
        
        # Determine environment filter for search
        if environment is None:
            # Use same environment as query image
            search_environment = img_environment
        elif environment.lower() == "all":
            # No environment filter
            search_environment = None
        else:
            # Use specified environment
            search_environment = environment
        
        try:
            # Find nearest neighbors
            neighbors = find_nearest_neighbors_by_id(
                client=client,
                collection_name=collection_name,
                query_image_id=image_id,
                feature_type=feature_extractor.name.lower(),
                num_neighbors=k * 2 if without_siblings else k,  # Get more if filtering
                environment=search_environment,
                exclude_self=True
            )
            
            # Filter siblings if requested
            if without_siblings:
                neighbors = filter_siblings(neighbors, parent_id)
            
            # Limit to k neighbors
            neighbors = neighbors[:k]
            
            # Store raw results for this image
            raw_results_per_image.append({
                'query_image_id': image_id,
                'query_parent_id': parent_id,
                'query_environment': img_environment,
                'neighbors': neighbors
            })
            
            # Add to all neighbors list
            all_neighbors.extend(neighbors)
            
        except Exception as e:
            print(f"    Error querying image {image_id}: {e}")
            continue
    
    # Aggregate predictions
    aggregated_results = aggregate_predictions(all_neighbors, strain_to_specy, k)
    
    # Get predicted species (top-1)
    predicted_specy = aggregated_results[0][0] if aggregated_results else 'unknown'
    predicted_confidence = aggregated_results[0][1] if aggregated_results else 0.0
    
    print(f"\nAggregated predictions:")
    for specy, score in aggregated_results[:5]:  # Show top 5
        marker = " ✓" if specy == ground_truth_specy else ""
        print(f"  {specy}: {score:.4f}{marker}")
    
    print(f"\nPredicted: {predicted_specy} (confidence: {predicted_confidence:.4f})")
    print(f"Ground truth: {ground_truth_specy}")
    print(f"Correct: {predicted_specy == ground_truth_specy}")
    
    # Prepare result dictionary
    result = {
        'strain': strain,
        'ground_truth': ground_truth_specy,
        'predicted_specy': predicted_specy,
        'predicted_confidence': predicted_confidence,
        'correct': predicted_specy == ground_truth_specy,
        'num_query_images': len(strain_images),
        'num_neighbors_total': len(all_neighbors),
        'raw_results': raw_results_per_image,
        'aggregated_results': [{'specy': s, 'score': sc} for s, sc in aggregated_results],
        'feature_extractor': feature_extractor.name,
        'k': k,
        'without_siblings': without_siblings,
        'environment': environment,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_str = environment if environment else "same"
    sibling_str = "no_siblings" if without_siblings else "with_siblings"
    filename = f"prediction_{strain.replace(' ', '_')}_{feature_extractor.name.lower()}_k{k}_{env_str}_{sibling_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    # Visualize false predictions
    if not result['correct']:
        visualize_false_predictions(
            client=client,
            collection_name=collection_name,
            result=result,
            segmented_image_dir=segmented_image_dir,
            output_dir=output_dir,
            feature_extractor=feature_extractor
        )
    
    return result


def visualize_false_predictions(
    client: QdrantClient,
    collection_name: str,
    result: Dict[str, Any],
    segmented_image_dir: str,
    output_dir: str,
    feature_extractor: FeatureExtractor,
    max_visualizations: int = 3
) -> None:
    """
    Create visualizations for incorrectly predicted images.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        result: Result dictionary from predict()
        segmented_image_dir: Directory containing segmented images
        output_dir: Directory to save visualizations
        feature_extractor: Feature extractor instance
        max_visualizations: Maximum number of visualizations to create
    """
    print(f"\n{'='*80}")
    print(f"Creating visualizations for false prediction: {result['strain']}")
    print(f"Predicted: {result['predicted_specy']}, Ground truth: {result['ground_truth']}")
    print(f"{'='*80}\n")
    
    raw_results = result['raw_results']
    
    # Create visualizations for first few query images
    for idx, image_result in enumerate(raw_results[:max_visualizations]):
        query_image_id = image_result['query_image_id']
        neighbors = image_result['neighbors']
        
        query_image_path = os.path.join(segmented_image_dir, f"{query_image_id}.jpg")
        
        if not os.path.exists(query_image_path):
            print(f"  Skipping {query_image_id}: image file not found")
            continue
        
        # Get query metadata
        query_metadata = get_image_metadata(
            client=client,
            collection_name=collection_name,
            image_id=query_image_id
        )
        
        # Create output path
        output_filename = f"false_pred_{result['strain'].replace(' ', '_')}_{query_image_id}_{feature_extractor.name.lower()}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create visualization
        try:
            visualize_neighbors(
                query_image_path=query_image_path,
                neighbors=neighbors,
                segmented_image_dir=segmented_image_dir,
                output_path=output_path,
                query_metadata=query_metadata,
                max_neighbors=min(7, len(neighbors))
            )
            print(f"  Visualization saved: {output_filename}")
        except Exception as e:
            print(f"  Error creating visualization for {query_image_id}: {e}")


def draw_confusion_matrix(
    predictions: List[Dict[str, Any]],
    output_path: str = "./results/confusion_matrix.png",
    figsize: Tuple[int, int] = (12, 10)
) -> None:
    """
    Draw confusion matrix from prediction results.
    
    Args:
        predictions: List of prediction result dictionaries
        output_path: Path to save the confusion matrix plot
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("Error: matplotlib, seaborn, and scikit-learn are required for confusion matrix.")
        print("Install with: uv pip install matplotlib seaborn scikit-learn")
        return
    
    # Extract ground truth and predictions
    y_true = []
    y_pred = []
    
    for pred in predictions:
        if pred.get('ground_truth') and pred.get('predicted_specy'):
            y_true.append(pred['ground_truth'])
            y_pred.append(pred['predicted_specy'])
    
    if not y_true:
        print("No valid predictions to create confusion matrix")
        return
    
    # Get unique labels sorted alphabetically
    labels = sorted(list(set(y_true + y_pred)))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})', fontsize=14, fontweight='bold')
    plt.ylabel('True Species', fontsize=12)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    
    plt.close()


def batch_predict(
    client: QdrantClient,
    collection_name: str,
    strains: List[str],
    feature_extractor: FeatureExtractor,
    k: int = 5,
    without_siblings: bool = True,
    environment: Optional[str] = None,
    strain_to_specy_path: str = "../Dataset/strain_to_specy.csv",
    segmented_image_dir: str = "../Dataset/segmented_image",
    output_dir: str = "./results"
) -> List[Dict[str, Any]]:
    """
    Run predictions for multiple strains and create confusion matrix.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        strains: List of strain names to predict
        feature_extractor: Feature extractor instance
        k: Number of nearest neighbors to consider
        without_siblings: Whether to exclude images with same parent_id
        environment: Environment filter
        strain_to_specy_path: Path to strain-to-species mapping CSV
        segmented_image_dir: Directory containing segmented images
        output_dir: Directory to save results
        
    Returns:
        List of prediction result dictionaries
    """
    all_results = []
    
    # Get available strains in database
    available_strains = set(get_available_strains(client, collection_name))
    
    # Filter to only strains that exist in database
    valid_strains = [s for s in strains if s in available_strains]
    skipped_strains = [s for s in strains if s not in available_strains]
    
    if skipped_strains:
        print(f"\n⚠ Warning: {len(skipped_strains)} strains not found in database and will be skipped:")
        for strain in skipped_strains[:5]:  # Show first 5
            print(f"  - {strain}")
        if len(skipped_strains) > 5:
            print(f"  ... and {len(skipped_strains) - 5} more")
    
    print(f"\n{'='*80}")
    print(f"Running batch prediction for {len(valid_strains)} strains (out of {len(strains)} requested)")
    print(f"{'='*80}\n")
    
    for idx, strain in enumerate(valid_strains):
        print(f"\nProcessing strain {idx+1}/{len(valid_strains)}: {strain}")
        
        try:
            result = predict(
                client=client,
                collection_name=collection_name,
                strain=strain,
                feature_extractor=feature_extractor,
                k=k,
                without_siblings=without_siblings,
                environment=environment,
                strain_to_specy_path=strain_to_specy_path,
                segmented_image_dir=segmented_image_dir,
                output_dir=output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing strain {strain}: {e}")
            continue
    
    # Create confusion matrix
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_str = environment if environment else "same"
        sibling_str = "no_siblings" if without_siblings else "with_siblings"
        cm_filename = f"confusion_matrix_{feature_extractor.name.lower()}_k{k}_{env_str}_{sibling_str}_{timestamp}.png"
        cm_path = os.path.join(output_dir, cm_filename)
        
        draw_confusion_matrix(all_results, output_path=cm_path)
        
        # Save summary
        summary_filename = f"batch_summary_{feature_extractor.name.lower()}_k{k}_{env_str}_{sibling_str}_{timestamp}.json"
        summary_path = os.path.join(output_dir, summary_filename)
        
        summary = {
            'total_strains': len(strains),
            'successful_predictions': len(all_results),
            'correct_predictions': sum(1 for r in all_results if r.get('correct', False)),
            'accuracy': sum(1 for r in all_results if r.get('correct', False)) / len(all_results) if all_results else 0,
            'feature_extractor': feature_extractor.name,
            'k': k,
            'without_siblings': without_siblings,
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Batch prediction complete!")
        print(f"Total strains: {summary['total_strains']}")
        print(f"Successful predictions: {summary['successful_predictions']}")
        print(f"Correct predictions: {summary['correct_predictions']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")
    
    return all_results


# Example usage
if __name__ == "__main__":
    from feature_extractors import ResNet50Extractor, HOGExtractor, ColorHistogramExtractor, GaborExtractor
    
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Load all strains
    strain_to_specy = load_strain_to_species_mapping()
    all_strains = list(strain_to_specy.keys())
    
    # Test with one strain (use a strain that exists in database)
    test_strain = "DTO 148-C8"  # Change this to any strain in your database
    extractor = ResNet50Extractor()
    
    result = predict(
        client=client,
        collection_name=collection_name,
        strain=test_strain,
        feature_extractor=extractor,
        k=5,
        without_siblings=True,
        environment=None  # Use same environment as query
    )
    
    # Batch prediction example (commented out)
    # batch_results = batch_predict(
    #     client=client,
    #     collection_name=collection_name,
    #     strains=all_strains[:10],  # Test with first 10 strains
    #     feature_extractor=extractor,
    #     k=5,
    #     without_siblings=True,
    #     environment=None
    # )
