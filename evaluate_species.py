"""
Species-level evaluation script.
Takes one strain per species, runs predictions, and generates comprehensive reports.
"""
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from qdrant_client import QdrantClient

from prediction import (
    predict,
    draw_confusion_matrix,
    load_strain_to_species_mapping,
    get_available_strains,
    get_all_images_for_strain
)
from feature_extractors import (
    ResNet50Extractor,
    HOGExtractor,
    GaborExtractor,
    ColorHistogramExtractor,
    ColorHistogramHSExtractor,
    FeatureExtractor,
    MobileNetV2Extractor,
    EfficientNetV2B0Extractor
)


def select_one_strain_per_species(
    available_strains: List[str],
    strain_to_specy: Dict[str, str]
) -> Dict[str, str]:
    """
    Select one strain for each species from available strains.
    
    Args:
        available_strains: List of strains available in database
        strain_to_specy: Mapping from strain to species
        
    Returns:
        Dictionary mapping species to selected strain
    """
    species_to_strains = defaultdict(list)
    
    # Group available strains by species
    for strain in available_strains:
        if strain in strain_to_specy:
            species = strain_to_specy[strain]
            species_to_strains[species].append(strain)
    
    # Select first strain for each species (or could be random)
    selected = {}
    for species, strains in species_to_strains.items():
        if len(strains) > 1:
            selected[species] = strains[1]  # Take first strain
        else:
            selected[species] = strains[0]
    
    return selected


def print_selection_report(
    selected: Dict[str, str],
    output_dir: str
) -> str:
    """
    Print and save strain selection report.
    
    Args:
        selected: Dictionary mapping species to selected strain
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"strain_selection_report_{timestamp}.txt")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("STRAIN SELECTION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total species: {len(selected)}")
    report_lines.append("\n" + "-"*80)
    report_lines.append(f"{'Species':<40} {'Selected Strain'}")
    report_lines.append("-"*80)
    
    for species in sorted(selected.keys()):
        strain = selected[species]
        report_lines.append(f"{species:<40} {strain}")
    
    report_lines.append("-"*80)
    report_lines.append(f"\nTotal strains selected: {len(selected)}")
    report_lines.append("="*80)
    
    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nSelection report saved to: {report_path}\n")
    return report_path


def print_prediction_results(
    results: List[Dict[str, Any]],
    output_dir: str
) -> str:
    """
    Print and save detailed prediction results.
    
    Args:
        results: List of prediction result dictionaries
        output_dir: Directory to save report
        
    Returns:
        Path to saved report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"prediction_results_{timestamp}.txt")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PREDICTION RESULTS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total predictions: {len(results)}")
    
    if results:
        correct = sum(1 for r in results if r.get('correct', False))
        accuracy = correct / len(results) * 100
        report_lines.append(f"Correct predictions: {correct}")
        report_lines.append(f"Accuracy: {accuracy:.2f}%")
        report_lines.append(f"Feature extractor: {results[0].get('feature_extractor', 'unknown')}")
        report_lines.append(f"K: {results[0].get('k', 'unknown')}")
        report_lines.append(f"Without siblings: {results[0].get('without_siblings', 'unknown')}")
        report_lines.append(f"Environment: {results[0].get('environment', 'unknown')}")
    
    # Detailed results for each strain
    for idx, result in enumerate(results, 1):
        report_lines.append("\n" + "="*80)
        report_lines.append(f"PREDICTION #{idx}: {result['strain']}")
        report_lines.append("="*80)
        
        # Basic info
        report_lines.append(f"\nStrain: {result['strain']}")
        report_lines.append(f"Ground Truth: {result['ground_truth']}")
        report_lines.append(f"Predicted: {result['predicted_specy']}")
        report_lines.append(f"Confidence: {result['predicted_confidence']:.4f} ({result['predicted_confidence']*100:.2f}%)")
        report_lines.append(f"Correct: {'✓ YES' if result['correct'] else '✗ NO'}")
        report_lines.append(f"Query Images: {result['num_query_images']}")
        report_lines.append(f"Total Neighbors: {result['num_neighbors_total']}")
        
        # Aggregated results (top 5)
        report_lines.append("\nAggregated Predictions (Top 5):")
        report_lines.append("-"*80)
        report_lines.append(f"{'Rank':<6} {'Species':<40} {'Score':<12} {'%'}")
        report_lines.append("-"*80)
        
        for rank, agg_result in enumerate(result['aggregated_results'][:5], 1):
            specy = agg_result['specy']
            score = agg_result['score']
            percentage = score * 100
            marker = " ✓" if specy == result['ground_truth'] else ""
            report_lines.append(f"{rank:<6} {specy:<40} {score:<12.4f} {percentage:>6.2f}%{marker}")
        
        # Raw results summary (per query image)
        if result['raw_results']:
            report_lines.append("\nRaw Results Summary (Per Query Image):")
            report_lines.append("-"*80)
            report_lines.append(f"{'Image #':<10} {'Image ID':<35} {'Neighbors':<12} {'Environment'}")
            report_lines.append("-"*80)
            
            for img_idx, raw in enumerate(result['raw_results'][:10], 1):  # Show first 10
                img_id = raw['query_image_id']
                num_neighbors = len(raw['neighbors'])
                env = raw.get('query_environment', 'unknown')
                report_lines.append(f"{img_idx:<10} {img_id:<35} {num_neighbors:<12} {env}")
            
            if len(result['raw_results']) > 10:
                report_lines.append(f"... and {len(result['raw_results']) - 10} more query images")
    
    # Summary statistics
    report_lines.append("\n" + "="*80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("="*80)
    
    if results:
        # Species-level statistics
        species_correct = defaultdict(int)
        species_total = defaultdict(int)
        
        for result in results:
            species = result['ground_truth']
            species_total[species] += 1
            if result['correct']:
                species_correct[species] += 1
        
        report_lines.append("\nPer-Species Accuracy:")
        report_lines.append("-"*80)
        report_lines.append(f"{'Species':<40} {'Correct':<10} {'Total':<10} {'Accuracy'}")
        report_lines.append("-"*80)
        
        for species in sorted(species_total.keys()):
            correct = species_correct[species]
            total = species_total[species]
            acc = correct / total * 100 if total > 0 else 0
            report_lines.append(f"{species:<40} {correct:<10} {total:<10} {acc:>6.2f}%")
        
        # Confidence statistics
        report_lines.append("\nConfidence Statistics:")
        report_lines.append("-"*80)
        confidences = [r['predicted_confidence'] for r in results]
        correct_confidences = [r['predicted_confidence'] for r in results if r['correct']]
        incorrect_confidences = [r['predicted_confidence'] for r in results if not r['correct']]
        
        report_lines.append(f"Average confidence (all): {sum(confidences)/len(confidences):.4f}")
        if correct_confidences:
            report_lines.append(f"Average confidence (correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
        if incorrect_confidences:
            report_lines.append(f"Average confidence (incorrect): {sum(incorrect_confidences)/len(incorrect_confidences):.4f}")
    
    report_lines.append("\n" + "="*80)
    
    # Print to console
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\nDetailed results saved to: {report_path}\n")
    return report_path


def collect_testset(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    environment_strategy: str
) -> List[List[Dict[str, Any]]]:
    """
    Collect test sets for a strain based on environment strategy.
    Creates up to 6 test sets, where each test set contains one image per environment.
    
    For E1/E2: Each test set has one image from each environment (e.g., 7 images per set if 7 envs)
    For E3: Each test set has one image from the specific environment (1 image per set)
    
    Args:
        client: Qdrant client instance
        collection_name: Name of Qdrant collection
        strain: Strain name
        environment_strategy: Environment strategy ("E1", "E2", or "E3_<env_name>")
        
    Returns:
        List of test sets, where each test set is a list of image metadata dicts
        
    Example:
        For E1/E2 with 7 environments and 6 images per env:
        Returns 6 test sets, each with 7 images (one per environment)
        
        For E3_PDA with 6 images in PDA environment:
        Returns 6 test sets, each with 1 image from PDA
    """
    # Determine if E3 and extract environment name
    if environment_strategy.startswith("E3_"):
        is_e3 = True
        target_env = environment_strategy[3:]  # Extract environment name after "E3_"
        # Get images from specific environment only
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=target_env
        )
    else:
        # E1 or E2: Get ALL images from ALL environments
        is_e3 = False
        target_env = None
        strain_images = get_all_images_for_strain(
            client=client,
            collection_name=collection_name,
            strain=strain,
            environment=None
        )
    
    if not strain_images:
        return []
    
    if is_e3:
        # E3: Create test sets with one image each from the specific environment
        # Group by segment_index to get different "views"
        test_sets = []
        for img in strain_images[:6]:  # Take up to 6 different images
            test_sets.append([img])
        return test_sets
    else:
        # E1/E2: Create test sets with one image per environment
        # Group images by environment and segment_index
        env_segment_images = defaultdict(lambda: defaultdict(list))
        
        for img in strain_images:
            env = img.get('environment', 'unknown')
            segment_idx = img.get('segment_index', 0)
            env_segment_images[env][segment_idx].append(img)
        
        # Determine how many test sets we can create (max 6)
        # Each test set needs one image from each environment
        max_test_sets = 6
        test_sets = []
        
        # For each segment index (0-5), create a test set
        for test_set_idx in range(max_test_sets):
            test_set = []
            
            # For each environment, pick the image with this segment_index
            for env in sorted(env_segment_images.keys()):
                segment_images = env_segment_images[env]
                
                # Try to get image with current segment_index
                if test_set_idx in segment_images and segment_images[test_set_idx]:
                    # Prioritize obverse angle
                    candidates = segment_images[test_set_idx]
                    obverse_imgs = [img for img in candidates if img.get('angle', '').lower() == 'obverse']
                    if obverse_imgs:
                        test_set.append(obverse_imgs[0])
                    else:
                        test_set.append(candidates[0])
                else:
                    # Fallback: use any available image from this environment
                    for segment_idx in sorted(segment_images.keys()):
                        if segment_images[segment_idx]:
                            test_set.append(segment_images[segment_idx][0])
                            break
            
            # Only add test set if it has images from all environments
            if test_set and len(test_set) == len(env_segment_images):
                test_sets.append(test_set)
        
        return test_sets


def create_segment_groups(
    strain_images: List[Dict[str, Any]],
    group_size: int = 3
) -> List[List[Dict[str, Any]]]:
    """
    Group strain images into fixed-size groups for independent evaluation.
    
    Args:
        strain_images: List of image metadata dictionaries
        group_size: Number of segments per group (default: 3)
        
    Returns:
        List of image groups, where each group contains group_size images
    """
    # Sort by parent_id and segment_index for consistency
    sorted_images = sorted(
        strain_images, 
        key=lambda x: (x.get('parent_id', ''), x.get('segment_index', 0))
    )
    
    # Create groups
    groups = []
    for i in range(0, len(sorted_images), group_size):
        group = sorted_images[i:i+group_size]
        if len(group) == group_size:  # Only include complete groups
            groups.append(group)
    
    return groups


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
    strain_to_specy_path: str = "../Dataset/strain_to_specy.csv",
) -> Dict[str, Any]:
    """
    Predict species for a specific segment group using k-nearest neighbors.
    Similar to predict() but operates on a specific group of segments.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        test_group: List of image metadata dictionaries for this test group
        strain: Strain name
        feature_extractor: Feature extractor instance
        k: Number of nearest neighbors to consider
        min_samples: Minimum number of samples required for a species
        without_siblings: Whether to exclude images with same parent_id
        environment: Environment filter
        strategy: Aggregation strategy - "avg" or "uni"
        strain_to_specy_path: Path to strain-to-species mapping CSV
        
    Returns:
        Dictionary containing prediction results for this group
    """
    from prediction import aggregate_predictions, filter_siblings
    from query_utils import find_nearest_neighbors_by_id
    
    # Load strain to species mapping
    strain_to_specy = load_strain_to_species_mapping(strain_to_specy_path)
    ground_truth_specy = strain_to_specy.get(strain, 'unknown')
    
    # Query for each image in the test group
    all_neighbors = []
    raw_results_per_image = []
    
    for image_data in test_group:
        image_id = image_data['image_id']
        parent_id = image_data.get('parent_id', 'unknown')
        img_environment = image_data.get('environment', 'unknown')
        
        # Determine environment filter
        if environment is None:
            search_environment = img_environment
        elif environment.lower() == "all":
            search_environment = None
        else:
            search_environment = environment
        
        try:
            # Find nearest neighbors
            neighbors = find_nearest_neighbors_by_id(
                client=client,
                collection_name=collection_name,
                query_image_id=image_id,
                feature_type=feature_extractor.name.lower(),
                num_neighbors=k * 2 if without_siblings else k,
                environment=search_environment,
                exclude_self=True
            )
            
            # Filter siblings if requested
            if without_siblings:
                neighbors = filter_siblings(neighbors, parent_id)
            
            # Take top k after filtering
            neighbors = neighbors[:k]
            all_neighbors.extend(neighbors)
            raw_results_per_image.append({
                'query_image_id': image_id,  # Match the expected key name
                'query_environment': img_environment,  # Add environment info
                'neighbors': neighbors
            })
        except Exception as e:
            print(f"    Error querying image {image_id}: {e}")
            continue
    
    # Aggregate predictions
    aggregated_results = aggregate_predictions(
        all_neighbors, strain_to_specy, k, min_samples, strategy
    )
    
    # Get predicted species
    predicted_specy = aggregated_results[0][0] if aggregated_results else 'unknown'
    predicted_confidence = aggregated_results[0][1] if aggregated_results else 0.0
    
    # Prepare result dictionary
    result = {
        'strain': strain,
        'ground_truth': ground_truth_specy,
        'predicted_specy': predicted_specy,
        'predicted_confidence': predicted_confidence,
        'correct': predicted_specy == ground_truth_specy,
        'num_query_images': len(test_group),
        'num_neighbors_total': len(all_neighbors),
        'raw_results': raw_results_per_image,
        'aggregated_results': [{'specy': s, 'score': sc} for s, sc in aggregated_results],
        'feature_extractor': feature_extractor.name,
        'k': k,
        'min_samples': min_samples,
        'without_siblings': without_siblings,
        'environment': environment,
        'strategy': strategy,
        'timestamp': datetime.now().isoformat()
    }
    
    return result


def run_species_evaluation(
    client: QdrantClient,
    collection_name: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    min_samples: int = None,
    without_siblings: bool = True,
    environment: str = None,
    strategy: str = "avg",
    segment_group_size: int = 3,
    output_dir: str = "./results"
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Run species-level evaluation by selecting one strain per species.
    
    Algorithm:
    1. Select one strain per species using select_one_strain_per_species()
    2. For each selected strain:
       a. Collect 6 test sets using collect_testset()
       b. For each test set:
          - Run prediction using predict_segment_group()
          - Each test set produces ONE prediction
    3. Aggregate and report results
    
    Creates 6 test sets per strain, where each test set contains one image per environment.
    
    Evaluation Strategies:
    - E1 (environment=None): Create 6 test sets, each with one image from each environment.
      Evaluate each test set WITH environment filter (query searches same env).
      Result: 6 predictions per strain.
      
    - E2 (environment="all"): Create 6 test sets, each with one image from each environment.
      Evaluate each test set WITHOUT environment filter (query searches all envs).
      Result: 6 predictions per strain.
      
    - E3 (environment=specific, e.g., "PDA"): Create 6 test sets from specific environment.
      Each test set has 1 image. Evaluate WITH environment filter.
      Result: 6 predictions per strain.
    
    Example: For 8 test strains:
      - E1: 8 strains × 6 test sets = 48 predictions
      - E2: 8 strains × 6 test sets = 48 predictions
      - E3: 8 strains × 6 test sets = 48 predictions
    
    Args:
        client: Qdrant client instance
        collection_name: Name of Qdrant collection
        feature_extractor: Feature extractor to use
        k: Number of nearest neighbors
        min_samples: Minimum number of samples (M) required for a species to be predicted
        without_siblings: Whether to exclude siblings
        environment: Environment strategy:
                    - None: E1 strategy (6 test sets with one img per env each)
                    - "all": E2 strategy (6 test sets with one img per env each)
                    - specific name: E3 strategy (6 test sets with 1 img each from that env)
        strategy: Aggregation strategy - "avg" or "uni"
        segment_group_size: Deprecated parameter (kept for compatibility)
        output_dir: Output directory for results
        
    Returns:
        Tuple of (results list, report file paths list)
    """
    print("="*80)
    print("SPECIES-LEVEL EVALUATION (6 TEST SETS PER STRAIN)")
    print("="*80)
    print(f"\nFeature Extractor: {feature_extractor.name}")
    print(f"K: {k}")
    print(f"Min Samples: {min_samples}")
    print(f"Without Siblings: {without_siblings}")
    
    # Determine evaluation strategy
    if environment is None:
        eval_strategy = "E1"
        env_strategy_code = "E1"
        env_description = "6 test sets, each with one image per env, query same env"
    elif environment.lower() == "all":
        eval_strategy = "E2"
        env_strategy_code = "E2"
        env_description = "6 test sets, each with one image per env, query all envs"
    else:
        eval_strategy = "E3"
        env_strategy_code = f"E3_{environment}"
        env_description = f"6 test sets, each with 1 image from {environment}"
    
    print(f"Evaluation Strategy: {eval_strategy} ({env_description})")
    print(f"Aggregation Strategy: {strategy}")
    print(f"Note: Each test set is evaluated to produce ONE prediction")
    print("="*80 + "\n")
    
    # Load mappings
    strain_to_specy = load_strain_to_species_mapping()
    available_strains = get_available_strains(client, collection_name)
    
    # STEP 1: Select one strain per species
    print("Step 1: Selecting one strain per species...")
    selected = select_one_strain_per_species(available_strains, strain_to_specy)
    print(f"Selected {len(selected)} strains (one per species)\n")
    
    # Print and save selection report
    selection_report_path = print_selection_report(selected, output_dir)
    
    # STEP 2: Run predictions (6 test sets per strain)
    # For each strain:
    #   - Collect 6 test sets (each with one image per environment for E1/E2, or one image from specific env for E3)
    #   - Evaluate each test set independently
    #   - Each test set produces ONE prediction
    print("\nStep 2: Running predictions (6 test sets per strain)...\n")
    results = []
    total_test_sets = 0
    
    for idx, (species, strain) in enumerate(sorted(selected.items()), 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(selected)}: {species}")
        print(f"Selected strain: {strain}")
        print(f"{'='*80}\n")
        
        try:
            # Collect test sets for this strain
            test_sets = collect_testset(
                client=client,
                collection_name=collection_name,
                strain=strain,
                environment_strategy=env_strategy_code
            )
            
            if not test_sets:
                print(f"No test sets created for strain {strain}")
                continue
            
            print(f"Created {len(test_sets)} test sets")
            if test_sets:
                print(f"Each test set has {len(test_sets[0])} images")
            print()
            
            # Evaluate EACH test set
            for test_set_idx, test_set in enumerate(test_sets, 1):
                total_test_sets += 1
                
                # Show what's in this test set
                if len(test_set) == 1:
                    img = test_set[0]
                    print(f"  Test set {test_set_idx}/{len(test_sets)}: "
                          f"1 image ({img.get('image_id')}, env={img.get('environment')}, "
                          f"seg={img.get('segment_index')})...")
                else:
                    envs = sorted(set(img.get('environment', 'unknown') for img in test_set))
                    seg_idx = test_set[0].get('segment_index', '?')
                    print(f"  Test set {test_set_idx}/{len(test_sets)}: "
                          f"{len(test_set)} images (segment_index={seg_idx}, "
                          f"envs={', '.join(envs)})...")
                
                # Make ONE prediction using this test set
                result = predict_segment_group(
                    client=client,
                    collection_name=collection_name,
                    test_group=test_set,
                    strain=strain,
                    feature_extractor=feature_extractor,
                    k=k,
                    min_samples=min_samples,
                    without_siblings=without_siblings,
                    environment=environment,
                    strategy=strategy
                )
                
                # Add metadata about this test set
                result['test_set_index'] = test_set_idx
                result['test_set_size'] = len(test_set)
                result['evaluation_strategy'] = eval_strategy
                
                results.append(result)
                
                # Print result
                correct_mark = "✓" if result['correct'] else "✗"
                print(f"    {correct_mark} Predicted: {result['predicted_specy']} "
                      f"(conf: {result['predicted_confidence']:.4f})")
                
        except Exception as e:
            print(f"Error processing strain {strain}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Total strains evaluated: {len(selected)}")
    print(f"Total test sets evaluated: {total_test_sets}")
    print(f"Total predictions: {len(results)}")
    print(f"Average predictions per strain: {len(results)/len(selected):.1f}" if selected else "N/A")
    print(f"{'='*80}")
    
    # Generate reports
    report_paths = [selection_report_path]
    
    if results:
        print("\n" + "="*80)
        print("Step 3: Generating reports...")
        print("="*80 + "\n")
        
        # Detailed prediction results
        results_report_path = print_prediction_results(results, output_dir)
        report_paths.append(results_report_path)
        
        # Confusion matrix
        print("\nGenerating confusion matrix...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_str = environment if environment else "same"
        sibling_str = "no_siblings" if without_siblings else "with_siblings"
        min_samples_str = f"_m{min_samples}" if min_samples is not None else ""
        cm_path = os.path.join(
            output_dir,
            f"species_evaluation_{eval_strategy}_{feature_extractor.name.lower()}_k{k}{min_samples_str}_{env_str}_{sibling_str}_{strategy}_{timestamp}.png"
        )
        draw_confusion_matrix(results, output_path=cm_path)
        report_paths.append(cm_path)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(report_paths)} report files:")
    for path in report_paths:
        print(f"  - {path}")
    print("\n")
    
    return results, report_paths


def compare_extractors(
    client: QdrantClient,
    collection_name: str,
    k: int = 5,
    min_samples: int = None,
    without_siblings: bool = True,
    environment: str = None,
    output_dir: str = "./results"
) -> None:
    """
    Run species evaluation for all feature extractors and compare results.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of Qdrant collection
        k: Number of nearest neighbors
        min_samples: Minimum number of samples (M) required for a species to be predicted
        without_siblings: Whether to exclude siblings
        environment: Environment filter
        output_dir: Output directory for results
    """
    extractors = [
        # ResNet50Extractor(),
        # HOGExtractor(),
        # GaborExtractor(),
        ColorHistogramExtractor(),
        ColorHistogramHSExtractor(),
        # MobileNetV2Extractor(),
        # EfficientNetV2B0Extractor()
    ]
    
    print("\n" + "="*80)
    print("COMPARING ALL FEATURE EXTRACTORS")
    print("="*80)
    print(f"\nTesting {len(extractors)} extractors: {', '.join(e.name for e in extractors)}")
    print(f"K: {k}, Min Samples: {min_samples}, Without Siblings: {without_siblings}")
    print("="*80 + "\n")
    
    all_results = {}
    
    for extractor in extractors:
        print(f"\n{'#'*80}")
        print(f"# Testing: {extractor.name}")
        print(f"{'#'*80}\n")
        
        results, _ = run_species_evaluation(
            client=client,
            collection_name=collection_name,
            feature_extractor=extractor,
            k=k,
            min_samples=min_samples,
            without_siblings=without_siblings,
            environment=environment,
            output_dir=output_dir
        )
        
        all_results[extractor.name] = results
    
    # Generate comparison report
    print("\n" + "="*80)
    print("EXTRACTOR COMPARISON REPORT")
    print("="*80 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(output_dir, f"extractor_comparison_{timestamp}.txt")
    
    comparison_lines = []
    comparison_lines.append("="*80)
    comparison_lines.append("FEATURE EXTRACTOR COMPARISON")
    comparison_lines.append("="*80)
    comparison_lines.append(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    comparison_lines.append(f"K: {k}")
    comparison_lines.append(f"Min Samples: {min_samples}")
    comparison_lines.append(f"Without Siblings: {without_siblings}")
    comparison_lines.append(f"Environment: {environment if environment else 'same as query'}")
    comparison_lines.append("\n" + "-"*80)
    comparison_lines.append(f"{'Extractor':<25} {'Accuracy':<15} {'Correct/Total':<20} {'Avg Confidence'}")
    comparison_lines.append("-"*80)
    
    for extractor_name, results in all_results.items():
        if results:
            correct = sum(1 for r in results if r.get('correct', False))
            total = len(results)
            accuracy = correct / total * 100 if total > 0 else 0
            avg_conf = sum(r['predicted_confidence'] for r in results) / total
            
            comparison_lines.append(
                f"{extractor_name:<25} {accuracy:>6.2f}%{'':<8} {correct}/{total:<16} {avg_conf:.4f}"
            )
    
    comparison_lines.append("-"*80)
    comparison_lines.append("\n" + "="*80)
    
    comparison_text = "\n".join(comparison_lines)
    print(comparison_text)
    
    with open(comparison_path, 'w') as f:
        f.write(comparison_text)
    
    print(f"\nComparison report saved to: {comparison_path}\n")


def run_comprehensive_evaluation(
    client: QdrantClient,
    collection_name: str,
    feature_extractors: List[FeatureExtractor] = None,
    k: int = 3,
    min_samples: int = None,
    without_siblings: bool = True,
    test_environments: List[str] = None,
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation testing all combinations of feature extractors, 
    environment and aggregation strategies.
    
    Feature Extractors:
    - Tests all provided extractors (ResNet50, HOG, Gabor, ColorHistogram, etc.)
    
    Environment strategies:
    - E1 (environment=None): One image per environment, query searches same environment
    - E2 (environment="all"): One image per environment, query searches all environments
    - E3 (environment=specific): One image from specific environment only
    
    Aggregation strategies:
    - S1: "avg" - weighted by similarity score
    - S2: "uni" - uniform voting (each match counts as 1)
    
    Note: Each combination evaluates ONE image per environment independently.
    For example, with 10 species, if each strain has images in 7 environments:
    - E1/E2: 10 strains × 7 images (one per env) = 70 predictions
    - E3 (one env): 10 strains × 1 image = 10 predictions
    
    Args:
        client: Qdrant client instance
        collection_name: Name of Qdrant collection
        feature_extractors: List of feature extractors to test (if None, uses all available)
        k: Number of nearest neighbors (default: 3)
        min_samples: Minimum number of samples required
        without_siblings: Whether to exclude siblings
        test_environments: List of specific environment names to test (for E3)
        output_dir: Output directory for results
        
    Returns:
        Dictionary containing all results and summary table
    """
    # Default feature extractors if none provided
    if feature_extractors is None:
        feature_extractors = [
            ResNet50Extractor(),
            HOGExtractor(),
            GaborExtractor(),
            ColorHistogramExtractor(),
            ColorHistogramHSExtractor(),
            MobileNetV2Extractor(),
            EfficientNetV2B0Extractor()
        ]
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("Testing all combinations of feature extractors, environment and aggregation strategies")
    print("="*80)
    print(f"\nFeature Extractors: {[fe.name for fe in feature_extractors]}")
    print(f"K: {k}")
    print(f"Min Samples: {min_samples}")
    print(f"Without Siblings: {without_siblings}")
    print("="*80 + "\n")
    
    # Define environment strategies
    env_strategies = [
        ("E1", None, "same as query"),
        ("E2", "all", "all environments"),
    ]
    
    # Add specific environments if provided
    if test_environments:
        for env in test_environments:
            env_strategies.append((f"E3_{env}", env, env))
    
    # Define aggregation strategies
    agg_strategies = [
        ("S1", "avg", "weighted by score"),
        ("S2", "uni", "uniform voting"),
    ]
    
    # Store all results
    all_results = {}
    summary_data = []
    
    # Run evaluation for each combination (Feature Extractor x Environment x Aggregation)
    total_combinations = len(feature_extractors) * len(env_strategies) * len(agg_strategies)
    current = 0
    
    for feature_extractor in feature_extractors:
        for env_code, env_value, env_desc in env_strategies:
            for agg_code, agg_value, agg_desc in agg_strategies:
                current += 1
                combo_name = f"{feature_extractor.name}_{env_code}_{agg_code}"
                
                print("\n" + "#"*80)
                print(f"# Combination {current}/{total_combinations}: {combo_name}")
                print(f"# Feature Extractor: {feature_extractor.name}")
                print(f"# Environment: {env_desc}")
                print(f"# Aggregation: {agg_desc}")
                print("#"*80 + "\n")
                
                try:
                    results, _ = run_species_evaluation(
                        client=client,
                        collection_name=collection_name,
                        feature_extractor=feature_extractor,
                        k=k,
                        min_samples=min_samples,
                        without_siblings=without_siblings,
                        environment=env_value,
                        strategy=agg_value,
                        output_dir=os.path.join(output_dir, combo_name)
                    )
                    
                    # Calculate accuracy
                    correct = sum(1 for r in results if r.get('correct', False))
                    total = len(results)
                    
                    # Warn if no results
                    if total == 0:
                        print(f"\n>>> WARNING: No results for this combination (no images found)\n")
                    
                    accuracy = (correct / total * 100) if total > 0 else 0.0
                    
                    all_results[combo_name] = {
                        'feature_extractor': feature_extractor.name,
                        'env_strategy': env_code,
                        'env_value': env_value,
                        'env_desc': env_desc,
                        'agg_strategy': agg_code,
                        'agg_value': agg_value,
                        'agg_desc': agg_desc,
                        'results': results,
                        'correct': correct,
                        'total': total,
                        'accuracy': accuracy
                    }
                    
                    # Always add to summary, even if no results
                    summary_data.append({
                        'Feature Extractor': feature_extractor.name,
                        'Environment Strategy': env_code,
                        'Aggregation Strategy': agg_code,
                        'Accuracy': f"{accuracy:.2f}%" if total > 0 else "N/A",
                        'Correct/Total': f"{correct}/{total}" if total > 0 else "0/0 (no data)"
                    })
                    
                    if total > 0:
                        print(f"\n>>> Results: {correct}/{total} correct ({accuracy:.2f}%)\n")
                    else:
                        print(f"\n>>> Skipped: No data available for this combination\n")
                
                except Exception as e:
                    print(f"Error in combination {combo_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Generate summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Print table header with Feature Extractor column
    print(f"{'Feature Extractor':<25} {'Environment':<15} {'Aggregation':<15} {'Accuracy':<12} {'Correct/Total'}")
    print("-"*100)
    
    # Print table rows
    for row in summary_data:
        print(f"{row['Feature Extractor']:<25} {row['Environment Strategy']:<15} "
              f"{row['Aggregation Strategy']:<15} {row['Accuracy']:<12} {row['Correct/Total']}")
    
    print("-"*100)
    print("\n" + "="*80 + "\n")
    
    # Warn if no data collected
    if not summary_data:
        print("⚠️  WARNING: No evaluation data was collected!")
        print("   This usually means all combinations failed or found no images.")
        print("   Check the errors above for details.\n")
    
    # Save summary table as CSV
    import csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"comprehensive_evaluation_summary_{timestamp}.csv")
    
    with open(summary_path, 'w', newline='') as f:
        # Write metadata as comments
        f.write(f"# Comprehensive Evaluation Results\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Feature Extractors: {', '.join([fe.name for fe in feature_extractors])}\n")
        f.write(f"# K: {k}\n")
        f.write(f"# Min Samples: {min_samples}\n")
        f.write(f"# Without Siblings: {without_siblings}\n")
        f.write("#\n")
        
        # Write CSV data
        fieldnames = ['Feature Extractor', 'Environment Strategy', 'Aggregation Strategy', 'Accuracy', 'Correct/Total']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    
    print(f"Summary table saved to: {summary_path}\n")
    
    return {
        'all_results': all_results,
        'summary_data': summary_data,
        'summary_path': summary_path
    }


# Main execution
if __name__ == "__main__":
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features_all"
    
    # Configuration
    K = 7
    MIN_SAMPLES = 0 # Set to an integer to enable minimum sample constraint
    WITHOUT_SIBLINGS = True
    ENVIRONMENT = None # None for same as query, "all" for no filter
    
    # Generate folder name based on parameters
    env_str = ENVIRONMENT if ENVIRONMENT else "same"
    sibling_str = "no_siblings" if WITHOUT_SIBLINGS else "with_siblings"
    OUTPUT_DIR = f"./results/k{K}_{env_str}"
    
    print("\n" + "="*80)
    print("MYCO FUNGI SPECIES EVALUATION SYSTEM")
    print("="*80)
    print("\nConfiguration:")
    print(f"  K: {K}")
    print(f"  Min Samples: {MIN_SAMPLES}")
    print(f"  Without siblings: {WITHOUT_SIBLINGS}")
    print(f"  Environment: {ENVIRONMENT if ENVIRONMENT else 'same as query'}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Option 1: Run single evaluation with segment-based sampling
    print("Option 1: Running segment-based evaluation with ResNet50...")
    results, report_paths = run_species_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractor=ResNet50Extractor(),
        k=K,
        min_samples=MIN_SAMPLES,
        without_siblings=WITHOUT_SIBLINGS,
        environment=ENVIRONMENT,
        strategy="avg",
        output_dir=OUTPUT_DIR
    )
    
    # Option 2: Run comprehensive evaluation (tests all combinations)
    # Uncomment to run:
    """
    print("\nOption 2: Running comprehensive evaluation...")
    comprehensive_results = run_comprehensive_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractors=[
            ResNet50Extractor(),
            ColorHistogramExtractor(),
            ColorHistogramHSExtractor()
        ],  # List of feature extractors to test
        k=3,  # Fixed K=3 as specified
        min_samples=MIN_SAMPLES,
        without_siblings=WITHOUT_SIBLINGS,
        test_environments=["PDA", "MEA"],  # Add your specific environments for E3
        output_dir="./results/comprehensive"
    )
    """
    
    # Option 3: Compare all extractors (uncomment to run)
    """
    print("\nOption 3: Comparing all feature extractors...")
    compare_extractors(
        client=client,
        collection_name=collection_name,
        k=K,
        min_samples=MIN_SAMPLES,
        without_siblings=WITHOUT_SIBLINGS,
        environment=ENVIRONMENT,
        output_dir=OUTPUT_DIR
    )
    """
