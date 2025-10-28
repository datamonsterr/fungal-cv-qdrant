"""
Species-level evaluation script.
Takes one strain per species, runs predictions, and generates comprehensive reports.
"""
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from qdrant_client import QdrantClient

from prediction import (
    predict,
    draw_confusion_matrix,
    load_strain_to_species_mapping,
    get_available_strains
)
from feature_extractors import (
    ResNet50Extractor,
    HOGExtractor,
    GaborExtractor,
    ColorHistogramExtractor,
    FeatureExtractor
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
        selected[species] = strains[0]  # Take first strain
    
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


def run_species_evaluation(
    client: QdrantClient,
    collection_name: str,
    feature_extractor: FeatureExtractor,
    k: int = 5,
    without_siblings: bool = True,
    environment: str = None,
    output_dir: str = "./results"
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Run species-level evaluation by selecting one strain per species.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of Qdrant collection
        feature_extractor: Feature extractor to use
        k: Number of nearest neighbors
        without_siblings: Whether to exclude siblings
        environment: Environment filter
        output_dir: Output directory for results
        
    Returns:
        Tuple of (results list, report file paths list)
    """
    print("="*80)
    print("SPECIES-LEVEL EVALUATION")
    print("="*80)
    print(f"\nFeature Extractor: {feature_extractor.name}")
    print(f"K: {k}")
    print(f"Without Siblings: {without_siblings}")
    print(f"Environment: {environment if environment else 'same as query'}")
    print("="*80 + "\n")
    
    # Load mappings
    strain_to_specy = load_strain_to_species_mapping()
    available_strains = get_available_strains(client, collection_name)
    
    # Select one strain per species
    print("Step 1: Selecting one strain per species...")
    selected = select_one_strain_per_species(available_strains, strain_to_specy)
    print(f"Selected {len(selected)} strains (one per species)\n")
    
    # Print and save selection report
    selection_report_path = print_selection_report(selected, output_dir)
    
    # Run predictions
    print("\nStep 2: Running predictions...\n")
    results = []
    
    for idx, (species, strain) in enumerate(sorted(selected.items()), 1):
        print(f"\n{'='*80}")
        print(f"Processing {idx}/{len(selected)}: {species}")
        print(f"Selected strain: {strain}")
        print(f"{'='*80}\n")
        
        try:
            result = predict(
                client=client,
                collection_name=collection_name,
                strain=strain,
                feature_extractor=feature_extractor,
                k=k,
                without_siblings=without_siblings,
                environment=environment,
                output_dir=output_dir
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing strain {strain}: {e}")
            continue
    
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
        cm_path = os.path.join(
            output_dir,
            f"species_evaluation_cm_{feature_extractor.name.lower()}_k{k}_{timestamp}.png"
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
        without_siblings: Whether to exclude siblings
        environment: Environment filter
        output_dir: Output directory for results
    """
    extractors = [
        ResNet50Extractor(),
        HOGExtractor(),
        GaborExtractor(),
        ColorHistogramExtractor()
    ]
    
    print("\n" + "="*80)
    print("COMPARING ALL FEATURE EXTRACTORS")
    print("="*80)
    print(f"\nTesting {len(extractors)} extractors: {', '.join(e.name for e in extractors)}")
    print(f"K: {k}, Without Siblings: {without_siblings}")
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


# Main execution
if __name__ == "__main__":
    # Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "myco_fungi_features"
    
    # Configuration
    OUTPUT_DIR = "./results/species_evaluation"
    K = 5
    WITHOUT_SIBLINGS = True
    ENVIRONMENT = None  # None for same as query, "all" for no filter
    
    print("\n" + "="*80)
    print("MYCO FUNGI SPECIES EVALUATION SYSTEM")
    print("="*80)
    print("\nConfiguration:")
    print(f"  K: {K}")
    print(f"  Without siblings: {WITHOUT_SIBLINGS}")
    print(f"  Environment: {ENVIRONMENT if ENVIRONMENT else 'same as query'}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*80 + "\n")
    
    # Option 1: Run evaluation with single extractor
    print("Option 1: Running evaluation with ResNet50...")
    results, report_paths = run_species_evaluation(
        client=client,
        collection_name=collection_name,
        feature_extractor=ResNet50Extractor(),
        k=K,
        without_siblings=WITHOUT_SIBLINGS,
        environment=ENVIRONMENT,
        output_dir=OUTPUT_DIR
    )
    
    # Option 2: Compare all extractors (uncomment to run)
    # print("\nOption 2: Comparing all feature extractors...")
    # compare_extractors(
    #     client=client,
    #     collection_name=collection_name,
    #     k=K,
    #     without_siblings=WITHOUT_SIBLINGS,
    #     environment=ENVIRONMENT,
    #     output_dir=OUTPUT_DIR
    # )
