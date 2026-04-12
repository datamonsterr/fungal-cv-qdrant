"""
Generate a comprehensive report from ensemble analysis results.

This script creates detailed visualizations and analysis for ensemble predictions:
- Agreement matrix showing how often models agree on correct predictions
- Performance by species comparing individual models vs ensemble
- Confidence distribution across all models
- Performance trends across test sets

Usage:
    python ensemble_report.py [strategy]

Arguments:
    strategy: Optional. One of 'weighted', 'simple_avg', 'manual_weighted'
              Default: 'weighted'

Examples:
    python ensemble_report.py                    # Uses weighted strategy
    python ensemble_report.py weighted           # Explicit weighted strategy
    python ensemble_report.py simple_avg         # Simple average strategy
    python ensemble_report.py manual_weighted    # Manual weighted strategy

Output:
    - detailed_comparison_{strategy}.png: 4-chart visualization with:
        * Model agreement matrix
        * Performance by species
        * Confidence distributions
        * Performance across test sets

Note: This script works with any combination of feature extractors used in the
      ensemble analysis, not just the default 3 (ColorHistogramHS, ResNet50,
      EfficientNetV2B0).
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_ensemble_results(json_path: str):
    """Load ensemble results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def analyze_improvement_cases(results):
    """Analyze cases where ensemble improves over individual models."""

    feature_extractors = results["feature_extractors"]

    improvements = {"ensemble_better_than_all": [], "all_correct": [], "all_wrong": []}

    # Add rescue categories for each feature extractor
    for fe in feature_extractors:
        improvements[f"ensemble_rescues_{fe}"] = []

    for pred in results["ensemble_predictions"]:
        ensemble_correct = pred["correct"]

        indiv = pred["individual_predictions"]

        # Check if all models are correct or all wrong
        all_correct = all(indiv[fe]["correct"] for fe in feature_extractors)
        all_wrong = all(not indiv[fe]["correct"] for fe in feature_extractors)

        # Count different scenarios
        if all_correct:
            improvements["all_correct"].append(pred)
        elif all_wrong:
            improvements["all_wrong"].append(pred)
            if ensemble_correct:
                improvements["ensemble_better_than_all"].append(pred)
        else:
            # Mixed results - check which models the ensemble rescues
            if ensemble_correct:
                for fe in feature_extractors:
                    if not indiv[fe]["correct"]:
                        improvements[f"ensemble_rescues_{fe}"].append(pred)

    return improvements


def create_detailed_comparison_chart(results, output_path):
    """Create a detailed comparison chart showing prediction agreement."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data
    predictions = results["ensemble_predictions"]
    feature_extractors = results["feature_extractors"]

    # Chart 1: Agreement matrix
    num_models = len(feature_extractors) + 1  # +1 for ensemble
    agreement_matrix = np.zeros((num_models, num_models))

    # Shorten model names for display
    display_names = [
        fe.replace("ColorHistogramHS", "ColorHist").replace(
            "EfficientNetV2B0", "Efficient"
        )
        for fe in feature_extractors
    ] + ["Ensemble"]

    for pred in predictions:
        correct_vector = [
            pred["individual_predictions"][fe]["correct"] for fe in feature_extractors
        ] + [pred["correct"]]

        for i in range(num_models):
            for j in range(num_models):
                if correct_vector[i] and correct_vector[j]:
                    agreement_matrix[i][j] += 1

    # Normalize to percentages
    total_predictions = len(predictions)
    agreement_matrix = (agreement_matrix / total_predictions) * 100

    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=axes[0, 0],
        cbar_kws={"label": "Agreement %"},
    )
    axes[0, 0].set_title(
        "Model Agreement Matrix\n(% of cases both correct)",
        fontsize=12,
        fontweight="bold",
    )

    # Chart 2: Individual vs Ensemble performance by species
    species_stats = defaultdict(
        lambda: {"ensemble": 0, "total": 0, **{fe: 0 for fe in feature_extractors}}
    )

    for pred in predictions:
        species = pred["ground_truth"]
        species_stats[species]["total"] += 1
        for fe in feature_extractors:
            if pred["individual_predictions"][fe]["correct"]:
                species_stats[species][fe] += 1
        if pred["correct"]:
            species_stats[species]["ensemble"] += 1

    # Convert to percentages and prepare data
    species_names = []
    accuracies = {fe: [] for fe in feature_extractors}
    accuracies["Ensemble"] = []

    for species, stats in sorted(species_stats.items()):
        # Shorten species names for display
        short_name = species.replace("Penicillium ", "P. ")
        species_names.append(short_name)

        total = stats["total"]
        for fe in feature_extractors:
            accuracies[fe].append(stats[fe] / total * 100)
        accuracies["Ensemble"].append(stats["ensemble"] / total * 100)

    # Plot grouped bar chart
    x = np.arange(len(species_names))
    num_bars = len(feature_extractors) + 1
    width = 0.8 / num_bars
    colors = [
        "#ff6b6b",
        "#4ecdc4",
        "#45b7d1",
        "#f38181",
        "#aa96da",
        "#fcbad3",
        "#ffffd2",
    ]

    for i, fe in enumerate(feature_extractors):
        offset = (i - num_bars / 2 + 0.5) * width
        short_name = fe.replace("ColorHistogramHS", "ColorHist").replace(
            "EfficientNetV2B0", "Efficient"
        )
        axes[0, 1].bar(
            x + offset,
            accuracies[fe],
            width,
            label=short_name,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    # Ensemble bar (always last)
    offset = (num_bars - num_bars / 2 - 0.5) * width
    axes[0, 1].bar(
        x + offset,
        accuracies["Ensemble"],
        width,
        label="Ensemble",
        color="#95e1d3",
        alpha=0.8,
        edgecolor="black",
        linewidth=2,
    )

    axes[0, 1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 1].set_title("Performance by Species", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(species_names, rotation=45, ha="right", fontsize=9)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].set_ylim(0, 110)

    # Chart 3: Confidence distribution
    confidences = {}
    for fe in feature_extractors:
        confidences[fe] = [
            p["individual_predictions"][fe]["confidence"] for p in predictions
        ]
    confidences["Ensemble"] = [p["predicted_confidence"] for p in predictions]

    # Prepare data for boxplot
    confidence_data = [confidences[fe] for fe in feature_extractors] + [
        confidences["Ensemble"]
    ]
    labels = [
        fe.replace("ColorHistogramHS", "ColorHist").replace(
            "EfficientNetV2B0", "Efficient"
        )
        for fe in feature_extractors
    ] + ["Ensemble"]

    axes[1, 0].boxplot(
        confidence_data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )
    axes[1, 0].set_ylabel("Confidence Score", fontsize=11)
    axes[1, 0].set_title(
        "Prediction Confidence Distribution", fontsize=12, fontweight="bold"
    )
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].tick_params(axis="x", rotation=45)
    for label in axes[1, 0].get_xticklabels():
        label.set_ha("right")

    # Chart 4: Cumulative accuracy by test set
    test_set_indices = sorted(set(p["test_set_index"] for p in predictions))

    cumulative_correct = {fe: [] for fe in feature_extractors}
    cumulative_correct["Ensemble"] = []

    for test_idx in test_set_indices:
        test_predictions = [p for p in predictions if p["test_set_index"] == test_idx]

        for fe in feature_extractors:
            cumulative_correct[fe].append(
                sum(
                    1
                    for p in test_predictions
                    if p["individual_predictions"][fe]["correct"]
                )
            )
        cumulative_correct["Ensemble"].append(
            sum(1 for p in test_predictions if p["correct"])
        )

    # Plot with different markers and colors
    markers = ["o", "s", "^", "v", "D", "p", "*"]

    for i, fe in enumerate(feature_extractors):
        short_name = fe.replace("ColorHistogramHS", "ColorHist").replace(
            "EfficientNetV2B0", "Efficient"
        )
        axes[1, 1].plot(
            test_set_indices,
            cumulative_correct[fe],
            marker=markers[i % len(markers)],
            label=short_name,
            linewidth=2,
            color=colors[i % len(colors)],
        )

    # Ensemble line (always last, prominent)
    axes[1, 1].plot(
        test_set_indices,
        cumulative_correct["Ensemble"],
        marker="D",
        label="Ensemble",
        linewidth=3,
        color="#95e1d3",
        linestyle="--",
        markersize=8,
    )

    axes[1, 1].set_xlabel("Test Set Index", fontsize=11)
    axes[1, 1].set_ylabel("Correct Predictions", fontsize=11)
    axes[1, 1].set_title("Performance Across Test Sets", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved detailed comparison chart to: {output_path}")
    plt.close()


def print_detailed_report(results):
    """Print detailed text report."""

    print("\n" + "=" * 80)
    print("DETAILED ENSEMBLE ANALYSIS REPORT")
    print("=" * 80)

    feature_extractors = results["feature_extractors"]
    improvements = analyze_improvement_cases(results)

    print("\n" + "-" * 80)
    print("PERFORMANCE SUMMARY")
    print("-" * 80)

    total = results["total_predictions"]

    # Calculate individual accuracies dynamically
    print()
    max_name_len = max(len(fe) for fe in feature_extractors)
    for fe in feature_extractors:
        correct = sum(
            1
            for p in results["ensemble_predictions"]
            if p["individual_predictions"][fe]["correct"]
        )
        print(f"{fe:<{max_name_len}}: {correct}/{total} ({correct / total * 100:.2f}%)")

    ensemble_correct = results["correct_count"]
    print(
        f"{'Ensemble':<{max_name_len}}: {ensemble_correct}/{total} ({ensemble_correct / total * 100:.2f}%)"
    )

    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 80)

    print(
        f"\nCases where all models correct:        {len(improvements['all_correct'])}"
    )
    print(f"Cases where all models wrong:           {len(improvements['all_wrong'])}")
    print(
        f"Cases where ensemble beats all:         {len(improvements['ensemble_better_than_all'])}"
    )

    print("\nCases where ensemble rescues:")
    for fe in feature_extractors:
        count = len(improvements[f"ensemble_rescues_{fe}"])
        print(f"  - {fe} wrong: {count:>5}")

    # Show examples of ensemble rescues
    if improvements["ensemble_better_than_all"]:
        print("\n" + "-" * 80)
        print("EXAMPLES: Ensemble Correct When All Models Wrong")
        print("-" * 80)
        for i, pred in enumerate(improvements["ensemble_better_than_all"][:3], 1):
            print(f"\n{i}. {pred['strain']} (Test Set {pred['test_set_index']})")
            print(f"   Ground Truth: {pred['ground_truth']}")
            for fe in feature_extractors:
                predicted = pred["individual_predictions"][fe]["predicted"]
                print(f"   {fe} predicted: {predicted}")
            print(f"   Ensemble predicted: {pred['predicted_specy']} ✓")
            print(f"   Ensemble confidence: {pred['predicted_confidence']:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main function."""

    import sys

    from src.config import RESULTS_DIR

    results_dir = RESULTS_DIR / "ensemble_analysis"

    # Allow specifying strategy as argument, default to weighted
    strategy = sys.argv[1] if len(sys.argv) > 1 else "weighted"
    JSON_PATH = str(results_dir / f"ensemble_results_{strategy}.json")

    print(f"Loading ensemble results for strategy: {strategy}")
    results = load_ensemble_results(JSON_PATH)

    # Print detailed report
    print_detailed_report(results)

    # Create detailed comparison chart
    print("\nGenerating detailed comparison charts...")
    output_path = str(results_dir / f"detailed_comparison_{strategy}.png")
    create_detailed_comparison_chart(results, output_path)

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files in {results_dir}:")
    print(f"  - detailed_comparison_{strategy}.png")
    print(f"  - ensemble_results_{strategy}.json")
    print("\nNote: Other visualizations are generated by ensemble_analysis.py:")
    print("  - correction_analysis.png")
    print("  - ground_truth_rankings.png")
    print("  - ensemble_strategy_comparison.png")


if __name__ == "__main__":
    main()
