"""
Ensemble analysis script for combining multiple feature extractors.
Analyzes results from comprehensive evaluation and creates weighted ensemble.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pydantic import BaseModel

from src.config import RESULTS_DIR, SPECIES_WEIGHTS_PATH

# ========== Pydantic Models ==========


class AggregatedResult(BaseModel):
    """Individual aggregated result with species and score."""

    specy: str
    score: float


class PredictionResult(BaseModel):
    """Individual prediction result from evaluation."""

    strain: str
    ground_truth: str
    predicted_specy: str
    predicted_confidence: float
    correct: bool
    num_query_images: int
    num_neighbors_total: int
    aggregated_results: List[AggregatedResult]
    feature_extractor: str
    k: int
    min_samples: Optional[int]
    without_siblings: bool
    environment: Optional[str]
    strategy: str
    timestamp: str
    test_set_index: int
    test_set_size: int
    evaluation_strategy: str
    species: str


class ResultMetadata(BaseModel):
    """Metadata for evaluation results."""

    timestamp: str
    feature_extractor: str
    k: int
    min_samples: Optional[int]
    without_siblings: bool
    environment: Optional[str]
    eval_strategy: str
    aggregation_strategy: str
    total_strains: int
    total_predictions: int


class EvaluationResults(BaseModel):
    """Complete evaluation results file."""

    metadata: ResultMetadata
    predictions: List[PredictionResult]


class AccuracyData(BaseModel):
    """Accuracy data from CSV."""

    feature_extractor: str
    environment_strategy: str
    aggregation_strategy: str
    accuracy: float
    correct: int
    total: int


class SpeciesWeights(BaseModel):
    """Manual species-specific weights for feature extractors."""

    description: str
    weights: Dict[str, Dict[str, float]]
    notes: List[str]


# ========== Data Loading Functions ==========


def load_results_json(file_path: str) -> EvaluationResults:
    """Load and parse results JSON file."""
    print(f"    Reading file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    results = EvaluationResults(**data)
    print(f"    Loaded {len(results.predictions)} predictions")
    print(
        f"    Metadata: {results.metadata.feature_extractor}, {results.metadata.eval_strategy}, {results.metadata.aggregation_strategy}"
    )
    return results


def load_species_weights(json_path: str) -> SpeciesWeights:
    """Load species-specific manual weights from JSON file."""
    print(f"    Reading species weights: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    weights = SpeciesWeights(**data)
    print(f"    Loaded weights for {len(weights.weights) - 1} species (+ default)")
    return weights


def load_accuracy_from_csv(csv_path: str) -> Dict[Tuple[str, str, str], AccuracyData]:
    """
    Load accuracy data from comprehensive CSV.

    Returns:
        Dictionary mapping (feature_extractor, env_strategy, agg_strategy) to AccuracyData
    """
    accuracy_dict: Dict[Tuple[str, str, str], AccuracyData] = {}

    with open(csv_path, "r") as f:
        # Skip comment lines starting with #
        lines = [line for line in f if not line.strip().startswith("#")]

    # Parse as CSV from filtered lines
    reader = csv.DictReader(lines)
    for row in reader:
        if not row.get("Feature Extractor"):  # Skip empty lines
            continue

        # Parse accuracy percentage
        accuracy_str = row["Accuracy"].rstrip("%")
        accuracy = float(accuracy_str) / 100.0

        # Parse correct/total
        correct_total = row["Correct/Total"]
        correct, total = map(int, correct_total.split("/"))

        data = AccuracyData(
            feature_extractor=row["Feature Extractor"],
            environment_strategy=row["Environment Strategy"],
            aggregation_strategy=row["Aggregation Strategy"],
            accuracy=accuracy,
            correct=correct,
            total=total,
        )

        key = (
            data.feature_extractor,
            data.environment_strategy,
            data.aggregation_strategy,
        )
        accuracy_dict[key] = data

    return accuracy_dict


def find_result_file(
    base_dir: str, feature_extractor: str, env_strategy: str, agg_strategy: str
) -> Optional[str]:
    """
    Find the results JSON file for a specific configuration.

    Args:
        base_dir: Base results directory
        feature_extractor: Feature extractor name
        env_strategy: Environment strategy (E1, E2, E3_xxx)
        agg_strategy: Aggregation strategy (S1 for weighted, S2 for uni)

    Returns:
        Path to JSON file or None if not found
    """
    # Map S1/S2 to AVG/UNI
    agg_map = {
        "S1": "WEIGHTED",
        "S2": "UNI",
        "avg": "WEIGHTED",
        "weighted": "WEIGHTED",
        "uni": "UNI",
    }
    agg_folder = agg_map.get(agg_strategy, agg_strategy.upper())

    folder_name = f"{feature_extractor}_{env_strategy}_{agg_folder}"
    folder_path = os.path.join(base_dir, folder_name)

    print(f"  [DEBUG] Looking for folder: {folder_path}")

    if not os.path.exists(folder_path):
        print("  [DEBUG] Folder not found!")
        return None

    # Find JSON file in folder
    json_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".json") and f.startswith("results_")
    ]

    print(f"  [DEBUG] Found {len(json_files)} result JSON files")

    if not json_files:
        return None

    # Return the most recent file (sorted by name, which includes timestamp)
    json_files.sort(reverse=True)
    selected_file = os.path.join(folder_path, json_files[0])
    print(f"  [DEBUG] Selected: {json_files[0]}")
    return selected_file


# ========== Analysis Functions ==========


def analyze_false_cases(
    colorhist_results: EvaluationResults,
    resnet_results: EvaluationResults,
    efficient_results: EvaluationResults,
) -> Dict[str, Any]:
    """
    Analyze cases where ColorHistogramHS is wrong but ResNet50/EfficientNet are correct.

    Returns:
        Dictionary with analysis results
    """

    # Create lookup by (strain, test_set_index)
    def create_lookup(
        results: EvaluationResults,
    ) -> Dict[Tuple[str, int], PredictionResult]:
        return {(p.strain, p.test_set_index): p for p in results.predictions}

    colorhist_lookup = create_lookup(colorhist_results)
    resnet_lookup = create_lookup(resnet_results)
    efficient_lookup = create_lookup(efficient_results)

    # Find cases where ColorHist is wrong but others are correct
    colorhist_wrong_resnet_correct = []
    colorhist_wrong_efficient_correct = []
    colorhist_wrong_both_correct = []
    all_colorhist_wrong = []

    for key, colorhist_pred in colorhist_lookup.items():
        if colorhist_pred.correct:
            continue

        all_colorhist_wrong.append(colorhist_pred)

        resnet_pred = resnet_lookup.get(key)
        efficient_pred = efficient_lookup.get(key)

        resnet_correct = resnet_pred and resnet_pred.correct
        efficient_correct = efficient_pred and efficient_pred.correct

        if resnet_correct and efficient_correct:
            colorhist_wrong_both_correct.append(
                {
                    "colorhist": colorhist_pred,
                    "resnet": resnet_pred,
                    "efficient": efficient_pred,
                }
            )
        elif resnet_correct:
            colorhist_wrong_resnet_correct.append(
                {
                    "colorhist": colorhist_pred,
                    "resnet": resnet_pred,
                    "efficient": efficient_pred,
                }
            )
        elif efficient_correct:
            colorhist_wrong_efficient_correct.append(
                {
                    "colorhist": colorhist_pred,
                    "resnet": resnet_pred,
                    "efficient": efficient_pred,
                }
            )

    return {
        "total_colorhist_wrong": len(all_colorhist_wrong),
        "colorhist_wrong_resnet_correct": colorhist_wrong_resnet_correct,
        "colorhist_wrong_efficient_correct": colorhist_wrong_efficient_correct,
        "colorhist_wrong_both_correct": colorhist_wrong_both_correct,
        "all_colorhist_wrong": all_colorhist_wrong,
        "total_predictions": len(colorhist_lookup),
    }


def get_ground_truth_rank(
    aggregated_results: List[AggregatedResult], ground_truth: str
) -> Optional[int]:
    """
    Get the rank of ground truth species in aggregated results.

    Returns:
        Rank (1-indexed) or None if not found
    """
    for i, result in enumerate(aggregated_results, 1):
        if result.specy == ground_truth:
            return i
    return None


# ========== Visualization Functions ==========


def visualize_correction_analysis(analysis: Dict[str, Any], output_dir: str):
    """
    Visualize how many cases ColorHist is wrong but others predict correctly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Chart 1: Venn-like comparison
    categories = [
        "ColorHist Wrong\nOnly",
        "ResNet Correct\nOnly",
        "EfficientNet\nCorrect Only",
        "Both Correct",
    ]

    colorhist_only_wrong = (
        analysis["total_colorhist_wrong"]
        - len(analysis["colorhist_wrong_resnet_correct"])
        - len(analysis["colorhist_wrong_efficient_correct"])
        - len(analysis["colorhist_wrong_both_correct"])
    )

    values = [
        colorhist_only_wrong,
        len(analysis["colorhist_wrong_resnet_correct"]),
        len(analysis["colorhist_wrong_efficient_correct"]),
        len(analysis["colorhist_wrong_both_correct"]),
    ]

    colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#95e1d3"]
    bars = axes[0].bar(
        categories, values, color=colors, edgecolor="black", linewidth=1.5
    )
    axes[0].set_ylabel("Number of Cases", fontsize=12)
    axes[0].set_title(
        "Cases Where ColorHistogramHS is Wrong\nvs Other Models Correct",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_ylim(0, max(values) * 1.2)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_axisbelow(True)

    # Chart 2: Improvement potential
    total_wrong = analysis["total_colorhist_wrong"]
    correctable = (
        len(analysis["colorhist_wrong_resnet_correct"])
        + len(analysis["colorhist_wrong_efficient_correct"])
        + len(analysis["colorhist_wrong_both_correct"])
    )

    improvement_data = {
        "ColorHist Alone": analysis["total_predictions"] - total_wrong,
        "Potential with\nEnsemble": analysis["total_predictions"]
        - total_wrong
        + correctable,
    }

    bars2 = axes[1].bar(
        improvement_data.keys(),
        improvement_data.values(),
        color=["#ff6b6b", "#95e1d3"],
        edgecolor="black",
        linewidth=1.5,
    )
    axes[1].set_ylabel("Correct Predictions", fontsize=12)
    axes[1].set_title(
        "Potential Improvement with Ensemble", fontsize=14, fontweight="bold"
    )
    axes[1].set_ylim(0, analysis["total_predictions"] * 1.1)

    # Add value labels
    for bar, value in zip(bars2, improvement_data.values()):
        height = bar.get_height()
        percentage = (value / analysis["total_predictions"]) * 100
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(value)}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add horizontal line for total
    axes[1].axhline(
        y=analysis["total_predictions"],
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Total Predictions",
    )
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "correction_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved correction analysis chart to: {output_path}")
    plt.close()


def create_strategy_comparison_chart(
    weighted_result: Dict[str, Any],
    simple_result: Dict[str, Any],
    output_dir: str,
    manual_result: Optional[Dict[str, Any]] = None,
):
    """
    Create visualization comparing ensemble strategies.
    Includes weighted, simple average, and optionally manual weighted strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Overall accuracy comparison
    strategies = ["Weighted\nSum", "Simple\nAverage"]
    accuracies = [weighted_result["accuracy"] * 100, simple_result["accuracy"] * 100]
    colors_chart1 = ["#4ecdc4", "#45b7d1"]

    if manual_result:
        strategies.insert(2, "Manual\nWeighted")
        accuracies.insert(2, manual_result["accuracy"] * 100)
        colors_chart1.insert(2, "#a29bfe")

    # Calculate best individual model accuracy from predictions (not normalized weights!)
    best_individual_acc = 0.0
    best_individual_name = ""
    for fe in weighted_result["feature_extractors"]:
        individual_correct = sum(
            1
            for p in weighted_result["ensemble_predictions"]
            if p["individual_predictions"][fe]["correct"]
        )
        individual_acc = individual_correct / weighted_result["total_predictions"]
        if individual_acc > best_individual_acc:
            best_individual_acc = individual_acc
            best_individual_name = fe

    strategies.append(f"Best Individual\n({best_individual_name})")
    accuracies.append(best_individual_acc * 100)
    colors_chart1.append("#ff6b6b")

    bars = axes[0, 0].bar(
        strategies,
        accuracies,
        color=colors_chart1,
        edgecolor="black",
        linewidth=2,
        alpha=0.8,
    )
    axes[0, 0].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    axes[0, 0].set_title("Ensemble Strategy Comparison", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].set_axisbelow(True)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Chart 2: Prediction agreement (with or without manual)
    weighted_preds = weighted_result["ensemble_predictions"]
    simple_preds = simple_result["ensemble_predictions"]

    if manual_result:
        # Three-way comparison with manual weighted
        manual_preds = manual_result["ensemble_predictions"]

        # Count agreement patterns
        all_three_correct = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if wp["correct"] and sp["correct"] and mp["correct"]
        )
        manual_only = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if not wp["correct"] and not sp["correct"] and mp["correct"]
        )
        weighted_only = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if wp["correct"] and not sp["correct"] and not mp["correct"]
        )
        simple_only = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if not wp["correct"] and sp["correct"] and not mp["correct"]
        )
        two_correct = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if (wp["correct"] + sp["correct"] + mp["correct"]) == 2
        )
        all_three_wrong = sum(
            1
            for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds)
            if not wp["correct"] and not sp["correct"] and not mp["correct"]
        )

        categories = [
            "All 3\nCorrect",
            "2/3\nCorrect",
            "Manual\nOnly",
            "Weighted\nOnly",
            "Simple\nOnly",
            "All 3\nWrong",
        ]
        values = [
            all_three_correct,
            two_correct,
            manual_only,
            weighted_only,
            simple_only,
            all_three_wrong,
        ]
        colors_chart2 = [
            "#00b894",
            "#95e1d3",
            "#a29bfe",
            "#4ecdc4",
            "#45b7d1",
            "#ff6b6b",
        ]
    else:
        # Two-way comparison (original)
        both_correct = sum(
            1
            for wp, sp in zip(weighted_preds, simple_preds)
            if wp["correct"] and sp["correct"]
        )
        weighted_only = sum(
            1
            for wp, sp in zip(weighted_preds, simple_preds)
            if wp["correct"] and not sp["correct"]
        )
        simple_only = sum(
            1
            for wp, sp in zip(weighted_preds, simple_preds)
            if not wp["correct"] and sp["correct"]
        )
        both_wrong = sum(
            1
            for wp, sp in zip(weighted_preds, simple_preds)
            if not wp["correct"] and not sp["correct"]
        )

        categories = [
            "Both\nCorrect",
            "Weighted\nOnly",
            "Simple Avg\nOnly",
            "Both\nWrong",
        ]
        values = [both_correct, weighted_only, simple_only, both_wrong]
        colors_chart2 = ["#95e1d3", "#4ecdc4", "#45b7d1", "#ff6b6b"]

    bars2 = axes[0, 1].bar(
        categories,
        values,
        color=colors_chart2,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )
    axes[0, 1].set_ylabel("Number of Predictions", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Prediction Agreement", fontsize=14, fontweight="bold")
    axes[0, 1].set_ylim(0, max(values) * 1.2)
    axes[0, 1].grid(axis="y", alpha=0.3)
    axes[0, 1].set_axisbelow(True)

    for bar, val in zip(bars2, values):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(val)}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Chart 3: Confidence distribution comparison
    weighted_confidences = [p["predicted_confidence"] for p in weighted_preds]
    simple_confidences = [p["predicted_confidence"] for p in simple_preds]

    if manual_result:
        manual_preds = manual_result["ensemble_predictions"]
        manual_confidences = [p["predicted_confidence"] for p in manual_preds]

        bp = axes[1, 0].boxplot(
            [weighted_confidences, simple_confidences, manual_confidences],
            tick_labels=["Weighted", "Simple Avg", "Manual"],
            patch_artist=True,
            boxprops=dict(alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )

        # Color the boxes
        bp["boxes"][0].set_facecolor("#4ecdc4")
        bp["boxes"][1].set_facecolor("#45b7d1")
        bp["boxes"][2].set_facecolor("#a29bfe")

        # Add mean lines
        weighted_mean = np.mean(weighted_confidences)
        simple_mean = np.mean(simple_confidences)
        manual_mean = np.mean(manual_confidences)
        axes[1, 0].text(
            1,
            weighted_mean,
            f"μ={weighted_mean:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        axes[1, 0].text(
            2,
            simple_mean,
            f"μ={simple_mean:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        axes[1, 0].text(
            3,
            manual_mean,
            f"μ={manual_mean:.3f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    else:
        bp = axes[1, 0].boxplot(
            [weighted_confidences, simple_confidences],
            tick_labels=["Weighted", "Simple Average"],
            patch_artist=True,
            boxprops=dict(alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
        )

        # Color the boxes
        bp["boxes"][0].set_facecolor("#4ecdc4")
        bp["boxes"][1].set_facecolor("#45b7d1")

        # Add mean lines
        weighted_mean = np.mean(weighted_confidences)
        simple_mean = np.mean(simple_confidences)
        axes[1, 0].text(
            1,
            weighted_mean,
            f"μ={weighted_mean:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        axes[1, 0].text(
            2,
            simple_mean,
            f"μ={simple_mean:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    axes[1, 0].set_ylabel("Prediction Confidence", fontsize=12, fontweight="bold")
    axes[1, 0].set_title("Confidence Distribution", fontsize=14, fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 0].set_axisbelow(True)

    # Chart 4: Confusion analysis - where do they differ?
    if manual_result:
        manual_preds = manual_result["ensemble_predictions"]

        # Count cases where any strategy differs
        differences = []
        for wp, sp, mp in zip(weighted_preds, simple_preds, manual_preds):
            preds = [
                wp["predicted_specy"],
                sp["predicted_specy"],
                mp["predicted_specy"],
            ]
            if len(set(preds)) > 1:  # At least one differs
                differences.append(
                    {
                        "weighted_correct": wp["correct"],
                        "simple_correct": sp["correct"],
                        "manual_correct": mp["correct"],
                        "ground_truth": wp["ground_truth"],
                    }
                )

        if differences:
            # Count different outcome patterns
            all_wrong = sum(
                1
                for d in differences
                if not d["weighted_correct"]
                and not d["simple_correct"]
                and not d["manual_correct"]
            )
            weighted_best = sum(
                1
                for d in differences
                if d["weighted_correct"]
                and not d["simple_correct"]
                and not d["manual_correct"]
            )
            simple_best = sum(
                1
                for d in differences
                if not d["weighted_correct"]
                and d["simple_correct"]
                and not d["manual_correct"]
            )
            manual_best = sum(
                1
                for d in differences
                if not d["weighted_correct"]
                and not d["simple_correct"]
                and d["manual_correct"]
            )
            multiple_correct = sum(
                1
                for d in differences
                if (d["weighted_correct"] + d["simple_correct"] + d["manual_correct"])
                >= 2
            )

            diff_categories = [
                "All\nWrong",
                "Weighted\nBest",
                "Simple\nBest",
                "Manual\nBest",
                "2+\nCorrect",
            ]
            diff_values = [
                all_wrong,
                weighted_best,
                simple_best,
                manual_best,
                multiple_correct,
            ]
            colors_chart4 = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#a29bfe", "#00b894"]

            bars4 = axes[1, 1].bar(
                diff_categories,
                diff_values,
                color=colors_chart4,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )
            axes[1, 1].set_ylabel("Number of Cases", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                f"Analysis of {len(differences)} Different Predictions",
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 1].set_ylim(
                0, max(diff_values) * 1.3 if max(diff_values) > 0 else 1
            )
            axes[1, 1].grid(axis="y", alpha=0.3)
            axes[1, 1].set_axisbelow(True)

            for bar, val in zip(bars4, diff_values):
                if val > 0:
                    height = bar.get_height()
                    axes[1, 1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(val)}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=10,
                    )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "All strategies predict the same",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 1].set_title(
                "Analysis of Different Predictions", fontsize=14, fontweight="bold"
            )
    else:
        # Two-way comparison (original)
        differences = []
        for wp, sp in zip(weighted_preds, simple_preds):
            if wp["predicted_specy"] != sp["predicted_specy"]:
                differences.append(
                    {
                        "weighted_correct": wp["correct"],
                        "simple_correct": sp["correct"],
                        "ground_truth": wp["ground_truth"],
                    }
                )

        if differences:
            diff_categories = [
                "Both Wrong\n(Different Pred)",
                "Weighted Better",
                "Simple Better",
                "Both Correct\n(Different Pred)",
            ]

            both_wrong_diff = sum(
                1
                for d in differences
                if not d["weighted_correct"] and not d["simple_correct"]
            )
            weighted_better_diff = sum(
                1
                for d in differences
                if d["weighted_correct"] and not d["simple_correct"]
            )
            simple_better_diff = sum(
                1
                for d in differences
                if not d["weighted_correct"] and d["simple_correct"]
            )
            both_correct_diff = sum(
                1 for d in differences if d["weighted_correct"] and d["simple_correct"]
            )

            diff_values = [
                both_wrong_diff,
                weighted_better_diff,
                simple_better_diff,
                both_correct_diff,
            ]
            colors_chart4 = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#95e1d3"]

            bars4 = axes[1, 1].bar(
                diff_categories,
                diff_values,
                color=colors_chart4,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )
            axes[1, 1].set_ylabel("Number of Cases", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                f"Analysis of {len(differences)} Different Predictions",
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 1].set_ylim(
                0, max(diff_values) * 1.3 if max(diff_values) > 0 else 1
            )
            axes[1, 1].grid(axis="y", alpha=0.3)
            axes[1, 1].set_axisbelow(True)

            for bar, val in zip(bars4, diff_values):
                if val > 0:
                    height = bar.get_height()
                    axes[1, 1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(val)}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        fontsize=11,
                    )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No differences in predictions",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
                fontsize=14,
                fontweight="bold",
            )
            axes[1, 1].set_title(
                "Analysis of Different Predictions", fontsize=14, fontweight="bold"
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "strategy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved strategy comparison chart to: {output_path}")
    plt.close()


def visualize_ground_truth_rankings(analysis: Dict[str, Any], output_dir: str):
    """
    Visualize the rank of ground truth in aggregated results for false cases.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = [
        ("ColorHistogramHS", "colorhist"),
        ("ResNet50", "resnet"),
        ("EfficientNetV2B0", "efficient"),
    ]

    for idx, (model_name, model_key) in enumerate(models):
        # Collect ranks for all false cases from ColorHist perspective
        ranks = []

        for case in analysis["all_colorhist_wrong"]:
            if model_key == "colorhist":
                pred = case
            else:
                # Find corresponding prediction in other models
                # We need to get from the analysis structure
                # For now, collect from the colorhist wrong cases
                pred = None

        # Alternative: collect from the wrong cases lists
        if model_key == "colorhist":
            cases_to_check = analysis["all_colorhist_wrong"]
        else:
            # Collect all cases where we have comparison data
            cases_to_check = []
            for case_dict in (
                analysis["colorhist_wrong_resnet_correct"]
                + analysis["colorhist_wrong_efficient_correct"]
                + analysis["colorhist_wrong_both_correct"]
            ):
                if model_key in case_dict and case_dict[model_key]:
                    cases_to_check.append(case_dict[model_key])

        # Get ranks
        for pred in cases_to_check:
            rank = get_ground_truth_rank(pred.aggregated_results, pred.ground_truth)
            if rank is not None:
                ranks.append(rank)

        if not ranks:
            continue

        # Plot histogram
        max_rank = max(ranks)
        bins = range(1, min(max_rank + 2, 21))  # Cap at top 20

        axes[idx].hist(ranks, bins=bins, edgecolor="black", alpha=0.7, color="skyblue")
        axes[idx].set_xlabel("Rank of Ground Truth", fontsize=11)
        axes[idx].set_ylabel("Frequency", fontsize=11)
        axes[idx].set_title(
            f"{model_name}\nGround Truth Rankings (False Cases Only)",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].set_axisbelow(True)

        # Add statistics
        mean_rank = np.mean(ranks)
        median_rank = np.median(ranks)
        axes[idx].axvline(
            mean_rank,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_rank:.1f}",
        )
        axes[idx].axvline(
            median_rank,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_rank:.1f}",
        )
        axes[idx].legend()

        # Add text box with stats
        stats_text = (
            f"n = {len(ranks)}\nMean: {mean_rank:.2f}\nMedian: {median_rank:.1f}"
        )
        axes[idx].text(
            0.98,
            0.97,
            stats_text,
            transform=axes[idx].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontsize=10,
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ground_truth_rankings.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved ground truth rankings chart to: {output_path}")
    plt.close()


def draw_confusion_matrix(
    predictions: List[Dict[str, Any]],
    output_path: str,
    strategy_name: str = "Ensemble",
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Draw confusion matrix from ensemble prediction results.

    Args:
        predictions: List of prediction dictionaries with 'ground_truth' and 'predicted_specy'
        output_path: Path to save the confusion matrix plot
        strategy_name: Name of the strategy for the title
        figsize: Figure size (width, height)
    """
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("Error: scikit-learn is required for confusion matrix.")
        print("Install with: uv sync")
        return

    # Extract ground truth and predictions
    y_true = []
    y_pred = []

    for pred in predictions:
        if pred.get("ground_truth") and pred.get("predicted_specy"):
            y_true.append(pred["ground_truth"])
            y_pred.append(pred["predicted_specy"])

    if not y_true:
        print(f"No valid predictions to create confusion matrix for {strategy_name}")
        return

    # Get unique labels sorted alphabetically
    labels = sorted(list(set(y_true + y_pred)))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate accuracy
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    # Shorten species names for display
    short_labels = [label.replace("Penicillium ", "P. ") for label in labels]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        square=True,
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix - {strategy_name}\n(Accuracy: {accuracy:.2%})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("True Species", fontsize=12)
    ax.set_xlabel("Predicted Species", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix to: {output_path}")

    # Print classification report
    print(f"\n{strategy_name} - Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    plt.close()


# ========== Ensemble Combination Functions ==========


def combine_aggregated_results(
    predictions: List[Tuple[PredictionResult, float]],
    strategy: str = "weighted_sum",
    debug: bool = False,
    species_weights: Optional[SpeciesWeights] = None,
) -> List[Tuple[str, float]]:
    """
    Combine aggregated results from multiple models.

    Args:
        predictions: List of (PredictionResult, weight) tuples
        strategy: Combination strategy:
            - 'weighted_sum': Multiply scores by model accuracy weights and sum
            - 'average': Weighted average (normalized by total weight)
            - 'simple_average': Unweighted average (ignores weights)
            - 'manual_weighted': Use species-specific manual weights (requires species_weights)
        debug: Enable debug output
        species_weights: SpeciesWeights object for manual weighting strategy

    Returns:
        List of (species, combined_score) sorted by score descending
    """
    species_scores: DefaultDict[str, float] = defaultdict(float)
    total_weight = sum(weight for _, weight in predictions)
    num_models = len(predictions)

    if debug:
        print("  [DEBUG combine_aggregated_results]")
        print(f"    Strategy: {strategy}")
        print(f"    Num models: {num_models}")
        print(f"    Total weight: {total_weight}")
        print(f"    Weights: {[weight for _, weight in predictions]}")
        if strategy == "manual_weighted" and species_weights:
            print(
                f"    Using manual species weights: {len(species_weights.weights)} species"
            )

    for pred, weight in predictions:
        for agg_result in pred.aggregated_results:
            if strategy == "weighted_sum":
                contribution = agg_result.score * weight
                species_scores[agg_result.specy] += contribution
            elif strategy == "simple_average":
                # Ignore weights, just average the scores
                contribution = agg_result.score / num_models
                species_scores[agg_result.specy] += contribution
            elif strategy == "manual_weighted":
                # Use species-specific manual weights
                if species_weights:
                    # Get species-specific weight for this model, or default
                    species_name = agg_result.specy
                    if species_name in species_weights.weights:
                        model_weight = species_weights.weights[species_name].get(
                            pred.feature_extractor, 1.0
                        )
                    else:
                        # Use default weights
                        model_weight = species_weights.weights.get("default", {}).get(
                            pred.feature_extractor, 1.0
                        )

                    contribution = agg_result.score * model_weight
                    species_scores[agg_result.specy] += contribution

                    if debug and agg_result.specy in [
                        "Penicillium aurantiogriseum",
                        "Penicillium viridicatum",
                        "Penicillium cyclopium",
                        "Penicillium freii",
                    ]:
                        print(
                            f"    Model {pred.feature_extractor}: {agg_result.specy} score={agg_result.score:.4f}, manual_weight={model_weight:.4f}, contribution={contribution:.4f}"
                        )
                else:
                    # Fallback to simple average if no species weights provided
                    contribution = agg_result.score / num_models
                    species_scores[agg_result.specy] += contribution
            else:  # average (weighted average)
                contribution = agg_result.score * weight / total_weight
                species_scores[agg_result.specy] += contribution

            if (
                debug
                and strategy != "manual_weighted"
                and agg_result.specy
                in ["Penicillium aurantiogriseum", "Penicillium viridicatum"]
            ):
                print(
                    f"    Model {pred.feature_extractor}: {agg_result.specy} score={agg_result.score:.4f}, weight={weight:.4f}, contribution={contribution:.4f}"
                )

    # Sort by score descending
    sorted_results = sorted(species_scores.items(), key=lambda x: x[1], reverse=True)

    if debug:
        print("    Top 3 combined:")
        for i, (species, score) in enumerate(sorted_results[:3], 1):
            print(f"      {i}. {species}: {score:.4f}")

    return sorted_results


def evaluate_ensemble(
    results_dict: Dict[str, EvaluationResults],
    accuracy_dict: Dict[Tuple[str, str, str], AccuracyData],
    feature_extractors: List[str],
    env_strategy: str = "E2",
    agg_strategy: str = "S1",
    combination_strategy: str = "weighted_sum",
    species_weights: Optional[SpeciesWeights] = None,
) -> Dict[str, Any]:
    """
    Evaluate ensemble model by combining multiple feature extractors.

    Args:
        results_dict: Dictionary mapping feature_extractor to EvaluationResults
        accuracy_dict: Dictionary with accuracy data
        feature_extractors: List of feature extractors to combine
        env_strategy: Environment strategy to use
        agg_strategy: Aggregation strategy to use
        combination_strategy: How to combine scores:
            - 'weighted_sum': Use model accuracy as weights
            - 'average': Weighted average (normalized)
            - 'simple_average': Unweighted average
            - 'manual_weighted': Use species-specific manual weights
        species_weights: SpeciesWeights object for manual weighting (required for 'manual_weighted')

    Returns:
        Dictionary with ensemble evaluation results
    """
    # Get weights from accuracy
    weights: Dict[str, float] = {}
    for fe in feature_extractors:
        key = (fe, env_strategy, agg_strategy)
        if key in accuracy_dict:
            weights[fe] = accuracy_dict[key].accuracy
        else:
            print(
                f"Warning: No accuracy data found for {fe}, {env_strategy}, {agg_strategy}"
            )
            weights[fe] = 0.0

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    print("\nEnsemble weights:")
    for fe, w in weights.items():
        print(f"  {fe}: {w:.4f}")

    # Create lookup for predictions by (strain, test_set_index)
    predictions_by_key: DefaultDict[tuple[str, int], Dict[str, PredictionResult]] = defaultdict(dict)

    for fe, results in results_dict.items():
        for pred in results.predictions:
            prediction_key: tuple[str, int] = (pred.strain, pred.test_set_index)
            predictions_by_key[prediction_key][fe] = pred

    # Combine predictions
    ensemble_predictions: List[Dict[str, Any]] = []
    correct_count = 0

    for prediction_key, preds in predictions_by_key.items():
        # Check if we have all models for this key
        if len(preds) != len(feature_extractors):
            continue

        # Get ground truth from first prediction
        ground_truth = list(preds.values())[0].ground_truth
        strain = list(preds.values())[0].strain
        test_set_index = list(preds.values())[0].test_set_index

        # Combine aggregated results
        pred_weight_pairs = [(preds[fe], weights[fe]) for fe in feature_extractors]
        debug_first = len(ensemble_predictions) == 0
        combined_results = combine_aggregated_results(
            pred_weight_pairs,
            strategy=combination_strategy,
            debug=debug_first,
            species_weights=species_weights,
        )

        # Debug: Print first prediction details
        if debug_first:
            print(f"\n[DEBUG] First prediction - {strain} (test set {test_set_index}):")
            print(f"  Ground truth: {ground_truth}")
            for fe in feature_extractors:
                pred = preds[fe]
                print(f"  {fe}:")
                print(f"    Predicted: {pred.predicted_specy}")
                print(f"    Confidence: {pred.predicted_confidence:.4f}")
                print("    Aggregated results (top 3):")
                for i, agg in enumerate(pred.aggregated_results[:3], 1):
                    print(f"      {i}. {agg.specy}: {agg.score:.4f}")
            print("  Combined results (top 3):")
            for i, (species, score) in enumerate(combined_results[:3], 1):
                print(f"    {i}. {species}: {score:.4f}")

        # Get top prediction
        predicted_specy = combined_results[0][0] if combined_results else "unknown"
        predicted_confidence = combined_results[0][1] if combined_results else 0.0
        correct = predicted_specy == ground_truth

        if correct:
            correct_count += 1

        ensemble_predictions.append(
            {
                "strain": strain,
                "test_set_index": test_set_index,
                "ground_truth": ground_truth,
                "predicted_specy": predicted_specy,
                "predicted_confidence": predicted_confidence,
                "correct": correct,
                "combined_results": combined_results[:10],  # Top 10
                "individual_predictions": {
                    fe: {
                        "predicted": preds[fe].predicted_specy,
                        "confidence": preds[fe].predicted_confidence,
                        "correct": preds[fe].correct,
                    }
                    for fe in feature_extractors
                },
            }
        )

    total_predictions = len(ensemble_predictions)
    accuracy = correct_count / total_predictions if total_predictions > 0 else 0.0

    print("\n[DEBUG] Ensemble evaluation complete:")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Correct predictions: {correct_count}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Strategy: {combination_strategy}")

    # Verify all predictions have aggregated results
    missing_agg = sum(1 for p in ensemble_predictions if not p.get("combined_results"))
    if missing_agg > 0:
        print(f"  [WARNING] {missing_agg} predictions missing combined_results!")

    return {
        "ensemble_predictions": ensemble_predictions,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_predictions": total_predictions,
        "weights": weights,
        "feature_extractors": feature_extractors,
        "env_strategy": env_strategy,
        "agg_strategy": agg_strategy,
        "combination_strategy": combination_strategy,
    }


def save_ensemble_results(
    ensemble_result: Dict[str, Any], output_dir: str, suffix: str = ""
):
    """Save ensemble results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    filename = f"ensemble_results{suffix}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as f:
        json.dump(ensemble_result, f, indent=2)

    print(f"\nSaved ensemble results to: {output_path}")


def print_ensemble_summary(ensemble_result: Dict[str, Any]):
    """Print summary of ensemble results."""
    print("\n" + "=" * 80)
    print(
        f"ENSEMBLE EVALUATION RESULTS ({ensemble_result.get('combination_strategy', 'weighted_sum').upper()})"
    )
    print("=" * 80)
    print(f"\nFeature Extractors: {', '.join(ensemble_result['feature_extractors'])}")
    print(f"Environment Strategy: {ensemble_result['env_strategy']}")
    print(f"Aggregation Strategy: {ensemble_result['agg_strategy']}")
    print(
        f"Combination Strategy: {ensemble_result.get('combination_strategy', 'weighted_sum')}"
    )
    print(f"\nAccuracy: {ensemble_result['accuracy'] * 100:.2f}%")
    print(
        f"Correct: {ensemble_result['correct_count']}/{ensemble_result['total_predictions']}"
    )

    print("\n" + "-" * 80)
    print("Comparison with Individual Models:")
    print("-" * 80)

    for fe in ensemble_result["feature_extractors"]:
        weight = ensemble_result["weights"][fe]
        # Count individual accuracy from predictions
        individual_correct = sum(
            1
            for p in ensemble_result["ensemble_predictions"]
            if p["individual_predictions"][fe]["correct"]
        )
        individual_acc = individual_correct / ensemble_result["total_predictions"]
        print(f"{fe:25s}: {individual_acc * 100:6.2f}% (weight: {weight:.4f})")

    print(f"\nEnsemble                 : {ensemble_result['accuracy'] * 100:6.2f}%")
    print("=" * 80)


# ========== Complementary Case Visualization Functions ==========


def regenerate_prediction_with_details(
    strain: str, test_set_index: int, feature_extractor: str, metadata: Dict
) -> Dict[str, Any]:
    """
    Regenerate a prediction with full raw_results for visualization.
    Uses the predict() function to get detailed neighbor information.

    Note: This uses the single collection 'myco_fungi_features_full' with named vectors.
    """
    from qdrant_client import QdrantClient

    from src.experiments.retrieval.run import get_extractor_by_name, predict_segment_group

    # Single collection with multiple named vectors
    COLLECTION_NAME = "myco_fungi_features_full"

    # Create Qdrant client (default localhost:6333)
    qdrant_client = QdrantClient(host="localhost", port=6333)

    k = int(metadata.get("k", 7))
    environment = metadata.get("environment", "all")
    strategy = str(metadata.get("strategy", "weighted"))
    extractor_aliases = {
        "ColorHistogramHS": "colorhistogram_HS",
        "ResNet50": "resnet50",
        "EfficientNetV2B0": "efficientnetv2b0",
    }
    extractor = get_extractor_by_name(extractor_aliases.get(feature_extractor, feature_extractor))
    if extractor is None:
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")

    from src.experiments.retrieval.run import collect_testset

    test_sets = collect_testset(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        strain=strain,
        environment_strategy=str(environment or "E1"),
    )
    if test_set_index >= len(test_sets):
        raise IndexError(f"test_set_index out of range: {test_set_index}")

    return predict_segment_group(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        test_group=test_sets[test_set_index],
        strain=strain,
        feature_extractor=extractor,
        k=k,
        min_samples=None,
        without_siblings=True,
        environment=environment if environment != "all" else None,
        strategy=strategy,
    )


def visualize_complementary_cases(
    analysis: Dict[str, Any],
    results_dict: Dict[str, EvaluationResults],
    output_dir: str,
    k: int = 7,
):
    """
    Create visualizations for cases where ColorHistogramHS is wrong but other models are correct.
    Creates separate directories for resnet_only, efficient_only, and wrong_colorhistogramhs.
    """
    from src.analysis.visualization.visualize_prediction import (
        visualize_prediction_by_environment,
    )
    from src.config import SEGMENTED_IMAGE_DIR

    print("\n" + "=" * 80)
    print("VISUALIZING COMPLEMENTARY CASES")
    print("=" * 80)

    viz_output_dir = os.path.join(output_dir, "complementary_cases")
    os.makedirs(viz_output_dir, exist_ok=True)

    # Extract metadata from results
    colorhist_meta = {
        "k": results_dict["ColorHistogramHS"].metadata.k,
        "environment": results_dict["ColorHistogramHS"].metadata.environment,
        "strategy": results_dict["ColorHistogramHS"].metadata.aggregation_strategy,
    }
    resnet_meta = {
        "k": results_dict["ResNet50"].metadata.k,
        "environment": results_dict["ResNet50"].metadata.environment,
        "strategy": results_dict["ResNet50"].metadata.aggregation_strategy,
    }
    efficient_meta = {
        "k": results_dict["EfficientNetV2B0"].metadata.k,
        "environment": results_dict["EfficientNetV2B0"].metadata.environment,
        "strategy": results_dict["EfficientNetV2B0"].metadata.aggregation_strategy,
    }

    # 1. ResNet50 corrections
    resnet_cases = analysis["colorhist_wrong_resnet_correct"]
    if resnet_cases:
        print(
            f"\n[1/3] Creating {len(resnet_cases)} ResNet50 correction visualizations..."
        )
        resnet_output_dir = os.path.join(viz_output_dir, "resnet_only")
        os.makedirs(resnet_output_dir, exist_ok=True)

        for idx, case in enumerate(resnet_cases, 1):
            strain = case["resnet"].strain
            test_idx = case["resnet"].test_set_index
            print(f"  [{idx}/{len(resnet_cases)}] {strain} test_set_{test_idx}...")
            try:
                result = regenerate_prediction_with_details(
                    strain, test_idx, "ResNet50", resnet_meta
                )
                output_path = os.path.join(
                    resnet_output_dir,
                    f"{idx:03d}_{strain.replace(' ', '_')}_correct.jpg",
                )
                visualize_prediction_by_environment(
                    prediction_result=result,
                    segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                    output_path=output_path,
                    k=k,
                )
                print("    ✓ Saved")
            except Exception as e:
                print(f"    ✗ Error: {e}")
    else:
        print("\n[1/3] No ResNet50 corrections found - skipping")

    # 2. EfficientNetV2B0 corrections
    efficient_cases = analysis["colorhist_wrong_efficient_correct"]
    if efficient_cases:
        print(
            f"\n[2/3] Creating {len(efficient_cases)} EfficientNetV2B0 correction visualizations..."
        )
        efficient_output_dir = os.path.join(viz_output_dir, "efficient_only")
        os.makedirs(efficient_output_dir, exist_ok=True)

        for idx, case in enumerate(efficient_cases, 1):
            strain = case["efficient"].strain
            test_idx = case["efficient"].test_set_index
            print(f"  [{idx}/{len(efficient_cases)}] {strain} test_set_{test_idx}...")
            try:
                result = regenerate_prediction_with_details(
                    strain, test_idx, "EfficientNetV2B0", efficient_meta
                )
                output_path = os.path.join(
                    efficient_output_dir,
                    f"{idx:03d}_{strain.replace(' ', '_')}_correct.jpg",
                )
                visualize_prediction_by_environment(
                    prediction_result=result,
                    segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                    output_path=output_path,
                    k=k,
                )
                print("    ✓ Saved")
            except Exception as e:
                print(f"    ✗ Error: {e}")
    else:
        print("\n[2/3] No EfficientNetV2B0 corrections found - skipping")

    # 3. All ColorHistogramHS wrong cases
    all_wrong = analysis["all_colorhist_wrong"]
    print(f"\n[3/3] Creating {len(all_wrong)} ColorHistogramHS wrong visualizations...")
    wrong_output_dir = os.path.join(viz_output_dir, "wrong_colorhistogramhs")
    os.makedirs(wrong_output_dir, exist_ok=True)

    for idx, pred in enumerate(all_wrong, 1):
        strain = pred.strain
        test_idx = pred.test_set_index
        print(f"  [{idx}/{len(all_wrong)}] {strain} test_set_{test_idx}...")
        try:
            result = regenerate_prediction_with_details(
                strain, test_idx, "ColorHistogramHS", colorhist_meta
            )
            output_path = os.path.join(
                wrong_output_dir, f"{idx:03d}_{strain.replace(' ', '_')}_false.jpg"
            )
            visualize_prediction_by_environment(
                prediction_result=result,
                segmented_image_dir=str(SEGMENTED_IMAGE_DIR),
                output_path=output_path,
                k=k,
            )
            print("    ✓ Saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")

    # Summary
    print("\n" + "-" * 80)
    print("COMPLEMENTARY CASE VISUALIZATIONS COMPLETE")
    print("-" * 80)
    print("\nOutput directories:")
    if resnet_cases:
        print(f"  - {os.path.join(viz_output_dir, 'resnet_only')}/")
        print(
            f"    ({len(resnet_cases)} cases where ResNet50 correct, ColorHistogramHS wrong)"
        )
    if efficient_cases:
        print(f"  - {os.path.join(viz_output_dir, 'efficient_only')}/")
        print(
            f"    ({len(efficient_cases)} cases where EfficientNetV2B0 correct, ColorHistogramHS wrong)"
        )
    print(f"  - {os.path.join(viz_output_dir, 'wrong_colorhistogramhs')}/")
    print(f"    ({len(all_wrong)} cases where ColorHistogramHS wrong)")

    if all_wrong:
        correction_rate = (
            (len(resnet_cases) + len(efficient_cases)) / len(all_wrong) * 100
        )
        print(
            f"\nCorrection rate: {correction_rate:.1f}% of ColorHistogramHS errors were corrected by other models"
        )


# ========== Main Function ==========


def main():
    """Main function to run ensemble analysis."""

    # Configuration
    BASE_DIR = str(RESULTS_DIR / "comprehensive_k7_NoSib_6")
    CSV_PATH = os.path.join(
        BASE_DIR, "comprehensive_evaluation_summary_20251211_135108.csv"
    )
    OUTPUT_DIR = str(RESULTS_DIR / "ensemble_analysis")

    ENV_STRATEGY = "E2"
    AGG_STRATEGY = "S1"  # AVG

    FEATURE_EXTRACTORS = ["ColorHistogramHS", "ResNet50", "EfficientNetV2B0"]

    print("=" * 80)
    print("ENSEMBLE ANALYSIS")
    print("=" * 80)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Environment Strategy: {ENV_STRATEGY}")
    print(f"Aggregation Strategy: {AGG_STRATEGY} (AVG)")
    print(f"Feature Extractors: {', '.join(FEATURE_EXTRACTORS)}")

    # Load accuracy data from CSV
    print("\n" + "-" * 80)
    print("Loading accuracy data from CSV...")
    accuracy_dict = load_accuracy_from_csv(CSV_PATH)
    print(f"Loaded accuracy data for {len(accuracy_dict)} configurations")

    # Load results for each feature extractor
    print("\n" + "-" * 80)
    print("Loading evaluation results...")
    results_dict = {}

    for fe in FEATURE_EXTRACTORS:
        result_file = find_result_file(BASE_DIR, fe, ENV_STRATEGY, AGG_STRATEGY)
        if result_file:
            print(f"  Loading {fe}...")
            results_dict[fe] = load_results_json(result_file)
        else:
            print(f"  WARNING: Could not find results for {fe}")

    if len(results_dict) != len(FEATURE_EXTRACTORS):
        print("\nError: Not all feature extractor results found!")
        return

    # Verify aggregated results are loaded
    print("\n" + "-" * 80)
    print("Verifying aggregated results...")
    for fe, results in results_dict.items():
        first_pred = results.predictions[0]
        num_agg = len(first_pred.aggregated_results)
        print(f"  {fe}: First prediction has {num_agg} aggregated results")
        if num_agg > 0:
            print(f"    Top species: {first_pred.aggregated_results[0].specy}")
            print(f"    Top score: {first_pred.aggregated_results[0].score:.4f}")
        else:
            print("    [ERROR] No aggregated results found!")

    # Analyze false cases
    print("\n" + "-" * 80)
    print("Analyzing false cases...")
    analysis = analyze_false_cases(
        results_dict["ColorHistogramHS"],
        results_dict["ResNet50"],
        results_dict["EfficientNetV2B0"],
    )

    print(f"\nTotal ColorHistogramHS wrong cases: {analysis['total_colorhist_wrong']}")
    print(
        f"  - ResNet50 correct only: {len(analysis['colorhist_wrong_resnet_correct'])}"
    )
    print(
        f"  - EfficientNetV2B0 correct only: {len(analysis['colorhist_wrong_efficient_correct'])}"
    )
    print(f"  - Both correct: {len(analysis['colorhist_wrong_both_correct'])}")

    # Save complementary cases list for visualization
    complementary_cases_dir = os.path.join(OUTPUT_DIR, "complementary_cases")
    os.makedirs(complementary_cases_dir, exist_ok=True)

    complementary_cases_list = {
        "resnet_only": [
            {
                "strain": case["colorhist"].strain,
                "test_set_index": case["colorhist"].test_set_index,
                "ground_truth": case["colorhist"].ground_truth,
            }
            for case in analysis["colorhist_wrong_resnet_correct"]
        ],
        "efficient_only": [
            {
                "strain": case["colorhist"].strain,
                "test_set_index": case["colorhist"].test_set_index,
                "ground_truth": case["colorhist"].ground_truth,
            }
            for case in analysis["colorhist_wrong_efficient_correct"]
        ],
        "all_colorhist_wrong": [
            {
                "strain": pred.strain,
                "test_set_index": pred.test_set_index,
                "ground_truth": pred.ground_truth,
            }
            for pred in analysis["all_colorhist_wrong"]
        ],
    }

    complementary_cases_path = os.path.join(
        complementary_cases_dir, "complementary_cases_list.json"
    )
    with open(complementary_cases_path, "w") as f:
        json.dump(complementary_cases_list, f, indent=2)
    print(f"\nSaved complementary cases list to: {complementary_cases_path}")

    # Create visualizations
    print("\n" + "-" * 80)
    print("Creating visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visualize_correction_analysis(analysis, OUTPUT_DIR)
    visualize_ground_truth_rankings(analysis, OUTPUT_DIR)

    # Evaluate ensemble with weighted strategy
    print("\n" + "-" * 80)
    print("Evaluating ensemble model (weighted)...")
    ensemble_result_weighted = evaluate_ensemble(
        results_dict,
        accuracy_dict,
        FEATURE_EXTRACTORS,
        ENV_STRATEGY,
        AGG_STRATEGY,
        combination_strategy="weighted_sum",
    )

    # Evaluate ensemble with simple average strategy
    print("\n" + "-" * 80)
    print("Evaluating ensemble model (simple average)...")
    ensemble_result_simple = evaluate_ensemble(
        results_dict,
        accuracy_dict,
        FEATURE_EXTRACTORS,
        ENV_STRATEGY,
        AGG_STRATEGY,
        combination_strategy="simple_average",
    )

    # Load species weights and evaluate manual weighted strategy
    print("\n" + "-" * 80)
    print("Loading species weights...")
    species_weights = None
    ensemble_result_manual = None

    if os.path.exists(SPECIES_WEIGHTS_PATH):
        species_weights = load_species_weights(str(SPECIES_WEIGHTS_PATH))

        print("\n" + "-" * 80)
        print("Evaluating ensemble model (manual weighted)...")
        ensemble_result_manual = evaluate_ensemble(
            results_dict,
            accuracy_dict,
            FEATURE_EXTRACTORS,
            ENV_STRATEGY,
            AGG_STRATEGY,
            combination_strategy="manual_weighted",
            species_weights=species_weights,
        )
    else:
        print(f"  Species weights file not found: {SPECIES_WEIGHTS_PATH}")
        print("  Skipping manual weighted strategy")

    # Save results
    save_ensemble_results(ensemble_result_weighted, OUTPUT_DIR, suffix="_weighted")
    save_ensemble_results(ensemble_result_simple, OUTPUT_DIR, suffix="_simple_avg")
    if ensemble_result_manual:
        save_ensemble_results(
            ensemble_result_manual, OUTPUT_DIR, suffix="_manual_weighted"
        )

    # Print summaries
    print_ensemble_summary(ensemble_result_weighted)
    print_ensemble_summary(ensemble_result_simple)
    if ensemble_result_manual:
        print_ensemble_summary(ensemble_result_manual)

    # Compare strategies
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(
        f"\nWeighted Ensemble:       {ensemble_result_weighted['accuracy'] * 100:.2f}% ({ensemble_result_weighted['correct_count']}/{ensemble_result_weighted['total_predictions']})"
    )
    print(
        f"Simple Average Ensemble: {ensemble_result_simple['accuracy'] * 100:.2f}% ({ensemble_result_simple['correct_count']}/{ensemble_result_simple['total_predictions']})"
    )
    if ensemble_result_manual:
        print(
            f"Manual Weighted Ensemble: {ensemble_result_manual['accuracy'] * 100:.2f}% ({ensemble_result_manual['correct_count']}/{ensemble_result_manual['total_predictions']})"
        )
    print(
        f"Best Individual Model:   {max(ensemble_result_weighted['weights'].values()) * 100:.2f}% (ColorHistogramHS)"
    )

    # Determine best strategy
    strategies = [
        ("Weighted", ensemble_result_weighted["accuracy"]),
        ("Simple Average", ensemble_result_simple["accuracy"]),
    ]
    if ensemble_result_manual:
        strategies.append(("Manual Weighted", ensemble_result_manual["accuracy"]))

    best_strategy = max(strategies, key=lambda x: x[1])
    print(
        f"\nBest Ensemble Strategy: {best_strategy[0]} ({best_strategy[1] * 100:.2f}%)"
    )

    # Analyze differences
    diff_count = 0
    improved_with_simple = 0
    for i, (wp, sp) in enumerate(
        zip(
            ensemble_result_weighted["ensemble_predictions"],
            ensemble_result_simple["ensemble_predictions"],
        )
    ):
        if wp["predicted_specy"] != sp["predicted_specy"]:
            diff_count += 1
            if sp["correct"] and not wp["correct"]:
                improved_with_simple += 1

    print(
        f"\nPredictions that differ: {diff_count}/{ensemble_result_weighted['total_predictions']}"
    )
    print(f"Cases where simple average is better: {improved_with_simple}")

    # Create strategy comparison visualization
    print("\n" + "-" * 80)
    print("Creating strategy comparison visualizations...")
    create_strategy_comparison_chart(
        ensemble_result_weighted,
        ensemble_result_simple,
        OUTPUT_DIR,
        ensemble_result_manual,
    )

    # Generate confusion matrices for each strategy
    print("\n" + "-" * 80)
    print("Generating confusion matrices...")

    # Weighted strategy
    draw_confusion_matrix(
        ensemble_result_weighted["ensemble_predictions"],
        os.path.join(OUTPUT_DIR, "confusion_matrix_weighted.png"),
        strategy_name="Weighted Sum Ensemble",
    )

    # Simple average strategy
    draw_confusion_matrix(
        ensemble_result_simple["ensemble_predictions"],
        os.path.join(OUTPUT_DIR, "confusion_matrix_simple_avg.png"),
        strategy_name="Simple Average Ensemble",
    )

    # Manual weighted strategy (if available)
    if ensemble_result_manual:
        draw_confusion_matrix(
            ensemble_result_manual["ensemble_predictions"],
            os.path.join(OUTPUT_DIR, "confusion_matrix_manual_weighted.png"),
            strategy_name="Manual Weighted Ensemble",
        )

    # Also generate confusion matrices for individual models
    print("\n" + "-" * 80)
    print("Generating confusion matrices for individual models...")

    for fe in FEATURE_EXTRACTORS:
        # Extract individual predictions from ensemble results
        individual_predictions = []
        for pred in ensemble_result_weighted["ensemble_predictions"]:
            individual_predictions.append(
                {
                    "ground_truth": pred["ground_truth"],
                    "predicted_specy": pred["individual_predictions"][fe]["predicted"],
                }
            )

        draw_confusion_matrix(
            individual_predictions,
            os.path.join(OUTPUT_DIR, f"confusion_matrix_{fe.lower()}.png"),
            strategy_name=f"{fe} (Individual)",
        )

    # Note about complementary case visualizations
    print("\n" + "-" * 80)
    print("NOTE: Complementary case visualizations")
    print("-" * 80)
    print("To create detailed visualizations of specific cases:")
    print("  1. Cases where ResNet50 correct but ColorHistogramHS wrong")
    print("  2. Cases where EfficientNetV2B0 correct but ColorHistogramHS wrong")
    print("  3. All cases where ColorHistogramHS wrong")
    print()
    print("Run the visualize_complementary_cases.py script separately,")
    print(
        "or use example_visualize_prediction.py with specific strain/test_set combinations."
    )
    print()
    print(
        f"Found {len(analysis['colorhist_wrong_resnet_correct'])} ResNet50 corrections"
    )
    print(
        f"Found {len(analysis['colorhist_wrong_efficient_correct'])} EfficientNetV2B0 corrections"
    )
    print(
        f"Found {len(analysis['all_colorhist_wrong'])} ColorHistogramHS wrong cases total"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("  - correction_analysis.png")
    print("  - ground_truth_rankings.png")
    print("  - strategy_comparison.png")
    print("  - ensemble_results_weighted.json")
    print("  - ensemble_results_simple_avg.json")
    if ensemble_result_manual:
        print("  - ensemble_results_manual_weighted.json")
    print("  - complementary_cases/ (detailed visualizations)")
    print("    * resnet_only/ (cases where ResNet50 correct, ColorHistogramHS wrong)")
    print(
        "    * efficient_only/ (cases where EfficientNetV2B0 correct, ColorHistogramHS wrong)"
    )
    print("    * wrong_colorhistogramhs/ (all ColorHistogramHS wrong cases)")


if __name__ == "__main__":
    main()
