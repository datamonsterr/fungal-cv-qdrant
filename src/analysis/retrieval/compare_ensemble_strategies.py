"""
Compare weighted vs simple average ensemble strategies in detail.
"""

import json

from src.config import RESULTS_DIR


def load_ensemble_results(json_path: str):
    """Load ensemble results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def compare_strategies():
    """Compare weighted and simple average strategies."""

    results_dir = RESULTS_DIR / "ensemble_analysis"

    print("=" * 80)
    print("DETAILED STRATEGY COMPARISON")
    print("=" * 80)

    # Load both results
    weighted_results = load_ensemble_results(
        str(results_dir / "ensemble_results_weighted.json")
    )
    simple_results = load_ensemble_results(
        str(results_dir / "ensemble_results_simple_avg.json")
    )

    # Find differences
    differences = []
    for w_pred, s_pred in zip(
        weighted_results["ensemble_predictions"], simple_results["ensemble_predictions"]
    ):
        if w_pred["predicted_specy"] != s_pred["predicted_specy"]:
            differences.append(
                {
                    "strain": w_pred["strain"],
                    "test_set": w_pred["test_set_index"],
                    "ground_truth": w_pred["ground_truth"],
                    "weighted": {
                        "predicted": w_pred["predicted_specy"],
                        "confidence": w_pred["predicted_confidence"],
                        "correct": w_pred["correct"],
                        "top_3": w_pred["combined_results"][:3],
                    },
                    "simple": {
                        "predicted": s_pred["predicted_specy"],
                        "confidence": s_pred["predicted_confidence"],
                        "correct": s_pred["correct"],
                        "top_3": s_pred["combined_results"][:3],
                    },
                    "individual": w_pred["individual_predictions"],
                }
            )

    print(f"\nTotal predictions: {len(weighted_results['ensemble_predictions'])}")
    print(f"Predictions that differ: {len(differences)}")
    print(f"\nWeighted accuracy: {weighted_results['accuracy'] * 100:.2f}%")
    print(f"Simple average accuracy: {simple_results['accuracy'] * 100:.2f}%")

    if not differences:
        print("\nNo differences found between strategies!")
        return

    print("\n" + "=" * 80)
    print("DETAILED DIFFERENCES")
    print("=" * 80)

    for i, diff in enumerate(differences, 1):
        print(f"\n{'-' * 80}")
        print(f"Case {i}: {diff['strain']} (Test Set {diff['test_set']})")
        print(f"{'-' * 80}")
        print(f"Ground Truth: {diff['ground_truth']}")

        print("\n  Individual Model Predictions:")
        for model, pred in diff["individual"].items():
            status = "✓" if pred["correct"] else "✗"
            print(
                f"    {model:20s}: {pred['predicted']:35s} (conf: {pred['confidence']:.4f}) {status}"
            )

        print("\n  Weighted Ensemble:")
        w_status = "✓" if diff["weighted"]["correct"] else "✗"
        print(
            f"    Predicted: {diff['weighted']['predicted']} (conf: {diff['weighted']['confidence']:.4f}) {w_status}"
        )
        print("    Top 3:")
        for species, score in diff["weighted"]["top_3"]:
            marker = "→" if species == diff["ground_truth"] else " "
            print(f"      {marker} {species:35s}: {score:.4f}")

        print("\n  Simple Average Ensemble:")
        s_status = "✓" if diff["simple"]["correct"] else "✗"
        print(
            f"    Predicted: {diff['simple']['predicted']} (conf: {diff['simple']['confidence']:.4f}) {s_status}"
        )
        print("    Top 3:")
        for species, score in diff["simple"]["top_3"]:
            marker = "→" if species == diff["ground_truth"] else " "
            print(f"      {marker} {species:35s}: {score:.4f}")

        # Analyze which strategy is better
        if diff["weighted"]["correct"] and not diff["simple"]["correct"]:
            print("\n  ✓ Weighted strategy is correct")
        elif diff["simple"]["correct"] and not diff["weighted"]["correct"]:
            print("\n  ✓ Simple average strategy is correct")
        elif diff["weighted"]["correct"] and diff["simple"]["correct"]:
            print("\n  ✓ Both strategies correct (different but both right)")
        else:
            print("\n  ✗ Both strategies wrong")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    weighted_better = sum(
        1
        for d in differences
        if d["weighted"]["correct"] and not d["simple"]["correct"]
    )
    simple_better = sum(
        1
        for d in differences
        if d["simple"]["correct"] and not d["weighted"]["correct"]
    )
    both_correct = sum(
        1 for d in differences if d["weighted"]["correct"] and d["simple"]["correct"]
    )
    both_wrong = sum(
        1
        for d in differences
        if not d["weighted"]["correct"] and not d["simple"]["correct"]
    )

    print(f"\nOf {len(differences)} different predictions:")
    print(f"  - Weighted better:     {weighted_better}")
    print(f"  - Simple avg better:   {simple_better}")
    print(f"  - Both correct:        {both_correct}")
    print(f"  - Both wrong:          {both_wrong}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_strategies()
