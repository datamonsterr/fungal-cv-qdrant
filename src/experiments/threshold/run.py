"""
Threshold experiment — run_accuracy()

Returns the best F1 score across all threshold strategies.
Called by src/run.py when --experiment threshold is used.

Returns:
    dict of {f"{strategy}_{algo}": f1} for all combinations,
    or a single float if called directly.

Direct usage:
    uv run python -m src.experiments.threshold.run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_accuracy(strategy: str = "best", **kwargs) -> float | Dict[str, float]:
    """
    Run threshold analysis on the diverse dataset and return F1 score(s).

    Assumes results/threshold/diverse_retrieval_results.csv exists
    (run retrieve_diverse first if not).

    Args:
        strategy: "best" = return max F1 across all strategies (default, float)
                  "all"  = return dict of all {name: f1}
                  or a specific strategy name to return its best F1
    Returns:
        float or dict of strategy F1s. For best/all, also prints the best
        strategy name so the caller can use it as description.
    """
    from src.experiments.threshold.expanded_threshold_analysis import (
        INPUT_CSV as EXPANDED_INPUT_CSV,
        run as expanded_run,
    )
    from src.experiments.threshold.retrieve_diverse import retrieve_diverse

    if not EXPANDED_INPUT_CSV.exists():
        print("Retrieval CSV not found — running retrieve_diverse first...")
        retrieve_diverse()

    # Run expanded analysis (192 formulas × 3 algorithms = 576 experiments)
    print("Running expanded threshold analysis...")
    all_results, best_overall = expanded_run()

    # Flatten to {strategy_algo: f1}
    flat: Dict[str, float] = {}
    for r in all_results:
        key = f"{r['formula']}_{r['algorithm']}"
        flat[key] = float(r["f1"])

    if strategy == "all":
        return flat
    if strategy == "best":
        # Return the full dict so autoresearch plots all strategies
        # Also identify best strategy name for description
        if flat:
            best_key = max(flat.keys(), key=lambda k: flat[k])
            best_f1 = flat[best_key]
            print(f"[threshold] best strategy: {best_key} = {best_f1:.6f}")
        return flat
    if strategy in all_f1s:
        return max(all_f1s[strategy].values())
    return max(flat.values()) if flat else 0.0


if __name__ == "__main__":
    result = run_accuracy(strategy="all")
    print(f"\nAll strategy-algorithm F1 scores:")
    for name, f1 in sorted(result.items(), key=lambda x: -x[1]):
        print(f"  {name}: {f1:.4f}")
    best = max(result.values()) if result else 0.0
    print(f"\nBest F1: {best:.4f}")
