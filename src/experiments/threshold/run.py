"""
Threshold experiment — run_accuracy()

Returns the best F1 score across all threshold strategies.
Called by src/run.py when --experiment threshold is used.

Returns:
    For strategy="all" or strategy="best", dict of {f"{strategy}_{algo}": f1}.
    For a specific strategy name, returns that strategy's F1 as float.
    Falls back to best float only when the requested strategy is missing.

Direct usage:
    uv run python -m src.experiments.threshold.run
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_accuracy(strategy: str = "best", **kwargs) -> float | Dict[str, float]:
    """
    Run threshold analysis on the diverse dataset and return F1 score(s).

    Assumes results/threshold/diverse_retrieval_results.csv exists
    (run retrieve_diverse first if not).

    Args:
        strategy: "best" = return dict of all strategies and print best strategy
                  "all"  = return dict of all {name: f1}
                  or a specific strategy name to return its F1
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
    if strategy in flat:
        return flat[strategy]
    return max(flat.values()) if flat else 0.0


if __name__ == "__main__":
    result = run_accuracy(strategy="all")
    if isinstance(result, dict):
        print("\nAll strategy-algorithm F1 scores:")
        for name, f1 in sorted(result.items(), key=lambda x: -x[1]):
            print(f"  {name}: {f1:.4f}")
        best = max(result.values()) if result else 0.0
        print(f"\nBest F1: {best:.4f}")
    else:
        print(f"\nBest F1: {result:.4f}")


__all__ = ["ExperimentParams", "ExperimentResult", "run", "run_accuracy"]


def _main_contract() -> None:
    pass


@dataclass
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@dataclass
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: list
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    """Execute one threshold experiment run scoped to params.output_root."""
    import json
    from pathlib import Path as _Path

    output_root = _Path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "log"
    log_dir.mkdir(exist_ok=True)

    try:
        all_results = run_accuracy(strategy="all")
        if isinstance(all_results, dict) and all_results:
            best_key = max(all_results.keys(), key=lambda k: all_results[k])
            f1 = float(all_results[best_key])
            strategy_name = best_key[:30]
        else:
            f1 = float(all_results) if isinstance(all_results, (int, float)) else 0.0
            strategy_name = params.description[:30] if params.description else "threshold"
        artifact_paths: list = []
    except Exception as exc:
        error_log = log_dir / "run.log"
        error_log.write_text(str(exc))
        raise

    result_data = {
        "f1_score": f1,
        "strategy_name": strategy_name,
        "artifact_paths": artifact_paths,
        "run_id": params.run_id,
    }
    results_json = output_root / "results.json"
    results_json.write_text(json.dumps(result_data, indent=2))
    artifact_paths = artifact_paths + [str(results_json)]

    return ExperimentResult(
        f1_score=f1,
        strategy_name=strategy_name,
        artifact_paths=artifact_paths,
        run_id=params.run_id,
    )
