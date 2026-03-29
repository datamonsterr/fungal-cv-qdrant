"""Validation checks for reorganized ensemble/retrieval modules."""

from src.analysis.retrieval import (
    batch_visualize_complementary,
    compare_ensemble_strategies,
    ensemble_report,
    visualize_complementary_cases,
)
from src.experiments.retrieval import ensemble_analysis


def run() -> None:
    """Run lightweight import/symbol checks for reorganized modules."""
    assert hasattr(ensemble_analysis, "main")
    assert hasattr(ensemble_report, "main")
    assert hasattr(compare_ensemble_strategies, "compare_strategies")
    assert hasattr(batch_visualize_complementary, "main")
    assert hasattr(visualize_complementary_cases, "main")
    print("ensemble_reorg_check: PASS")


if __name__ == "__main__":
    run()
