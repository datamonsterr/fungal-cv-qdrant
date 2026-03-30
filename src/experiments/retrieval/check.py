"""Validation checks for retrieval experiment outputs and module layout."""

from src.analysis.retrieval import (
    batch_visualize_complementary,
    compare_ensemble_strategies,
    ensemble_report,
    visualize_complementary_cases,
)
from src.experiments.retrieval import ensemble_analysis

MIN_RETRIEVAL_RESULTS = 1


def run_check(result_records: int) -> bool:
    """Return True when retrieval result count satisfies the minimum target."""
    return result_records >= MIN_RETRIEVAL_RESULTS


def run_reorg_check() -> None:
    """Run lightweight import/symbol checks for reorganized retrieval modules."""
    assert hasattr(ensemble_analysis, "main")
    assert hasattr(ensemble_report, "main")
    assert hasattr(compare_ensemble_strategies, "compare_strategies")
    assert hasattr(batch_visualize_complementary, "main")
    assert hasattr(visualize_complementary_cases, "main")
    print("retrieval_check: PASS")


if __name__ == "__main__":
    run_reorg_check()