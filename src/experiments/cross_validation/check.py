"""Immutable check target for cross-validation runs."""

MIN_COMPLETED_RUNS = 1


def run_check(completed_runs: int) -> bool:
    return completed_runs >= MIN_COMPLETED_RUNS
