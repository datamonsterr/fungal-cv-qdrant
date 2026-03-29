"""Immutable check target for retrieval experiment artifacts."""

MIN_RETRIEVAL_RESULTS = 1


def run_check(result_records: int) -> bool:
    return result_records >= MIN_RETRIEVAL_RESULTS
