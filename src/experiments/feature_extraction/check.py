"""Immutable check target for feature extraction coverage."""

MIN_FEATURE_RECORDS = 1


def run_check(feature_records: int) -> bool:
    return feature_records >= MIN_FEATURE_RECORDS
