"""Immutable check target for preprocessing quality."""

MIN_SEGMENTED_IMAGES = 1


def run_check(segmented_count: int) -> bool:
    return segmented_count >= MIN_SEGMENTED_IMAGES