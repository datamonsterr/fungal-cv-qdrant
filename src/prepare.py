"""
Experiment Prepare and Check
============================
Combines prerequisite checks with experiment execution.

Usage:
    uv run python src/prepare.py --experiment segmentation
    uv run python src/prepare.py --experiment feature-extractor --skip-checks

This script:
    1. Runs prerequisite checks (Qdrant, dataset, metadata)
    2. Runs the experiment via run.py
    3. Validates the result
    4. Reports pass/fail
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from src.config import (
    QDRANT_API_KEY,
    QDRANT_URL,
    SEGMENTED_METADATA_PATH,
    STRAIN_SPECIES_MAPPING_PATH,
)
from src.prepare.checks import check_dataset_root, check_metadata_exists

# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------



def check_qdrant() -> tuple[bool, str]:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()
        return True, f"Qdrant reachable at {QDRANT_URL}"
    except Exception as exc:
        return False, f"Qdrant connection failed: {exc}"


def check_experiment_requirements(experiment: str) -> List[tuple[bool, str]]:
    """
    Run all prerequisite checks for the given experiment.
    Returns list of (passed, message).
    """
    results: List[tuple[bool, str]] = []

    # Common checks: dataset and Qdrant
    results.append(check_dataset_root())
    results.append(check_metadata_exists())
    results.append(check_qdrant())

    # Experiment-specific checks
    if experiment in ("segmentation",):
        # Segmentation needs original images and segmented images
        from src.config import ORIGINAL_DATASET_PATH

        results.append(check_dataset_root(ORIGINAL_DATASET_PATH))

    if experiment in ("feature-extractor", "segmentation"):
        # Needs strain mapping
        if not STRAIN_SPECIES_MAPPING_PATH.exists():
            results.append(
                (False, f"Missing strain mapping: {STRAIN_SPECIES_MAPPING_PATH}")
            )
        else:
            results.append((True, f"Strain mapping OK: {STRAIN_SPECIES_MAPPING_PATH}"))

        # Needs segmented metadata
        if not SEGMENTED_METADATA_PATH.exists():
            results.append(
                (False, f"Missing segmented metadata: {SEGMENTED_METADATA_PATH}")
            )
        else:
            results.append((True, f"Segmented metadata OK: {SEGMENTED_METADATA_PATH}"))

    return results


def run_all_checks(experiment: str) -> bool:
    """Run all checks and print results. Returns True if all pass."""
    print(f"\n{'='*50}")
    print(f"Prerequisite checks for: {experiment}")
    print(f"{'='*50}")

    checks = check_experiment_requirements(experiment)
    all_passed = True
    for passed, msg in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "OK" if passed else "XX"
        print(f"  [{symbol}] {status}: {msg}")
        if not passed:
            all_passed = False

    print(f"\nResult: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    return all_passed


# ---------------------------------------------------------------------------
# Prepare + run
# ---------------------------------------------------------------------------


def run_prepare(
    experiment: str,
    skip_checks: bool = False,
    run_py_extra_args: Optional[List[str]] = None,
) -> None:
    """
    Run prerequisite checks, then execute the experiment via run.py.

    run_py_extra_args: additional args passed to run.py (e.g. ["--k", "7"])
    """
    if not skip_checks:
        checks_passed = run_all_checks(experiment)
        if not checks_passed:
            print("\nPrerequisites not met. Run with --skip-checks to bypass.")
            print("To fix, run the prepare pipeline first:")
            print(
                "  uv run python -m src.prepare.init --collection myco_fungi_features_full_finetuned"
            )
            sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Running experiment: {experiment}")
    print(f"{'='*50}\n")

    # Build run.py command
    run_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run.py"),
        "--experiment",
        experiment,
        "--no-plot",  # prepare.py skips plot to keep output clean
    ]
    if run_py_extra_args:
        run_cmd.extend(run_py_extra_args)

    result = subprocess.run(run_cmd)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and run an experiment: check prerequisites then execute."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment name (see src/run.py --experiment-list for available)",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip prerequisite checks",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override k value passed to run.py",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Override aggregation strategy passed to run.py",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Override number of CV folds passed to run.py",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description passed to run.py",
    )

    args = parser.parse_args(argv)

    extra_args: List[str] = []
    if args.description:
        extra_args += ["--description", args.description]
    if args.k is not None:
        extra_args += ["--k", str(args.k)]
    if args.strategy is not None:
        extra_args += ["--strategy", args.strategy]
    if args.n_folds is not None:
        extra_args += ["--n-folds", str(args.n_folds)]

    run_prepare(
        experiment=args.experiment,
        skip_checks=args.skip_checks,
        run_py_extra_args=extra_args,
    )


if __name__ == "__main__":
    main()
