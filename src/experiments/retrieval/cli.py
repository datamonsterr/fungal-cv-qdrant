"""CLI wrapper for retrieval experiment.

Usage:
    uv run python -m src.experiments.retrieval.cli \
        --run-id <run_id> \
        --output-root <path> \
        --description "<description>"

Exit codes: 0 success, 1 experiment failure, 2 config error.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval experiment CLI")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--output-root", required=True, help="Output root directory")
    parser.add_argument("--description", default="", help="Strategy description")
    args = parser.parse_args()

    try:
        from src.experiments.retrieval.run import ExperimentParams, run
    except ImportError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)

    params = ExperimentParams(
        run_id=args.run_id,
        output_root=args.output_root,
        description=args.description,
    )

    try:
        result = run(params)
    except Exception as exc:
        print(f"Experiment failure: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "f1_score": result.f1_score,
        "strategy_name": result.strategy_name,
        "artifact_paths": result.artifact_paths,
        "run_id": result.run_id,
    }, indent=2))


if __name__ == "__main__":
    main()
