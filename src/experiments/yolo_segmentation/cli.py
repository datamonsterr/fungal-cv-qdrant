"""CLI wrapper for yolo_segmentation experiment."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="yolo_segmentation experiment CLI")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args()

    try:
        from src.experiments.yolo_segmentation.run import ExperimentParams, run as _run
    except ImportError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)

    params = ExperimentParams(run_id=args.run_id, output_root=args.output_root, description=args.description)
    try:
        result = _run(params)
    except Exception as exc:
        print(f"Experiment failure: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({"f1_score": result.f1_score, "strategy_name": result.strategy_name, "artifact_paths": result.artifact_paths, "run_id": result.run_id}, indent=2))


if __name__ == "__main__":
    main()
