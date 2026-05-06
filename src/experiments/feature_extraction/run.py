"""Experiment contract interface for feature_extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentParams:
    run_id: str
    output_root: str
    description: str


@dataclass
class ExperimentResult:
    f1_score: float
    strategy_name: str
    artifact_paths: List[str]
    run_id: str


def run(params: ExperimentParams) -> ExperimentResult:
    """Uniform experiment contract wrapper. Scoped to params.output_root."""
    import json
    from pathlib import Path

    output_root = Path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy = params.description[:30] if params.description else "feature_extraction"
    result_data = {
        "f1_score": 0.0,
        "strategy_name": strategy,
        "artifact_paths": [],
        "run_id": params.run_id,
    }
    (output_root / "results.json").write_text(json.dumps(result_data, indent=2))
    return ExperimentResult(
        f1_score=0.0,
        strategy_name=strategy,
        artifact_paths=[str(output_root / "results.json")],
        run_id=params.run_id,
    )
