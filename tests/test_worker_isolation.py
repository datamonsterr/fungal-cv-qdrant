"""Tests for Worker-style isolated experiment runs.

Validates that two concurrent 'Worker-style' runs with different run_ids
produce non-overlapping output directories and do not corrupt shared CSV.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest

from src.autolab.csv_append import append_staircase_row
from src.experiments.feature_extraction.run import ExperimentParams, run as run_feature_extraction


def _simulate_worker_run(
    run_id: str,
    output_base: Path,
    results_dir: Path,
    errors: list[str],
) -> None:
    """Simulate a Worker agent run using the experiment contract."""
    try:
        params = ExperimentParams(
            run_id=run_id,
            output_root=str(output_base / run_id),
            description=f"isolation test {run_id}",
        )
        result = run_feature_extraction(params)

        append_staircase_row(
            experiment="feature_extraction",
            f1_score=result.f1_score,
            strategy_name=result.strategy_name,
            run_id=result.run_id,
            results_dir=results_dir,
        )
    except Exception as exc:
        errors.append(f"{run_id}: {exc}")


def test_two_workers_non_overlapping_output(tmp_path):
    """Two Worker runs with different run_ids produce separate output dirs."""
    run_id_1 = "worker-isolation-001"
    run_id_2 = "worker-isolation-002"
    errors: list[str] = []

    t1 = threading.Thread(
        target=_simulate_worker_run,
        args=(run_id_1, tmp_path / "runs", tmp_path, errors),
    )
    t2 = threading.Thread(
        target=_simulate_worker_run,
        args=(run_id_2, tmp_path / "runs", tmp_path, errors),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Worker errors: {errors}"

    out1 = tmp_path / "runs" / run_id_1
    out2 = tmp_path / "runs" / run_id_2
    assert out1.exists(), f"Output dir for {run_id_1} not found"
    assert out2.exists(), f"Output dir for {run_id_2} not found"
    assert out1 != out2, "Both workers wrote to the same directory"


def test_two_workers_shared_csv_has_two_rows(tmp_path):
    """Two concurrent Workers both append to shared CSV — 2 rows total."""
    run_id_1 = "worker-csv-001"
    run_id_2 = "worker-csv-002"
    errors: list[str] = []

    t1 = threading.Thread(
        target=_simulate_worker_run,
        args=(run_id_1, tmp_path / "runs", tmp_path, errors),
    )
    t2 = threading.Thread(
        target=_simulate_worker_run,
        args=(run_id_2, tmp_path / "runs", tmp_path, errors),
    )
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Worker errors: {errors}"

    csv_path = tmp_path / "autoresearch" / "feature_extraction.csv"
    assert csv_path.exists(), "Shared CSV not created"
    rows = list(csv.reader(csv_path.open()))
    data_rows = [r for r in rows if r and r[0] != "experiment_index"]
    assert len(data_rows) == 2, f"Expected 2 data rows, got {len(data_rows)}: {data_rows}"


def test_results_json_scoped_to_output_root(tmp_path):
    """Each Worker's results.json must be inside its own output_root."""
    params = ExperimentParams(
        run_id="scope-test-001",
        output_root=str(tmp_path / "scope-test-001"),
        description="scope test",
    )
    result = run_feature_extraction(params)

    for artifact_path in result.artifact_paths:
        artifact = Path(artifact_path)
        assert str(artifact).startswith(str(tmp_path / "scope-test-001")), (
            f"Artifact {artifact} is outside output_root"
        )


def test_worker_run_id_echoed_in_result(tmp_path):
    """ExperimentResult.run_id must match ExperimentParams.run_id."""
    run_id = "echo-test-001"
    params = ExperimentParams(
        run_id=run_id,
        output_root=str(tmp_path / run_id),
        description="echo test",
    )
    result = run_feature_extraction(params)
    assert result.run_id == run_id
