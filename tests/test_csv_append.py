"""Unit tests for lock-safe CSV append utility."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import pytest

from src.autolab.csv_append import append_row, append_staircase_row


def test_append_row_creates_file(tmp_path):
    csv_path = tmp_path / "test.csv"
    append_row(csv_path, ["a", "b", "c"], header=["col1", "col2", "col3"])
    assert csv_path.exists()


def test_append_row_writes_header_once(tmp_path):
    csv_path = tmp_path / "test.csv"
    append_row(csv_path, [1, 2, 3], header=["x", "y", "z"])
    append_row(csv_path, [4, 5, 6], header=["x", "y", "z"])
    rows = list(csv.reader(csv_path.open()))
    assert rows[0] == ["x", "y", "z"]
    assert len([r for r in rows if r]) == 3


def test_append_row_sequential(tmp_path):
    csv_path = tmp_path / "seq.csv"
    for i in range(5):
        append_row(csv_path, [i, f"val{i}"])
    rows = list(csv.reader(csv_path.open()))
    data_rows = [r for r in rows if r]
    assert len(data_rows) == 5
    assert data_rows[0] == ["0", "val0"]
    assert data_rows[4] == ["4", "val4"]


def test_concurrent_csv_writes_no_corruption(tmp_path):
    """Two threads write 50 rows each — no row should be corrupted."""
    csv_path = tmp_path / "concurrent.csv"
    errors: list[str] = []
    n_per_thread = 50

    def write_rows(thread_id: int) -> None:
        for i in range(n_per_thread):
            try:
                append_row(csv_path, [thread_id, i, f"t{thread_id}_r{i}"])
            except Exception as exc:
                errors.append(str(exc))

    t1 = threading.Thread(target=write_rows, args=(0,))
    t2 = threading.Thread(target=write_rows, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Errors during concurrent writes: {errors}"
    rows = [r for r in csv.reader(csv_path.open()) if r]
    assert len(rows) == n_per_thread * 2, f"Expected {n_per_thread * 2} rows, got {len(rows)}"
    for row in rows:
        assert len(row) == 3, f"Corrupted row (wrong column count): {row}"


def test_append_staircase_row(tmp_path):
    csv_path = tmp_path / "autoresearch" / "retrieval.csv"
    result_path = append_staircase_row(
        experiment="retrieval",
        f1_score=0.75,
        strategy_name="test_strategy",
        run_id="test-001",
        results_dir=tmp_path,
    )
    assert result_path.exists()
    rows = list(csv.reader(result_path.open()))
    header = rows[0]
    assert "f1_score" in header
    assert "experiment_index" in header
    data_rows = [r for r in rows[1:] if r]
    assert len(data_rows) == 1
    f1_idx = header.index("f1_score")
    assert float(data_rows[0][f1_idx]) == pytest.approx(0.75)


def test_append_staircase_row_increments_index(tmp_path):
    for i in range(3):
        append_staircase_row(
            experiment="threshold",
            f1_score=0.5 + i * 0.1,
            strategy_name=f"strategy_{i}",
            run_id=f"run-{i:03d}",
            results_dir=tmp_path,
        )

    csv_path = tmp_path / "autoresearch" / "threshold.csv"
    rows = list(csv.reader(csv_path.open()))
    data_rows = [r for r in rows[1:] if r]
    assert len(data_rows) == 3
    indices = [int(r[0]) for r in data_rows]
    assert indices == [0, 1, 2]


def test_concurrent_staircase_writes(tmp_path):
    """Two threads each write 10 staircase rows — total 20 rows, no corruption."""
    errors: list[str] = []
    n_per_thread = 10

    def write_staircase(thread_id: int) -> None:
        for i in range(n_per_thread):
            try:
                append_staircase_row(
                    experiment="retrieval",
                    f1_score=0.5,
                    strategy_name=f"t{thread_id}_s{i}",
                    run_id=f"t{thread_id}-run-{i}",
                    results_dir=tmp_path,
                )
            except Exception as exc:
                errors.append(str(exc))

    t1 = threading.Thread(target=write_staircase, args=(0,))
    t2 = threading.Thread(target=write_staircase, args=(1,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Errors: {errors}"
    csv_path = tmp_path / "autoresearch" / "retrieval.csv"
    rows = list(csv.reader(csv_path.open()))
    data_rows = [r for r in rows[1:] if r]
    assert len(data_rows) == n_per_thread * 2
