"""Lock-safe CSV append utility for concurrent Worker agents.

Uses fcntl.flock (Linux/macOS) to ensure atomic row writes to shared CSV files.
Multiple Workers can call append_row() simultaneously without data corruption.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Sequence


def append_row(csv_path: str | Path, row: Sequence[Any], header: Sequence[str] | None = None) -> None:
    """Append one row to a CSV file with exclusive file lock.

    Args:
        csv_path: Path to the CSV file. Created if absent.
        row: Values to append as a new row.
        header: Column names to write if the file is newly created.

    The lock is held only during the open+write+flush cycle.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            size = fd.seek(0, 2)
            write_header = size == 0
            writer = csv.writer(fd)
            if write_header and header:
                writer.writerow(header)
            writer.writerow(row)
            fd.flush()
            os.fsync(fd.fileno())
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)


def append_staircase_row(
    experiment: str,
    f1_score: float,
    strategy_name: str,
    run_id: str,
    results_dir: str | Path | None = None,
) -> Path:
    """Append one row to the shared staircase CSV for an experiment.

    CSV location: <results_dir>/autoresearch/<experiment>.csv
    Format: experiment_index,f1_score,strategy_name,run_id,timestamp

    Returns the path of the CSV file written.
    """
    if results_dir is None:
        from src.config import RESULTS_DIR
        results_dir = RESULTS_DIR

    csv_path = Path(results_dir) / "autoresearch" / f"{experiment}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(csv_path, "a+", newline="") as fd:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
        try:
            fd.seek(0)
            rows = list(csv.reader(fd))
            data_rows = [r for r in rows if r and r[0] != "experiment_index"]
            experiment_index = len(data_rows)
            fd.seek(0, os.SEEK_END)
            writer = csv.writer(fd)
            if not rows:
                writer.writerow(["experiment_index", "f1_score", "strategy_name", "run_id", "timestamp"])
            writer.writerow([experiment_index, f1_score, strategy_name, run_id, timestamp])
            fd.flush()
            os.fsync(fd.fileno())
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one staircase CSV row")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--f1-score", required=True, type=float)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    csv_path = append_staircase_row(
        experiment=args.experiment,
        f1_score=args.f1_score,
        strategy_name=args.strategy_name,
        run_id=args.run_id,
    )
    print(json.dumps({"csv_path": str(csv_path)}))


if __name__ == "__main__":
    main()
