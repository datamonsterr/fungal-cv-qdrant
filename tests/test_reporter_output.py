"""Tests for Reporter output format and best-F1 extraction."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional


def parse_staircase_csv(content: str) -> list[dict[str, str]]:
    """Parse staircase CSV content into list of row dicts."""
    reader = csv.DictReader(io.StringIO(content))
    return [row for row in reader]


def find_best_f1(rows: list[dict[str, str]]) -> tuple[float, Optional[str], Optional[int]]:
    """Return (best_f1, strategy_name, experiment_index) from CSV rows."""
    if not rows:
        return 0.0, None, None
    best_row = max(rows, key=lambda r: float(r.get("f1_score", 0.0)))
    return (
        float(best_row["f1_score"]),
        best_row.get("strategy_name"),
        int(best_row.get("experiment_index", -1)),
    )


def format_status_block(
    experiment: str,
    best_f1: float,
    best_strategy: Optional[str],
    experiment_index: Optional[int],
    staircase_chart_path: str,
    active_workers: int = 0,
    completed_this_session: int = 0,
) -> str:
    """Format a Reporter status block."""
    lines = [
        f"## Experiment Status",
        f"",
        f"**Best F1**: {best_f1:.3f} (experiment #{experiment_index}, strategy: {best_strategy})",
        f"**Active workers**: {active_workers}",
        f"**Completed this session**: {completed_this_session} runs",
        f"**Staircase chart**: {staircase_chart_path}",
    ]
    return "\n".join(lines)


MOCK_CSV_3_ROWS = """experiment_index,f1_score,strategy_name,run_id,timestamp
0,0.650,cosine_top5,run-001,2026-05-05T10:00:00Z
1,0.720,weighted_aggregation,run-002,2026-05-05T10:05:00Z
2,0.810,triplet_loss_top3,run-003,2026-05-05T10:10:00Z
"""

MOCK_CSV_SINGLE_ROW = """experiment_index,f1_score,strategy_name,run_id,timestamp
0,0.550,baseline,run-000,2026-05-05T09:00:00Z
"""

MOCK_CSV_EMPTY = """experiment_index,f1_score,strategy_name,run_id,timestamp
"""


def test_parse_staircase_csv():
    rows = parse_staircase_csv(MOCK_CSV_3_ROWS)
    assert len(rows) == 3


def test_find_best_f1_from_3_rows():
    rows = parse_staircase_csv(MOCK_CSV_3_ROWS)
    best_f1, strategy, idx = find_best_f1(rows)
    assert abs(best_f1 - 0.810) < 1e-6
    assert strategy == "triplet_loss_top3"
    assert idx == 2


def test_find_best_f1_single_row():
    rows = parse_staircase_csv(MOCK_CSV_SINGLE_ROW)
    best_f1, strategy, idx = find_best_f1(rows)
    assert abs(best_f1 - 0.550) < 1e-6
    assert strategy == "baseline"


def test_find_best_f1_empty_csv():
    rows = parse_staircase_csv(MOCK_CSV_EMPTY)
    best_f1, strategy, idx = find_best_f1(rows)
    assert best_f1 == 0.0
    assert strategy is None
    assert idx is None


def test_status_block_contains_required_fields():
    rows = parse_staircase_csv(MOCK_CSV_3_ROWS)
    best_f1, strategy, idx = find_best_f1(rows)
    block = format_status_block(
        experiment="retrieval",
        best_f1=best_f1,
        best_strategy=strategy,
        experiment_index=idx,
        staircase_chart_path="results/autoresearch/retrieval.png",
    )
    assert "Best F1" in block
    assert "0.810" in block
    assert "triplet_loss_top3" in block
    assert "results/autoresearch/retrieval.png" in block
    assert "Active workers" in block


def test_status_block_format():
    block = format_status_block(
        experiment="retrieval",
        best_f1=0.847,
        best_strategy="cosine_top5_retrieval",
        experiment_index=12,
        staircase_chart_path="results/autoresearch/retrieval.png",
        active_workers=1,
        completed_this_session=3,
    )
    assert "## Experiment Status" in block
    assert "experiment #12" in block
    assert "cosine_top5_retrieval" in block
    assert "1" in block


def test_reporter_graceful_with_no_rows():
    """Reporter should not crash when CSV has no data rows."""
    rows = parse_staircase_csv(MOCK_CSV_EMPTY)
    best_f1, strategy, idx = find_best_f1(rows)
    block = format_status_block(
        experiment="retrieval",
        best_f1=best_f1,
        best_strategy=strategy or "none",
        experiment_index=idx or 0,
        staircase_chart_path="results/autoresearch/retrieval.png",
    )
    assert "0.000" in block or "Best F1" in block
