"""Tests for Planner queue logic.

Validates that Planner correctly selects only pending hypotheses,
skips in-progress/completed ones, and marks selected as in-progress.
"""

from __future__ import annotations

import re
from typing import Literal


Status = Literal["pending", "in-progress", "completed", "rejected"]


def parse_hypotheses(content: str) -> list[dict[str, str]]:
    """Parse paper-ideas.md entries into hypothesis dicts."""
    entries = []
    blocks = re.split(r"^## Paper:", content, flags=re.MULTILINE)
    for block in blocks[1:]:
        entry: dict[str, str] = {}
        lines = block.strip().split("\n")
        entry["title"] = lines[0].strip() if lines else ""
        for field in ["URL", "Status", "Proposed Strategy"]:
            match = re.search(rf"\*\*{re.escape(field)}\*\*:\s*(.+)", block)
            entry[field] = match.group(1).strip() if match else ""
        entries.append(entry)
    return entries


def select_pending(hypotheses: list[dict[str, str]]) -> list[dict[str, str]]:
    """Planner logic: return only pending hypotheses."""
    return [h for h in hypotheses if h.get("Status") == "pending"]


def mark_in_progress(content: str, title: str) -> str:
    """Update status of one hypothesis from pending to in-progress."""
    return re.sub(
        rf"(## Paper: {re.escape(title)}.*?\*\*Status\*\*: )pending",
        r"\1in-progress",
        content,
        flags=re.DOTALL,
    )


MOCK_PAPER_IDEAS = """## Paper: Triplet Loss Fungal Retrieval

- **URL**: https://arxiv.org/abs/2001.00001
- **Status**: pending
- **Methodology**: Triplet loss with hard negative mining.
- **Fit Assessment**: Applicable to fungal retrieval.
- **Proposed Strategy**: Add triplet loss training.

## Paper: Already Running Experiment

- **URL**: https://arxiv.org/abs/2001.00002
- **Status**: in-progress
- **Methodology**: Contrastive learning approach.
- **Fit Assessment**: Maybe applicable.
- **Proposed Strategy**: Use contrastive loss.

## Paper: Completed Baseline

- **URL**: https://arxiv.org/abs/2001.00003
- **Status**: completed
- **Methodology**: Simple cosine similarity.
- **Fit Assessment**: Already done.
- **Proposed Strategy**: cosine_top5.

## Paper: Rejected Approach

- **URL**: heuristic
- **Status**: rejected
- **Methodology**: Random feature selection.
- **Fit Assessment**: Does not apply.
- **Proposed Strategy**: random_features.

## Paper: Another Pending Idea

- **URL**: https://arxiv.org/abs/2001.00004
- **Status**: pending
- **Methodology**: Attention-based aggregation.
- **Fit Assessment**: Could improve strain-level aggregation.
- **Proposed Strategy**: attention_aggregation.
"""


def test_planner_selects_only_pending():
    hypotheses = parse_hypotheses(MOCK_PAPER_IDEAS)
    pending = select_pending(hypotheses)
    assert len(pending) == 2
    for h in pending:
        assert h["Status"] == "pending"


def test_planner_skips_in_progress():
    hypotheses = parse_hypotheses(MOCK_PAPER_IDEAS)
    pending = select_pending(hypotheses)
    titles = [h["title"] for h in pending]
    assert "Already Running Experiment" not in titles


def test_planner_skips_completed():
    hypotheses = parse_hypotheses(MOCK_PAPER_IDEAS)
    pending = select_pending(hypotheses)
    titles = [h["title"] for h in pending]
    assert "Completed Baseline" not in titles


def test_planner_skips_rejected():
    hypotheses = parse_hypotheses(MOCK_PAPER_IDEAS)
    pending = select_pending(hypotheses)
    titles = [h["title"] for h in pending]
    assert "Rejected Approach" not in titles


def test_mark_in_progress_updates_status():
    updated = mark_in_progress(MOCK_PAPER_IDEAS, "Triplet Loss Fungal Retrieval")
    hypotheses = parse_hypotheses(updated)
    triplet = next(h for h in hypotheses if "Triplet" in h["title"])
    assert triplet["Status"] == "in-progress"


def test_second_planner_invocation_skips_already_queued():
    """After marking in-progress, second planner run skips it."""
    updated = mark_in_progress(MOCK_PAPER_IDEAS, "Triplet Loss Fungal Retrieval")
    hypotheses = parse_hypotheses(updated)
    pending = select_pending(hypotheses)
    assert len(pending) == 1
    assert pending[0]["title"] == "Another Pending Idea"


def test_empty_queue_returns_nothing():
    content = """## Paper: Done

- **URL**: https://example.com
- **Status**: completed
- **Methodology**: done.
- **Fit Assessment**: done.
- **Proposed Strategy**: done.
"""
    hypotheses = parse_hypotheses(content)
    pending = select_pending(hypotheses)
    assert pending == []
