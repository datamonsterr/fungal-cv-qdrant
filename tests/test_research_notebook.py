"""Tests for research notebook entry format validation.

Validates that paper-ideas.md entries contain all required fields
so Planner can reliably parse them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REQUIRED_FIELDS = ["URL", "Status", "Methodology", "Fit Assessment", "Proposed Strategy"]
VALID_STATUSES = {"pending", "in-progress", "completed", "rejected"}


def parse_paper_ideas_entries(content: str) -> list[dict[str, str]]:
    """Parse paper-ideas.md content into a list of entry dicts."""
    entries = []
    blocks = re.split(r"^## Paper:", content, flags=re.MULTILINE)
    for block in blocks[1:]:
        entry: dict[str, str] = {}
        lines = block.strip().split("\n")
        entry["title"] = lines[0].strip() if lines else ""
        for field in REQUIRED_FIELDS:
            pattern = rf"\*\*{re.escape(field)}\*\*:\s*(.+)"
            match = re.search(pattern, block)
            entry[field] = match.group(1).strip() if match else ""
        entries.append(entry)
    return entries


VALID_ENTRY = """## Paper: Test Paper Title

- **URL**: https://arxiv.org/abs/2001.00001
- **Status**: pending
- **Methodology**: This paper proposes a contrastive learning approach for fungal image retrieval using triplet loss with hard negative mining.
- **Fit Assessment**: Directly applicable to our fungal species retrieval pipeline as we already have strain-level labels suitable for triplet mining.
- **Proposed Strategy**: Add triplet loss training to the EfficientNetB1 feature extractor with online hard negative mining.
"""

ENTRY_MISSING_FIELD = """## Paper: Incomplete Paper

- **URL**: https://example.com
- **Status**: pending
- **Methodology**: Some methodology here.
"""

ENTRY_INVALID_STATUS = """## Paper: Bad Status Paper

- **URL**: https://example.com
- **Status**: unknown_status
- **Methodology**: Method.
- **Fit Assessment**: Fit.
- **Proposed Strategy**: Strategy.
"""


def test_parse_valid_entry():
    entries = parse_paper_ideas_entries(VALID_ENTRY)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["title"] == "Test Paper Title"
    assert entry["URL"] == "https://arxiv.org/abs/2001.00001"
    assert entry["Status"] == "pending"
    assert len(entry["Methodology"]) > 10
    assert len(entry["Fit Assessment"]) > 5
    assert len(entry["Proposed Strategy"]) > 5


def test_all_required_fields_present():
    entries = parse_paper_ideas_entries(VALID_ENTRY)
    entry = entries[0]
    for field in REQUIRED_FIELDS:
        assert entry[field], f"Required field '{field}' is missing or empty"


def test_status_is_valid():
    entries = parse_paper_ideas_entries(VALID_ENTRY)
    assert entries[0]["Status"] in VALID_STATUSES


def test_missing_field_detected():
    entries = parse_paper_ideas_entries(ENTRY_MISSING_FIELD)
    entry = entries[0]
    missing = [f for f in REQUIRED_FIELDS if not entry.get(f)]
    assert len(missing) > 0, "Expected missing fields to be detected"


def test_invalid_status_detected():
    entries = parse_paper_ideas_entries(ENTRY_INVALID_STATUS)
    entry = entries[0]
    assert entry["Status"] not in VALID_STATUSES


def test_multiple_entries_parsed():
    multi = VALID_ENTRY + "\n" + VALID_ENTRY.replace("Test Paper Title", "Second Paper")
    entries = parse_paper_ideas_entries(multi)
    assert len(entries) == 2


def test_paper_ideas_stub_file_has_correct_format():
    """The stub paper-ideas.md in research/ must be parseable."""
    stub_path = Path(__file__).parent.parent / "research" / "paper-ideas.md"
    if not stub_path.exists():
        pytest.skip("research/paper-ideas.md not found — run setup first")
    content = stub_path.read_text()
    assert "## Format" in content or "paper-ideas" in content.lower()


def test_do_not_repeat_stub_exists():
    stub_path = Path(__file__).parent.parent / "research" / "do-not-repeat.md"
    if not stub_path.exists():
        pytest.skip("research/do-not-repeat.md not found — run setup first")
    content = stub_path.read_text()
    assert "Do Not Repeat" in content or "tried" in content.lower() or "strategy" in content.lower()
