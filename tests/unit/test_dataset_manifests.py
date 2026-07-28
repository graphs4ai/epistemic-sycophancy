"""Dataset manifest loading tests (Phase A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_sycophancy.data import (
    count_questions_by_split,
    load_split_manifest,
)

FIXTURE_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "fixtures" / "data" / "split_manifest.csv"
)

# DATA-001: regression counts for the current pinned dataset version.
EXPECTED_SPLIT_COUNTS = {
    "feature_selection": 316,
    "optimization": 237,
    "behavior_validation": 118,
    "holdout_test_behavior": 119,
}
EXPECTED_TOTAL = 790


@pytest.mark.unit
def test_dataset__current_manifest__contains_expected_question_counts() -> None:
    """DATA-001: pinned split manifest matches expected question counts."""
    rows = load_split_manifest(FIXTURE_SPLIT_MANIFEST)
    counts = count_questions_by_split(rows)

    assert counts == EXPECTED_SPLIT_COUNTS
    assert sum(counts.values()) == EXPECTED_TOTAL
