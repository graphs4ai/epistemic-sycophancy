"""Dataset integrity validation tests (Phase A)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.data import (
    DataIntegrityError,
    assert_question_ids_in_exactly_one_split,
)


@pytest.mark.unit
def test_dataset__question_id__appears_in_exactly_one_split() -> None:
    """DATA-002: for every question_id, n_unique(split) == 1."""
    disjoint = [
        {"question_id": "q1", "split": "feature_selection"},
        {"question_id": "q2", "split": "optimization"},
        {"question_id": "q3", "split": "behavior_validation"},
        {"question_id": "q4", "split": "holdout_test_behavior"},
    ]
    assert_question_ids_in_exactly_one_split(disjoint)

    same_split_repeated = [
        {"question_id": "q1", "split": "feature_selection"},
        {"question_id": "q1", "split": "feature_selection"},
        {"question_id": "q2", "split": "optimization"},
    ]
    assert_question_ids_in_exactly_one_split(same_split_repeated)

    leaked = [
        {"question_id": "q1", "split": "feature_selection"},
        {"question_id": "q1", "split": "holdout_test_behavior"},
        {"question_id": "q2", "split": "optimization"},
    ]
    with pytest.raises(DataIntegrityError):
        assert_question_ids_in_exactly_one_split(leaked)
