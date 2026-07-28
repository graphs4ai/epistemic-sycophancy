"""Dataset integrity validation tests (Phase A)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.data import (
    DataIntegrityError,
    assert_derived_variants_inherit_parent_split,
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


@pytest.mark.unit
def test_dataset__derived_variants__inherit_parent_split() -> None:
    """DATA-003: every derived row's split equals its parent question_id split."""
    parents = [
        {"question_id": "q1", "split": "feature_selection"},
        {"question_id": "q2", "split": "optimization"},
    ]
    matching_derived = [
        {
            "question_id": "q1",
            "split": "feature_selection",
            "belief_condition": "N",
            "order_regime": "CF",
            "format": "MC0",
        },
        {
            "question_id": "q1",
            "split": "feature_selection",
            "belief_condition": "CB",
            "order_regime": "IF",
            "format": "MC1",
            "belief_variant_id": "cb_a",
        },
        {
            "question_id": "q1",
            "split": "feature_selection",
            "belief_condition": "IB",
            "order_regime": "CF",
            "format": "MC2",
            "belief_variant_id": "ib_a",
        },
        {
            "question_id": "q2",
            "split": "optimization",
            "belief_condition": "CB",
            "order_regime": "IF",
            "format": "MC0",
            "belief_variant_id": "cb_b",
        },
    ]
    assert_derived_variants_inherit_parent_split(parents, matching_derived)

    mismatched = [
        {
            "question_id": "q1",
            "split": "holdout_test_behavior",
            "belief_condition": "IB",
            "order_regime": "CF",
            "format": "MC0",
            "belief_variant_id": "ib_leak",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_derived_variants_inherit_parent_split(parents, mismatched)
