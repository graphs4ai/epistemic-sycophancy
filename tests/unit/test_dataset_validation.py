"""Dataset integrity validation tests (Phase A)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.data import (
    DataIntegrityError,
    assert_belief_variant_ids_are_unique_within_question_and_condition,
    assert_derived_variants_inherit_parent_split,
    assert_mc_targets_are_complete_and_noncontradictory,
    assert_neutral_rows_exactly_one_per_question_order_and_format,
    assert_normalized_question_hash_does_not_cross_splits,
    assert_question_ids_in_exactly_one_split,
    assert_question_macro_weights_sum_to_one_within_component,
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


@pytest.mark.unit
def test_dataset__normalized_question_hash__does_not_cross_splits() -> None:
    """DATA-004: same normalized content hash must not appear under different splits."""
    disjoint = [
        {
            "question_id": "q1",
            "split": "feature_selection",
            "question_text": "Is the sky blue?",
        },
        {
            "question_id": "q2",
            "split": "optimization",
            "question_text": "Do birds fly?",
        },
    ]
    assert_normalized_question_hash_does_not_cross_splits(disjoint)

    leaked = [
        {
            "question_id": "q_a",
            "split": "feature_selection",
            "question_text": "Café!",
        },
        {
            "question_id": "q_b",
            "split": "holdout_test_behavior",
            "question_text": "café",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_normalized_question_hash_does_not_cross_splits(leaked)

    same_split_duplicate_text = [
        {
            "question_id": "q_a",
            "split": "feature_selection",
            "question_text": "Same question?",
        },
        {
            "question_id": "q_b",
            "split": "feature_selection",
            "question_text": "Same question?",
        },
    ]
    assert_normalized_question_hash_does_not_cross_splits(same_split_duplicate_text)


@pytest.mark.unit
def test_dataset__mc_targets__are_complete_and_noncontradictory() -> None:
    """DATA-007: MC0/MC1/MC2 targets exist, differ, and satisfy format cardinalities."""
    valid_rows = [
        {
            "question_id": "q1",
            "format": "MC0",
            "targets": [
                {"answer_id": "ans_t", "text": "Truth", "label": 1},
                {"answer_id": "ans_f", "text": "False", "label": 0},
            ],
        },
        {
            "question_id": "q1",
            "format": "MC1",
            "targets": [
                {"answer_id": "ans_t", "text": "Truth", "label": 1},
                {"answer_id": "ans_f1", "text": "False 1", "label": 0},
                {"answer_id": "ans_f2", "text": "False 2", "label": 0},
            ],
        },
        {
            "question_id": "q1",
            "format": "MC2",
            "targets": [
                {"answer_id": "ans_t1", "text": "Truth 1", "label": 1},
                {"answer_id": "ans_t2", "text": "Truth 2", "label": 1},
                {"answer_id": "ans_f1", "text": "False 1", "label": 0},
            ],
        },
    ]
    assert_mc_targets_are_complete_and_noncontradictory(valid_rows)

    contradictory_mc0 = [
        {
            "question_id": "q2",
            "format": "MC0",
            "targets": [
                {"answer_id": "ans_same", "text": "Same text", "label": 1},
                {"answer_id": "ans_same", "text": "Same text", "label": 0},
            ],
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_mc_targets_are_complete_and_noncontradictory(contradictory_mc0)

    mc0_missing_incorrect = [
        {
            "question_id": "q3",
            "format": "MC0",
            "targets": [
                {"answer_id": "ans_t", "text": "Truth", "label": 1},
            ],
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_mc_targets_are_complete_and_noncontradictory(mc0_missing_incorrect)

    mc1_two_truthful = [
        {
            "question_id": "q4",
            "format": "MC1",
            "targets": [
                {"answer_id": "ans_t1", "text": "Truth 1", "label": 1},
                {"answer_id": "ans_t2", "text": "Truth 2", "label": 1},
                {"answer_id": "ans_f", "text": "False", "label": 0},
            ],
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_mc_targets_are_complete_and_noncontradictory(mc1_two_truthful)

    mc2_no_false = [
        {
            "question_id": "q5",
            "format": "MC2",
            "targets": [
                {"answer_id": "ans_t1", "text": "Truth 1", "label": 1},
                {"answer_id": "ans_t2", "text": "Truth 2", "label": 1},
            ],
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_mc_targets_are_complete_and_noncontradictory(mc2_no_false)


@pytest.mark.unit
def test_dataset__belief_variant_ids__are_unique_within_question_and_condition() -> None:
    """DATA-006: belief_variant_id unique per (question, condition); CB↔IB needs provenance."""
    unique_rows = [
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "b1",
        },
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "b2",
        },
        {
            "question_id": "q1",
            "belief_condition": "IB",
            "belief_variant_id": "b3",
        },
        {
            "question_id": "q1",
            "belief_condition": "N",
            "belief_variant_id": None,
        },
        {
            "question_id": "q2",
            "belief_condition": "CB",
            "belief_variant_id": "b1",
        },
    ]
    assert_belief_variant_ids_are_unique_within_question_and_condition(unique_rows)

    duplicate_within_condition = [
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "dup",
        },
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "dup",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_belief_variant_ids_are_unique_within_question_and_condition(
            duplicate_within_condition
        )

    cross_polarity_without_provenance = [
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "shared",
        },
        {
            "question_id": "q1",
            "belief_condition": "IB",
            "belief_variant_id": "shared",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_belief_variant_ids_are_unique_within_question_and_condition(
            cross_polarity_without_provenance
        )

    cross_polarity_with_provenance = [
        {
            "question_id": "q1",
            "belief_condition": "CB",
            "belief_variant_id": "shared",
        },
        {
            "question_id": "q1",
            "belief_condition": "IB",
            "belief_variant_id": "shared",
        },
    ]
    shared_provenance = [
        {"question_id": "q1", "belief_variant_id": "shared"},
    ]
    assert_belief_variant_ids_are_unique_within_question_and_condition(
        cross_polarity_with_provenance,
        shared_provenance=shared_provenance,
    )


@pytest.mark.unit
def test_dataset__neutral_rows__exactly_one_per_question_order_and_format() -> None:
    """DATA-005: one distinct neutral_prompt_hash per (qid, CF/IF, format)."""
    valid_rows = [
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "N",
            "answer_order": "true-first",
            "neutral_prompt_hash": "hash_cf",
        },
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "N",
            "answer_order": "true-first",
            "neutral_prompt_hash": "hash_cf",
        },
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "CB",
            "answer_order": "true-first",
            "belief_variant_id": "b1",
            "neutral_prompt_hash": "hash_cf",
        },
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "N",
            "order_regime": "IF",
            "neutral_prompt_hash": "hash_if",
        },
        {
            "question_id": "q2",
            "format": "MC0",
            "belief_condition": "N",
            "answer_order": "false-first",
            "neutral_prompt_hash": "hash_q2",
        },
    ]
    assert_neutral_rows_exactly_one_per_question_order_and_format(valid_rows)

    two_distinct_hashes = [
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "N",
            "answer_order": "true-first",
            "neutral_prompt_hash": "hash_a",
        },
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "N",
            "answer_order": "true-first",
            "neutral_prompt_hash": "hash_b",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_neutral_rows_exactly_one_per_question_order_and_format(
            two_distinct_hashes
        )

    missing_neutral_for_present_key = [
        {
            "question_id": "q1",
            "format": "MC0",
            "belief_condition": "CB",
            "answer_order": "true-first",
            "belief_variant_id": "b1",
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_neutral_rows_exactly_one_per_question_order_and_format(
            missing_neutral_for_present_key
        )


@pytest.mark.unit
def test_dataset__question_macro_weights__sum_to_one_within_component() -> None:
    """DATA-008: for every question, sum_b pair_weight == 1 within the component."""
    valid_component = [
        {"question_id": "q1", "belief_pair_id": "p1", "pair_weight": 0.5},
        {"question_id": "q1", "belief_pair_id": "p2", "pair_weight": 0.5},
        {"question_id": "q2", "belief_pair_id": "p3", "pair_weight": 0.25},
        {"question_id": "q2", "belief_pair_id": "p4", "pair_weight": 0.75},
    ]
    assert_question_macro_weights_sum_to_one_within_component(valid_component)

    unequal_but_normalized = [
        {"question_id": "q1", "belief_pair_id": "p1", "pair_weight": 0.2},
        {"question_id": "q1", "belief_pair_id": "p2", "pair_weight": 0.3},
        {"question_id": "q1", "belief_pair_id": "p3", "pair_weight": 0.5},
    ]
    assert_question_macro_weights_sum_to_one_within_component(unequal_but_normalized)

    bad_sum = [
        {"question_id": "q1", "belief_pair_id": "p1", "pair_weight": 0.4},
        {"question_id": "q1", "belief_pair_id": "p2", "pair_weight": 0.4},
    ]
    with pytest.raises(DataIntegrityError):
        assert_question_macro_weights_sum_to_one_within_component(bad_sum)

    # condition_weight must not be treated as w_{q,b}; these rows would "pass"
    # only if the wrong field were summed.
    wrong_field_would_look_normalized = [
        {
            "question_id": "q1",
            "belief_pair_id": "p1",
            "pair_weight": 0.4,
            "condition_weight": 0.5,
        },
        {
            "question_id": "q1",
            "belief_pair_id": "p2",
            "pair_weight": 0.4,
            "condition_weight": 0.5,
        },
    ]
    with pytest.raises(DataIntegrityError):
        assert_question_macro_weights_sum_to_one_within_component(
            wrong_field_would_look_normalized
        )


@pytest.mark.unit
def test_pipeline__optimization_split__does_not_load_mc1_or_mc2_rows() -> None:
    """MC-007: optimization split must be MC0-only; val/holdout may load all formats."""
    from epistemic_sycophancy.data.validation import (
        assert_optimization_split_is_mc0_only,
    )

    ok = [
        {"question_id": "q1", "split": "optimization", "format": "MC0"},
        {"question_id": "q2", "split": "behavior_validation", "format": "MC1"},
        {"question_id": "q3", "split": "holdout_test_behavior", "format": "MC2"},
    ]
    assert_optimization_split_is_mc0_only(ok)

    bad = [
        {"question_id": "q1", "split": "optimization", "format": "MC1"},
        {"question_id": "q2", "split": "optimization", "format": "MC0"},
    ]
    with pytest.raises(DataIntegrityError):
        assert_optimization_split_is_mc0_only(bad)

    bad_mc2 = [
        {"question_id": "q1", "split": "optimization", "format": "MC2"},
    ]
    with pytest.raises(DataIntegrityError):
        assert_optimization_split_is_mc0_only(bad_mc2)
