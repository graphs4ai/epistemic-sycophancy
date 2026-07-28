"""Answer-order mapping tests (Phase B)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.prompts.ordering import assign_order


@pytest.mark.unit
def test_ordering__correct_first__maps_truth_to_A() -> None:
    """PROMPT-002: CF maps truthful candidate to A."""
    assignment = assign_order(
        order_regime="CF",
        truthful_text="Paris",
        incorrect_text="Lyon",
    )
    assert assignment.candidate_a == "Paris"
    assert assignment.candidate_b == "Lyon"
    assert assignment.truthful_label == "A"
    assert assignment.incorrect_label == "B"
    assert assignment.order_regime == "CF"


@pytest.mark.unit
def test_ordering__incorrect_first__maps_truth_to_B() -> None:
    """PROMPT-003: IF maps truthful candidate to B."""
    assignment = assign_order(
        order_regime="IF",
        truthful_text="Paris",
        incorrect_text="Lyon",
    )
    assert assignment.candidate_a == "Lyon"
    assert assignment.candidate_b == "Paris"
    assert assignment.truthful_label == "B"
    assert assignment.incorrect_label == "A"
    assert assignment.order_regime == "IF"


@pytest.mark.unit
def test_ordering__random_order__is_deterministic_for_seed_and_question_id() -> None:
    """PROMPT-004: same (ro_seed, question_id) always yields the same RO assignment.

    Expected labels from DEC-009: SHA-256(f\"{ro_seed}\\0{question_id}\");
    LSB of first byte 0 → truthful_label A, else B.
    For ro_seed=42: q1→B, q2→A, q3→B.
    """
    ro_seed = 42
    expected_labels = {"q1": "B", "q2": "A", "q3": "B"}
    first_pass: dict[str, object] = {}
    for question_id, expected_label in expected_labels.items():
        assignment = assign_order(
            order_regime="RO",
            truthful_text="Paris",
            incorrect_text="Lyon",
            question_id=question_id,
            ro_seed=ro_seed,
        )
        assert assignment.order_regime == "RO"
        assert assignment.truthful_label == expected_label
        assert assignment.incorrect_label == ("A" if expected_label == "B" else "B")
        assert assignment.order_manifest_id == f"ro:primary:{ro_seed}"
        if expected_label == "A":
            assert assignment.candidate_a == "Paris"
            assert assignment.candidate_b == "Lyon"
        else:
            assert assignment.candidate_a == "Lyon"
            assert assignment.candidate_b == "Paris"
        first_pass[question_id] = assignment

    # Call order independence: reverse iteration must match.
    for question_id in reversed(list(expected_labels)):
        again = assign_order(
            order_regime="RO",
            truthful_text="Paris",
            incorrect_text="Lyon",
            question_id=question_id,
            ro_seed=ro_seed,
        )
        assert again == first_pass[question_id]
