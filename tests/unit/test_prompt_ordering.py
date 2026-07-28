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
