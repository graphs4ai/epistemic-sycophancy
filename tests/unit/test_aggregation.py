"""Question-macro aggregation tests (Phase D AGG scaffolding)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.objective.aggregation import question_macro_mean


@pytest.mark.unit
def test_aggregation__question_macro_mean__means_within_question_then_across_questions() -> None:
    """AGG-001: mean within question, then mean across questions (equal question weight).

    Hand-derived: q1 mean([1, 3]) = 2; q2 mean([4]) = 4; macro = mean(2, 4) = 3.
    """
    values_by_question = {
        "q1": [1.0, 3.0],
        "q2": [4.0],
    }
    assert question_macro_mean(values_by_question) == pytest.approx(3.0)


@pytest.mark.unit
def test_aggregation__unequal_variant_counts__preserve_equal_question_weight() -> None:
    """AGG-002: prompt-pool mean ≠ question macro under unequal variant counts.

    Hand-derived: q1 three 0s → 0; q2 one 1 → 1; macro = 0.5.
    Prompt pool would be 0.25 — must not be returned.
    """
    values_by_question = {
        "q1": [0.0, 0.0, 0.0],
        "q2": [1.0],
    }
    result = question_macro_mean(values_by_question)
    assert result == pytest.approx(0.5)
    assert result != pytest.approx(0.25)
