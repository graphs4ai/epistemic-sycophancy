"""Semantic truthful margin tests (Phase C SCORE)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.prompts.ordering import assign_order
from epistemic_sycophancy.scoring.margins import truthful_margin


@pytest.mark.unit
def test_margin__all_orders__subtracts_incorrect_score_from_truthful_score() -> None:
    """SCORE-001: M = s_truthful - s_incorrect under CF, IF, and RO."""
    score_a = 2.5
    score_b = -1.0

    cf = assign_order(
        order_regime="CF",
        truthful_text="Paris",
        incorrect_text="Lyon",
    )
    # CF: truthful=A, incorrect=B → M = s_A - s_B
    assert cf.truthful_label == "A"
    assert truthful_margin(
        score_a=score_a,
        score_b=score_b,
        truthful_label=cf.truthful_label,
    ) == pytest.approx(score_a - score_b, abs=1e-12, rel=1e-12)

    if_order = assign_order(
        order_regime="IF",
        truthful_text="Paris",
        incorrect_text="Lyon",
    )
    # IF: truthful=B, incorrect=A → M = s_B - s_A
    assert if_order.truthful_label == "B"
    assert truthful_margin(
        score_a=score_a,
        score_b=score_b,
        truthful_label=if_order.truthful_label,
    ) == pytest.approx(score_b - score_a, abs=1e-12, rel=1e-12)

    # RO uses truthful_label from DEC-009 assignment (not letter display order).
    ro = assign_order(
        order_regime="RO",
        truthful_text="Paris",
        incorrect_text="Lyon",
        question_id="q1",
        ro_seed=42,
    )
    expected_ro = (
        score_a - score_b if ro.truthful_label == "A" else score_b - score_a
    )
    assert truthful_margin(
        score_a=score_a,
        score_b=score_b,
        truthful_label=ro.truthful_label,
    ) == pytest.approx(expected_ro, abs=1e-12, rel=1e-12)


_finite_scores = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-1e6,
    max_value=1e6,
)


@pytest.mark.property
@given(score_t=_finite_scores, score_f=_finite_scores)
@settings(max_examples=100)
def test_margin__swapping_candidate_positions_and_scores__preserves_semantic_margin(
    score_t: float,
    score_f: float,
) -> None:
    """SCORE-002: rendering truth as A or B preserves M = s_T - s_F."""
    # Truth as A (CF-like): score_a = s_T, score_b = s_F
    margin_truth_as_a = truthful_margin(
        score_a=score_t,
        score_b=score_f,
        truthful_label="A",
    )
    # Truth as B (IF-like): score_a = s_F, score_b = s_T
    margin_truth_as_b = truthful_margin(
        score_a=score_f,
        score_b=score_t,
        truthful_label="B",
    )
    assert margin_truth_as_a == pytest.approx(
        margin_truth_as_b, abs=1e-12, rel=1e-12
    )
    assert margin_truth_as_a == pytest.approx(
        score_t - score_f, abs=1e-12, rel=1e-12
    )
    assert math.isfinite(margin_truth_as_a)
