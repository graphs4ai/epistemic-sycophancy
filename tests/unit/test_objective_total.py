"""Total objective golden fixture tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location("golden_objective_total", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_BASELINE_CB_MARGINS = _golden.GOLDEN_BASELINE_CB_MARGINS
GOLDEN_BASELINE_NEUTRAL_MARGINS = _golden.GOLDEN_BASELINE_NEUTRAL_MARGINS
GOLDEN_BETA = _golden.GOLDEN_BETA
GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_CURRENT_IB_MARGINS = _golden.GOLDEN_CURRENT_IB_MARGINS
GOLDEN_CURRENT_NEUTRAL_MARGINS = _golden.GOLDEN_CURRENT_NEUTRAL_MARGINS
GOLDEN_DELTA_C = _golden.GOLDEN_DELTA_C
GOLDEN_DELTA_N = _golden.GOLDEN_DELTA_N
GOLDEN_LAMBDA_BETA = _golden.GOLDEN_LAMBDA_BETA
GOLDEN_LAMBDA_C = _golden.GOLDEN_LAMBDA_C
GOLDEN_LAMBDA_N = _golden.GOLDEN_LAMBDA_N
GOLDEN_L_BEHAVIOR = _golden.GOLDEN_L_BEHAVIOR
GOLDEN_L_BETA = _golden.GOLDEN_L_BETA
GOLDEN_L_CORRECT = _golden.GOLDEN_L_CORRECT
GOLDEN_L_NEUTRAL = _golden.GOLDEN_L_NEUTRAL
GOLDEN_L_RECOVER = _golden.GOLDEN_L_RECOVER
GOLDEN_L_RESIST = _golden.GOLDEN_L_RESIST
GOLDEN_L_TOTAL = _golden.GOLDEN_L_TOTAL
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_TAU = _golden.GOLDEN_TAU
GOLDEN_W_R = _golden.GOLDEN_W_R
GOLDEN_W_U = _golden.GOLDEN_W_U


@pytest.mark.unit
def test_objective__golden_fixture__matches_expected_total() -> None:
    """OBJ-010: every golden component and L_total match transcribed §13.1 values."""
    from epistemic_sycophancy.objective.total import evaluate_objective

    result = evaluate_objective(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        cb_margins_by_question=GOLDEN_CURRENT_CB_MARGINS,
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        q_minus=GOLDEN_Q_MINUS,
        beta=GOLDEN_BETA,
        tau=GOLDEN_TAU,
        w_r=GOLDEN_W_R,
        w_u=GOLDEN_W_U,
        delta_n=GOLDEN_DELTA_N,
        delta_c=GOLDEN_DELTA_C,
        lambda_n=GOLDEN_LAMBDA_N,
        lambda_c=GOLDEN_LAMBDA_C,
        lambda_beta=GOLDEN_LAMBDA_BETA,
    )
    assert result.l_resist == pytest.approx(GOLDEN_L_RESIST, abs=1e-12, rel=1e-12)
    assert result.l_recover == pytest.approx(GOLDEN_L_RECOVER, abs=1e-12, rel=1e-12)
    assert result.l_behavior == pytest.approx(GOLDEN_L_BEHAVIOR, abs=1e-12, rel=1e-12)
    assert result.l_neutral == pytest.approx(GOLDEN_L_NEUTRAL, abs=1e-12, rel=1e-12)
    assert result.l_correct == pytest.approx(GOLDEN_L_CORRECT, abs=1e-12, rel=1e-12)
    assert result.l_beta == pytest.approx(GOLDEN_L_BETA, abs=1e-12, rel=1e-12)
    assert result.l_total == pytest.approx(GOLDEN_L_TOTAL, abs=1e-12, rel=1e-12)
    hand_total = (
        GOLDEN_L_BEHAVIOR
        + GOLDEN_LAMBDA_N * GOLDEN_L_NEUTRAL
        + GOLDEN_LAMBDA_C * GOLDEN_L_CORRECT
        + GOLDEN_LAMBDA_BETA * GOLDEN_L_BETA
    )
    assert result.l_total == pytest.approx(hand_total, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_objective__valid_inputs__always_returns_finite_scalar() -> None:
    """OBJ-016: valid golden-like inputs yield a finite scalar total."""
    from epistemic_sycophancy.objective.total import evaluate_objective

    result = evaluate_objective(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        cb_margins_by_question=GOLDEN_CURRENT_CB_MARGINS,
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        q_minus=GOLDEN_Q_MINUS,
        beta=GOLDEN_BETA,
        tau=GOLDEN_TAU,
        w_r=GOLDEN_W_R,
        w_u=GOLDEN_W_U,
        delta_n=GOLDEN_DELTA_N,
        delta_c=GOLDEN_DELTA_C,
        lambda_n=GOLDEN_LAMBDA_N,
        lambda_c=GOLDEN_LAMBDA_C,
        lambda_beta=GOLDEN_LAMBDA_BETA,
    )
    assert math.isfinite(result.l_total)
    for value in (
        result.l_resist,
        result.l_recover,
        result.l_behavior,
        result.l_neutral,
        result.l_correct,
        result.l_beta,
    ):
        assert math.isfinite(value)
