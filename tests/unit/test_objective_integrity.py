"""Objective data-integrity tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from epistemic_sycophancy.data.validation import DataIntegrityError

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location(
    "golden_objective_integrity", _GOLDEN_PATH
)
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
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_TAU = _golden.GOLDEN_TAU
GOLDEN_W_R = _golden.GOLDEN_W_R
GOLDEN_W_U = _golden.GOLDEN_W_U


def _evaluate(**overrides):
    from epistemic_sycophancy.objective.total import evaluate_objective

    kwargs = dict(
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
    kwargs.update(overrides)
    return evaluate_objective(**kwargs)


@pytest.mark.unit
def test_objective__missing_ib_or_cb_variants__raises_data_integrity_error() -> None:
    """OBJ-017 / DEC-028: missing required IB/CB rows raise DataIntegrityError."""
    # Missing IB variants for q1 ∈ Q+
    ib_missing = {
        "q1": [],
        "q2": GOLDEN_CURRENT_IB_MARGINS["q2"],
        "q3": GOLDEN_CURRENT_IB_MARGINS["q3"],
    }
    with pytest.raises(DataIntegrityError):
        _evaluate(ib_margins_by_question=ib_missing)

    # Missing CB variants for q2 ∈ Q-
    cb_missing_recovery = {
        "q1": GOLDEN_CURRENT_CB_MARGINS["q1"],
        "q2": [],
        "q3": GOLDEN_CURRENT_CB_MARGINS["q3"],
    }
    with pytest.raises(DataIntegrityError):
        _evaluate(cb_margins_by_question=cb_missing_recovery)

    # Missing CB variants for q3 ∈ Q+ (correct hinge)
    cb_missing_correct = {
        "q1": GOLDEN_CURRENT_CB_MARGINS["q1"],
        "q2": GOLDEN_CURRENT_CB_MARGINS["q2"],
        "q3": [],
    }
    with pytest.raises(DataIntegrityError):
        _evaluate(cb_margins_by_question=cb_missing_correct)
