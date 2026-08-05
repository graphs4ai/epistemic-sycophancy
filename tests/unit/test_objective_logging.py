"""Objective logging consistency tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location("golden_objective_logging", _GOLDEN_PATH)
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


@pytest.mark.unit
def test_objective__logged_components__sum_to_logged_total() -> None:
    """OBJ-011 / DEC-026: logged parts satisfy weighted sum identity."""
    from epistemic_sycophancy.logging.trial_records import build_objective_components
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
    logged = build_objective_components(
        result,
        lambda_n=GOLDEN_LAMBDA_N,
        lambda_c=GOLDEN_LAMBDA_C,
        lambda_beta=GOLDEN_LAMBDA_BETA,
    )
    reconstructed = (
        logged.l_behavior
        + GOLDEN_LAMBDA_N * logged.l_neutral
        + GOLDEN_LAMBDA_C * logged.l_correct
        + GOLDEN_LAMBDA_BETA * logged.l_beta
    )
    assert reconstructed == pytest.approx(logged.l_total, abs=1e-12, rel=1e-12)
    assert logged.l_total == pytest.approx(result.l_total, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_objective__initial_version__logs_but_does_not_add_residual_perturbation() -> None:
    """OBJ-018 / DEC-029/101: residual may be logged; current version omits it from L_total."""
    from epistemic_sycophancy.logging.trial_records import (
        OBJECTIVE_VERSION_CURRENT,
        build_objective_components,
    )
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
    residual = 7.5
    logged = build_objective_components(
        result,
        lambda_n=GOLDEN_LAMBDA_N,
        lambda_c=GOLDEN_LAMBDA_C,
        lambda_beta=GOLDEN_LAMBDA_BETA,
        l_residual_perturbation=residual,
    )
    assert logged.objective_version == OBJECTIVE_VERSION_CURRENT
    assert logged.objective_version == "v2_soft_hinge_no_residual"
    assert logged.l_residual_perturbation == pytest.approx(residual, abs=1e-12)
    # Total equals the soft-hinge assembly without residual
    without_residual = (
        logged.l_behavior
        + GOLDEN_LAMBDA_N * logged.l_neutral
        + GOLDEN_LAMBDA_C * logged.l_correct
        + GOLDEN_LAMBDA_BETA * logged.l_beta
    )
    assert logged.l_total == pytest.approx(without_residual, abs=1e-12, rel=1e-12)
    assert logged.l_total != pytest.approx(
        without_residual + residual, abs=1e-12, rel=1e-12
    )
