"""Optimizer objective determinism tests (Phase H OPT-001)."""

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
_spec = importlib.util.spec_from_file_location("golden_objective_opt", _GOLDEN_PATH)
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


def _golden_kwargs() -> dict:
    return {
        "ib_margins_by_question": GOLDEN_CURRENT_IB_MARGINS,
        "cb_margins_by_question": GOLDEN_CURRENT_CB_MARGINS,
        "baseline_cb_margins": GOLDEN_BASELINE_CB_MARGINS,
        "baseline_neutral_margins": GOLDEN_BASELINE_NEUTRAL_MARGINS,
        "current_neutral_margins": GOLDEN_CURRENT_NEUTRAL_MARGINS,
        "q_plus": GOLDEN_Q_PLUS,
        "q_minus": GOLDEN_Q_MINUS,
        "beta": list(GOLDEN_BETA),
        "tau": GOLDEN_TAU,
        "w_r": GOLDEN_W_R,
        "w_u": GOLDEN_W_U,
        "delta_n": GOLDEN_DELTA_N,
        "delta_c": GOLDEN_DELTA_C,
        "lambda_n": GOLDEN_LAMBDA_N,
        "lambda_c": GOLDEN_LAMBDA_C,
        "lambda_beta": GOLDEN_LAMBDA_BETA,
    }


@pytest.mark.unit
def test_optimizer_objective__same_beta_and_regime__returns_identical_scalar_and_components() -> None:
    """OPT-001: same β and regime → identical scalar and DEC-026 components."""
    from epistemic_sycophancy.optimization.objective import evaluate_optimizer_objective

    kwargs = _golden_kwargs()
    first = evaluate_optimizer_objective(**kwargs)
    second = evaluate_optimizer_objective(**kwargs)

    assert first.l_total == second.l_total
    assert first.l_resist == second.l_resist
    assert first.l_recover == second.l_recover
    assert first.l_behavior == second.l_behavior
    assert first.l_neutral == second.l_neutral
    assert first.l_correct == second.l_correct
    assert first.l_beta == second.l_beta
    assert first.l_residual_perturbation == second.l_residual_perturbation
    assert first.objective_version == second.objective_version
