"""Correct-belief preservation soft-hinge tests (Phase G OBJ / DEC-101)."""

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
_spec = importlib.util.spec_from_file_location("golden_objective_correct", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_BASELINE_CB_MARGINS = _golden.GOLDEN_BASELINE_CB_MARGINS
GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_DELTA_C = _golden.GOLDEN_DELTA_C
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_L_CORRECT = _golden.GOLDEN_L_CORRECT
GOLDEN_TAU = _golden.GOLDEN_TAU


def _stable_softplus(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@pytest.mark.unit
def test_objective__correct_belief_preservation__means_variants_within_question() -> None:
    """OBJ-008: mean CB-variant soft-hinges within question, then across Q+."""
    from epistemic_sycophancy.objective.total import correct_belief_preservation_loss

    l_correct = correct_belief_preservation_loss(
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        delta_c=GOLDEN_DELTA_C,
        tau=GOLDEN_TAU,
    )
    assert l_correct == pytest.approx(GOLDEN_L_CORRECT, abs=1e-12, rel=1e-12)
    q1_mean = (
        _stable_softplus(0.2 / GOLDEN_TAU) + _stable_softplus(0.9 / GOLDEN_TAU)
    ) / 2.0
    q3_mean = _stable_softplus(-0.15 / GOLDEN_TAU)
    assert l_correct == pytest.approx((q1_mean + q3_mean) / 2.0, abs=1e-12, rel=1e-12)


@pytest.mark.unit
def test_objective__correct_belief_preservation__uses_only_q_plus() -> None:
    """OBJ-007: correct-belief soft-hinge uses only Q+; q2 excluded."""
    from epistemic_sycophancy.objective.total import correct_belief_question_penalties

    penalties = correct_belief_question_penalties(
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        delta_c=GOLDEN_DELTA_C,
        tau=GOLDEN_TAU,
    )
    assert set(penalties) == set(GOLDEN_Q_PLUS)
    assert set(penalties).isdisjoint(GOLDEN_Q_MINUS)
    # q2 has CB margins but must not appear
    assert "q2" not in penalties
