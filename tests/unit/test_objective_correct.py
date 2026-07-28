"""Correct-belief preservation hinge tests (Phase G OBJ)."""

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


@pytest.mark.unit
def test_objective__correct_belief_preservation__uses_only_q_plus() -> None:
    """OBJ-007: correct-belief hinge uses only Q+; q2 excluded."""
    from epistemic_sycophancy.objective.total import correct_belief_question_penalties

    penalties = correct_belief_question_penalties(
        baseline_cb_margins=GOLDEN_BASELINE_CB_MARGINS,
        current_cb_margins=GOLDEN_CURRENT_CB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        delta_c=GOLDEN_DELTA_C,
    )
    assert set(penalties) == set(GOLDEN_Q_PLUS)
    assert set(penalties).isdisjoint(GOLDEN_Q_MINUS)
    # q2 has CB margins but must not appear
    assert "q2" not in penalties
