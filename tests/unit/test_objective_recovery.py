"""Recovery objective component tests (Phase G OBJ)."""

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
_spec = importlib.util.spec_from_file_location("golden_objective_recovery", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_TAU = _golden.GOLDEN_TAU
GOLDEN_L_RECOVER = _golden.GOLDEN_L_RECOVER


@pytest.mark.unit
def test_objective__recovery__means_within_question_then_across_q_minus() -> None:
    """OBJ-003: L_recover = question-macro φ on CB of q∈Q- only."""
    from epistemic_sycophancy.objective.total import recovery_loss

    l_recover = recovery_loss(
        cb_margins_by_question=GOLDEN_CURRENT_CB_MARGINS,
        q_minus=GOLDEN_Q_MINUS,
        tau=GOLDEN_TAU,
    )
    assert l_recover == pytest.approx(GOLDEN_L_RECOVER, abs=1e-12, rel=1e-12)

    # Q+ CB must not enter recovery
    only_q_minus = {qid: GOLDEN_CURRENT_CB_MARGINS[qid] for qid in GOLDEN_Q_MINUS}
    assert recovery_loss(
        cb_margins_by_question=only_q_minus,
        q_minus=GOLDEN_Q_MINUS,
        tau=GOLDEN_TAU,
    ) == pytest.approx(l_recover, abs=1e-12, rel=1e-12)
    assert set(GOLDEN_Q_PLUS).isdisjoint(GOLDEN_Q_MINUS)
