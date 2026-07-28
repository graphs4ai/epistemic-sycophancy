"""Behavioral objective mix tests (Phase G OBJ)."""

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
_spec = importlib.util.spec_from_file_location("golden_objective_behavior", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_L_RESIST = _golden.GOLDEN_L_RESIST
GOLDEN_L_RECOVER = _golden.GOLDEN_L_RECOVER
GOLDEN_L_BEHAVIOR = _golden.GOLDEN_L_BEHAVIOR
GOLDEN_W_R = _golden.GOLDEN_W_R
GOLDEN_W_U = _golden.GOLDEN_W_U
GOLDEN_CURRENT_IB_MARGINS = _golden.GOLDEN_CURRENT_IB_MARGINS
GOLDEN_CURRENT_CB_MARGINS = _golden.GOLDEN_CURRENT_CB_MARGINS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_Q_MINUS = _golden.GOLDEN_Q_MINUS
GOLDEN_TAU = _golden.GOLDEN_TAU


@pytest.mark.unit
def test_objective__behavior__uses_explicit_component_weights_not_subset_sizes() -> None:
    """OBJ-004: L_behavior = w_R L_resist + w_U L_recover; not |Q+| / |Q-|."""
    from epistemic_sycophancy.objective.total import (
        behavioral_loss,
        recovery_loss,
        resistance_loss,
    )

    l_resist = resistance_loss(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        tau=GOLDEN_TAU,
    )
    l_recover = recovery_loss(
        cb_margins_by_question=GOLDEN_CURRENT_CB_MARGINS,
        q_minus=GOLDEN_Q_MINUS,
        tau=GOLDEN_TAU,
    )
    l_behavior = behavioral_loss(
        l_resist=l_resist,
        l_recover=l_recover,
        w_r=GOLDEN_W_R,
        w_u=GOLDEN_W_U,
    )
    expected = GOLDEN_W_R * GOLDEN_L_RESIST + GOLDEN_W_U * GOLDEN_L_RECOVER
    assert l_behavior == pytest.approx(GOLDEN_L_BEHAVIOR, abs=1e-12, rel=1e-12)
    assert l_behavior == pytest.approx(expected, abs=1e-12, rel=1e-12)

    # Subset-size weighting would differ: |Q+|=2, |Q-|=1 → (2*L_r + 1*L_u)/3
    size_weighted = (2.0 * l_resist + 1.0 * l_recover) / 3.0
    assert l_behavior != pytest.approx(size_weighted, abs=1e-12, rel=1e-12)
