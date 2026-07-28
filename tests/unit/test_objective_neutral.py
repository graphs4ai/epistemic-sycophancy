"""Neutral preservation hinge tests (Phase G OBJ)."""

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
_spec = importlib.util.spec_from_file_location("golden_objective_neutral", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_BASELINE_NEUTRAL_MARGINS = _golden.GOLDEN_BASELINE_NEUTRAL_MARGINS
GOLDEN_CURRENT_NEUTRAL_MARGINS = _golden.GOLDEN_CURRENT_NEUTRAL_MARGINS
GOLDEN_DELTA_N = _golden.GOLDEN_DELTA_N
GOLDEN_L_NEUTRAL = _golden.GOLDEN_L_NEUTRAL
GOLDEN_OPTIMIZATION_QUESTIONS = _golden.GOLDEN_OPTIMIZATION_QUESTIONS


@pytest.mark.unit
def test_objective__neutral_penalty__penalizes_only_excess_margin_decrease() -> None:
    """OBJ-005: d_q,N = [M0 - M - δ_N]_+; improve/within-δ → 0."""
    from epistemic_sycophancy.objective.total import neutral_question_penalties

    penalties = neutral_question_penalties(
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        delta_n=GOLDEN_DELTA_N,
    )
    # Hand: q1 excess = 2.0 - 1.4 - 0.25 = 0.35; q2 improve; q3 improve within δ
    assert penalties["q1"] == pytest.approx(0.35, abs=1e-12, rel=1e-12)
    assert penalties["q2"] == pytest.approx(0.0, abs=1e-12, rel=1e-12)
    assert penalties["q3"] == pytest.approx(0.0, abs=1e-12, rel=1e-12)
