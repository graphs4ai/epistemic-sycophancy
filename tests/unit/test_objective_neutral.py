"""Neutral preservation soft-hinge tests (Phase G OBJ / DEC-101)."""

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
GOLDEN_TAU = _golden.GOLDEN_TAU


def _stable_softplus(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@pytest.mark.unit
def test_objective__neutral_penalty__averages_over_complete_optimization_question_set() -> None:
    """OBJ-006: L_neutral = (1/|Q|) Σ d_q,N over all optimization questions."""
    from epistemic_sycophancy.objective.total import neutral_preservation_loss

    l_neutral = neutral_preservation_loss(
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        delta_n=GOLDEN_DELTA_N,
        tau=GOLDEN_TAU,
    )
    assert l_neutral == pytest.approx(GOLDEN_L_NEUTRAL, abs=1e-12, rel=1e-12)
    assert len(GOLDEN_OPTIMIZATION_QUESTIONS) == 3
    # Denominator is |Q|=3, not |Q+|
    assert l_neutral != pytest.approx(
        (
            _stable_softplus((2.0 - 1.4 - 0.25) / GOLDEN_TAU)
            + _stable_softplus((0.5 - 0.8 - 0.25) / GOLDEN_TAU)
        )
        / 2.0,
        abs=1e-12,
        rel=1e-12,
    )


@pytest.mark.unit
def test_objective__neutral_penalty__penalizes_only_excess_margin_decrease() -> None:
    """OBJ-005: d_q,N = softplus((M0 - M - δ_N)/τ); larger when excess is larger."""
    from epistemic_sycophancy.objective.total import neutral_question_penalties

    penalties = neutral_question_penalties(
        baseline_neutral_margins=GOLDEN_BASELINE_NEUTRAL_MARGINS,
        current_neutral_margins=GOLDEN_CURRENT_NEUTRAL_MARGINS,
        delta_n=GOLDEN_DELTA_N,
        tau=GOLDEN_TAU,
    )
    # Hand softplus excesses: 0.35, -1.05, -0.55
    assert penalties["q1"] == pytest.approx(
        _stable_softplus(0.35 / GOLDEN_TAU), abs=1e-12, rel=1e-12
    )
    assert penalties["q2"] == pytest.approx(
        _stable_softplus(-1.05 / GOLDEN_TAU), abs=1e-12, rel=1e-12
    )
    assert penalties["q3"] == pytest.approx(
        _stable_softplus(-0.55 / GOLDEN_TAU), abs=1e-12, rel=1e-12
    )
    assert penalties["q1"] > penalties["q3"] > penalties["q2"]
