"""Resistance objective component tests (Phase G OBJ)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from epistemic_sycophancy.objective.total import resistance_prompt_losses

_GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "objective"
    / "golden_objective.py"
)
_spec = importlib.util.spec_from_file_location("golden_objective", _GOLDEN_PATH)
assert _spec is not None and _spec.loader is not None
_golden = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _golden
_spec.loader.exec_module(_golden)

GOLDEN_CURRENT_IB_MARGINS = _golden.GOLDEN_CURRENT_IB_MARGINS
GOLDEN_Q_PLUS = _golden.GOLDEN_Q_PLUS
GOLDEN_TAU = _golden.GOLDEN_TAU
GOLDEN_Q1_IB_MEAN = _golden.GOLDEN_Q1_IB_MEAN
GOLDEN_Q3_IB_MEAN = _golden.GOLDEN_Q3_IB_MEAN
GOLDEN_L_RESIST = _golden.GOLDEN_L_RESIST


@pytest.mark.unit
def test_objective__resistance__means_within_question_then_across_q_plus() -> None:
    """OBJ-002: L_resist = mean(q1_mean, q3_mean) over Q+ only."""
    from epistemic_sycophancy.objective.total import resistance_loss

    l_resist = resistance_loss(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        tau=GOLDEN_TAU,
    )
    prompt_losses = resistance_prompt_losses(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        tau=GOLDEN_TAU,
    )
    q1_mean = sum(prompt_losses["q1"]) / len(prompt_losses["q1"])
    q3_mean = sum(prompt_losses["q3"]) / len(prompt_losses["q3"])
    assert q1_mean == pytest.approx(GOLDEN_Q1_IB_MEAN, abs=1e-12, rel=1e-12)
    assert q3_mean == pytest.approx(GOLDEN_Q3_IB_MEAN, abs=1e-12, rel=1e-12)
    assert l_resist == pytest.approx(GOLDEN_L_RESIST, abs=1e-12, rel=1e-12)
    assert l_resist == pytest.approx(
        (GOLDEN_Q1_IB_MEAN + GOLDEN_Q3_IB_MEAN) / 2.0, abs=1e-12, rel=1e-12
    )


def _stable_softplus(x: float) -> float:
    """Independent softplus reference (not the production path)."""
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _phi(margin: float, *, tau: float) -> float:
    return _stable_softplus(-margin / tau)


@pytest.mark.unit
def test_objective__resistance__applies_logistic_loss_to_each_ib_prompt() -> None:
    """OBJ-001: φ(M) on each IB prompt of q∈Q+ only; q2 IB excluded."""
    losses = resistance_prompt_losses(
        ib_margins_by_question=GOLDEN_CURRENT_IB_MARGINS,
        q_plus=GOLDEN_Q_PLUS,
        tau=GOLDEN_TAU,
    )

    assert set(losses) == set(GOLDEN_Q_PLUS)
    assert "q2" not in losses

    for question_id in GOLDEN_Q_PLUS:
        margins = GOLDEN_CURRENT_IB_MARGINS[question_id]
        expected = [_phi(m, tau=GOLDEN_TAU) for m in margins]
        assert losses[question_id] == pytest.approx(expected, abs=1e-12, rel=1e-12)
