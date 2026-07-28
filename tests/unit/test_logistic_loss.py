"""Logistic margin loss tests (Phase C LOSS)."""

from __future__ import annotations

import math

import pytest

from epistemic_sycophancy.objective.losses import logistic_margin_loss


def _stable_softplus(x: float) -> float:
    """Independent high-precision softplus reference (not the production path)."""
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("margin", "tau"),
    [
        (0.0, 1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (2.5, 0.5),
        (-3.0, 2.0),
    ],
)
def test_logistic_loss__reference_values__match_stable_softplus(
    margin: float,
    tau: float,
) -> None:
    """LOSS-001: φ(M) = softplus(-M/τ) matches independent stable softplus."""
    expected = _stable_softplus(-margin / tau)
    assert logistic_margin_loss(margin, tau=tau) == pytest.approx(
        expected, abs=1e-12, rel=1e-12
    )
