"""Logistic margin loss tests (Phase C LOSS)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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


_finite = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=-1e3,
    max_value=1e3,
)
_tau = st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=64,
    min_value=1e-3,
    max_value=1e3,
)


@pytest.mark.property
@given(m1=_finite, m2=_finite, tau=_tau)
@settings(max_examples=100)
def test_logistic_loss__larger_truthful_margin__never_increases_loss(
    m1: float,
    m2: float,
    tau: float,
) -> None:
    """LOSS-002: M1 < M2 ⇒ φ(M1) ≥ φ(M2)."""
    lo, hi = (m1, m2) if m1 <= m2 else (m2, m1)
    if lo == hi:
        assert logistic_margin_loss(lo, tau=tau) == pytest.approx(
            logistic_margin_loss(hi, tau=tau), abs=1e-12, rel=1e-12
        )
        return
    loss_lo = logistic_margin_loss(lo, tau=tau)
    loss_hi = logistic_margin_loss(hi, tau=tau)
    assert loss_lo >= loss_hi - 1e-12
