"""Logistic margin loss tests (Phase C LOSS)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from epistemic_sycophancy.objective.losses import (
    baseline_relative_hinge,
    logistic_margin_loss,
)


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


@pytest.mark.unit
@pytest.mark.parametrize("tau", [0.1, 1.0, 2.5, 10.0])
def test_logistic_loss__zero_margin__equals_log_two(tau: float) -> None:
    """LOSS-003: φ(0) = log 2 for any valid τ."""
    assert logistic_margin_loss(0.0, tau=tau) == pytest.approx(
        math.log(2.0), abs=1e-12, rel=1e-12
    )


@pytest.mark.unit
def test_logistic_loss__tau__changes_margin_scale_but_not_ordering() -> None:
    """LOSS-004: τ changes magnitude at fixed M≠0; margin ordering of φ preserved."""
    margins = [-2.0, -0.5, 0.5, 2.0]
    tau_small = 0.5
    tau_large = 2.0

    losses_small = [logistic_margin_loss(m, tau=tau_small) for m in margins]
    losses_large = [logistic_margin_loss(m, tau=tau_large) for m in margins]

    # At fixed nonzero margin, changing τ changes loss magnitude.
    for m, ls, ll in zip(margins, losses_small, losses_large):
        if m == 0.0:
            continue
        assert ls != pytest.approx(ll, abs=1e-12, rel=1e-12)

    # Ordering by margin must not reverse.
    for i in range(len(margins) - 1):
        assert losses_small[i] >= losses_small[i + 1] - 1e-12
        assert losses_large[i] >= losses_large[i + 1] - 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("margin", [-1e4, -100.0, 100.0, 1e4])
def test_logistic_loss__extreme_margins__remains_finite(margin: float) -> None:
    """LOSS-005: extreme margins remain finite under production float dtype."""
    loss = logistic_margin_loss(margin, tau=1.0)
    assert math.isfinite(loss)


@pytest.mark.unit
def test_aggregation__loss_before_mean__does_not_equal_loss_of_mean_margin() -> None:
    """LOSS-006: mean_b φ(M_b) ≠ φ(mean_b M_b); production uses loss-before-mean."""
    from epistemic_sycophancy.objective.losses import mean_logistic_margin_loss

    margins = [3.0, -3.0]
    tau = 1.0

    mean_of_margins = sum(margins) / len(margins)
    assert mean_of_margins == pytest.approx(0.0, abs=1e-12, rel=1e-12)

    loss_of_mean = logistic_margin_loss(mean_of_margins, tau=tau)
    assert loss_of_mean == pytest.approx(math.log(2.0), abs=1e-12, rel=1e-12)

    # Independent check: mean(softplus(-M)) for ±3 exceeds log(2).
    independent_mean_loss = (
        _stable_softplus(-3.0 / tau) + _stable_softplus(-(-3.0) / tau)
    ) / 2.0
    assert independent_mean_loss > math.log(2.0)

    production = mean_logistic_margin_loss(margins, tau=tau)
    assert production == pytest.approx(independent_mean_loss, abs=1e-12, rel=1e-12)
    assert production > math.log(2.0)
    assert production != pytest.approx(loss_of_mean, abs=1e-12, rel=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("baseline", "current", "delta", "tau"),
    [
        # Active excess: softplus(0.35) > hard ReLU 0.35
        (2.0, 1.4, 0.25, 1.0),
        # Inactive (improve / within δ): softplus(negative) > 0, unlike hard hinge
        (-1.0, -0.2, 0.25, 1.0),
        (0.5, 0.8, 0.25, 1.0),
        # Null intervention with δ>0: softplus(-δ/τ) > 0 and not flat
        (2.0, 2.0, 0.25, 1.0),
        (1.5, 1.0, 0.1, 0.5),
    ],
)
def test_soft_hinge__excess__matches_stable_softplus(
    baseline: float,
    current: float,
    delta: float,
    tau: float,
) -> None:
    """LOSS-007: d = softplus((M0 - M - δ)/τ); not the hard ReLU hinge."""
    excess = baseline - current - delta
    expected = _stable_softplus(excess / tau)
    hard = max(0.0, excess)
    got = baseline_relative_hinge(
        baseline_margin=baseline,
        current_margin=current,
        delta=delta,
        tau=tau,
    )
    assert got == pytest.approx(expected, abs=1e-12, rel=1e-12)
    # Distinct from hard hinge except at the isolated root of softplus(x)=max(x,0)
    # (never for finite negative excess; generally not for positive excess either).
    if excess != 0.0:
        assert got != pytest.approx(hard, abs=1e-12, rel=1e-12)
