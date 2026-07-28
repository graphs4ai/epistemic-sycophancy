"""Logistic margin loss φ(M) = softplus(-M/τ)."""

from __future__ import annotations

import math


def _softplus(x: float) -> float:
    """Numerically stable softplus."""
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def logistic_margin_loss(margin: float, *, tau: float) -> float:
    """Return softplus(-margin / tau) for tau > 0."""
    if tau <= 0.0:
        raise ValueError(f"tau must be strictly positive; got {tau!r}")
    return _softplus(-float(margin) / float(tau))


def mean_logistic_margin_loss(margins: list[float], *, tau: float) -> float:
    """Return mean_b φ(M_b); not φ(mean_b M_b).

    Tiny inline mean for the LOSS-006 nonlinearity invariant only —
    not the Phase D question-macro aggregation module.
    """
    if not margins:
        raise ValueError("margins must be non-empty")
    total = 0.0
    for margin in margins:
        total += logistic_margin_loss(float(margin), tau=tau)
    return total / float(len(margins))
