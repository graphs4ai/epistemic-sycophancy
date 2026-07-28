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
