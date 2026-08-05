"""Logistic margin loss φ(M) = softplus(-M/τ) and soft preservation hinges."""

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


def baseline_relative_hard_hinge(
    *,
    baseline_margin: float | object,
    current_margin: float | object,
    delta: float,
) -> float | object:
    """Return hard [M0 - M(β) - δ]_+ (FEAT-011 contrast / historical hinge).

    Kept for tests that document why hard hinges are flat at β=0 when δ>0.
    The optimizer uses :func:`baseline_relative_hinge` (softplus) instead.
    """
    excess = baseline_margin - current_margin - float(delta)
    try:
        import torch

        if isinstance(excess, torch.Tensor):
            return torch.relu(excess)
    except ImportError:
        pass
    return max(0.0, float(excess))


def baseline_relative_hinge(
    *,
    baseline_margin: float | object,
    current_margin: float | object,
    delta: float,
    tau: float,
) -> float | object:
    """Return softplus((M0 - M(β) - δ)/τ) for τ > 0.

    Soft-hinge of the baseline-relative excess (DEC-101). Accepts Python
    floats or torch tensors. Unlike the hard ReLU hinge, this is C^∞ and
    nonzero (with nonzero ∂/∂M) at the null intervention when δ > 0.
    """
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be strictly positive; got {tau!r}")
    excess = baseline_margin - current_margin - float(delta)
    scaled = excess / float(tau)
    try:
        import torch

        if isinstance(scaled, torch.Tensor):
            return torch.nn.functional.softplus(scaled)
        if isinstance(excess, torch.Tensor):
            return torch.nn.functional.softplus(excess / float(tau))
    except ImportError:
        pass
    return _softplus(float(scaled))
