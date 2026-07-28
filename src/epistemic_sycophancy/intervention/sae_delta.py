"""Additive SAE latent updates and residual deltas."""

from __future__ import annotations

from collections.abc import Sequence


def normalized_coefficients(
    *,
    scales: Sequence[float],
    beta: Sequence[float],
) -> list[float]:
    """Return α_j = s_j β_j for each selected feature.

    Shapes: scales and beta must have equal length m (selected features).
    """
    return [float(s) * float(b) for s, b in zip(scales, beta)]
