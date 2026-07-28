"""Signed-Jacobian candidate ranking (Phase F, DEC-019)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class SuppressionCandidate:
    """One ranked candidate feature k = (layer, feature_id)."""

    layer: int
    feature_id: int
    signed_jacobian: float
    absolute_sensitivity: float
    suppression_beneficial: bool


def rank_suppression_candidates(
    *,
    signed_jacobians: Mapping[tuple[int, int], float],
) -> tuple[SuppressionCandidate, ...]:
    """Rank candidates by descending signed Jacobian (DEC-019).

    Because ΔL ≈ J_j Δβ_j with Δβ_j ≤ 0, a positive signed Jacobian predicts
    a loss reduction under suppression. Absolute sensitivity is kept as a
    diagnostic column only and never drives the order. Ties break on
    ascending ``(layer, feature_id)``.
    """
    candidates = [
        SuppressionCandidate(
            layer=layer,
            feature_id=feature_id,
            signed_jacobian=float(jacobian),
            absolute_sensitivity=abs(float(jacobian)),
            suppression_beneficial=float(jacobian) > 0.0,
        )
        for (layer, feature_id), jacobian in signed_jacobians.items()
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.signed_jacobian,
            candidate.layer,
            candidate.feature_id,
        )
    )
    return tuple(candidates)
