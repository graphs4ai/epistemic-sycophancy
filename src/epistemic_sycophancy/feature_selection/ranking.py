"""Signed-Jacobian candidate ranking (Phase F, DEC-019)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from epistemic_sycophancy.feature_selection.artifacts import FeatureSelectionArtifact


@dataclass(frozen=True)
class SuppressionCandidate:
    """One ranked candidate feature k = (layer, feature_id)."""

    layer: int
    feature_id: int
    signed_jacobian: float
    absolute_sensitivity: float
    suppression_beneficial: bool
    preferred_bidirectional_sign: float


@dataclass(frozen=True)
class AnnotatedSuppressionCandidate:
    """Behavior-ranked candidate with preservation-surrogate annotations (FEAT-026)."""

    layer: int
    feature_id: int
    signed_jacobian: float
    absolute_sensitivity: float
    suppression_beneficial: bool
    preferred_bidirectional_sign: float
    neutral_jacobian: float
    correct_surrogate_jacobian: float


def _preferred_bidirectional_sign(signed_jacobian: float) -> float:
    """Return -sign(J): the loss-decreasing direction for an unconstrained step."""
    if signed_jacobian > 0.0:
        return -1.0
    if signed_jacobian < 0.0:
        return 1.0
    return 0.0


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
            preferred_bidirectional_sign=_preferred_bidirectional_sign(
                float(jacobian)
            ),
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


def annotate_preservation_jacobians(
    *,
    candidates: Sequence[SuppressionCandidate],
    neutral_jacobians: Mapping[tuple[int, int], float],
    correct_surrogate_jacobians: Mapping[tuple[int, int], float],
) -> tuple[AnnotatedSuppressionCandidate, ...]:
    """Attach signed preservation Jacobians; do not re-rank (FEAT-026 / DEC-019).

    Neutral and correct-surrogate sensitivities are annotations only. They never
    enter the behavior rank and never veto eligibility.
    """
    annotated: list[AnnotatedSuppressionCandidate] = []
    for candidate in candidates:
        key = (candidate.layer, candidate.feature_id)
        annotated.append(
            AnnotatedSuppressionCandidate(
                layer=candidate.layer,
                feature_id=candidate.feature_id,
                signed_jacobian=candidate.signed_jacobian,
                absolute_sensitivity=candidate.absolute_sensitivity,
                suppression_beneficial=candidate.suppression_beneficial,
                preferred_bidirectional_sign=candidate.preferred_bidirectional_sign,
                neutral_jacobian=float(neutral_jacobians[key]),
                correct_surrogate_jacobian=float(correct_surrogate_jacobians[key]),
            )
        )
    return tuple(annotated)


def build_order_specific_artifacts(
    *,
    artifacts_by_order: Mapping[str, FeatureSelectionArtifact],
    component: str,
    model_revision_hash: str,
    sae_revision_hash: str,
    scope: str,
    scale_source: str,
    dataset_manifest_hash: str,
) -> dict[str, FeatureSelectionArtifact]:
    """Freeze one fingerprinted artifact per answer-order regime (FEAT-031).

    CF, IF, and RO remain distinct keyed entries. Scores are never overwritten
    or averaged across orders.
    """
    from epistemic_sycophancy.feature_selection.artifacts import (
        freeze_feature_selection_artifact,
    )

    required = ("CF", "IF", "RO")
    missing = [order for order in required if order not in artifacts_by_order]
    if missing:
        raise KeyError(f"missing order-specific artifacts for {missing!r}")
    return {
        order: freeze_feature_selection_artifact(
            artifact=artifacts_by_order[order],
            component=component,
            order_regime=order,
            model_revision_hash=model_revision_hash,
            sae_revision_hash=sae_revision_hash,
            scope=scope,
            scale_source=scale_source,
            dataset_manifest_hash=dataset_manifest_hash,
        )
        for order in required
    }
