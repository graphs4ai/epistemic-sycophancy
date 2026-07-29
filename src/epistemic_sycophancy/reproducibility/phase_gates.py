"""Optimization phase gates (Phase I REPRO-004/005/006)."""

from __future__ import annotations

from collections.abc import Collection

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError


class OptimizationBlockedError(Exception):
    """Raised when a mandatory phase gate blocks optimization entry."""


def require_identity_gate(*, identity_passed: bool) -> None:
    """Block optimization when the SAE identity suite failed (REPRO-004)."""
    if not identity_passed:
        raise OptimizationBlockedError(
            "optimization blocked: SAE identity suite failed (REPRO-004)"
        )


def require_baseline_partition_gate(
    *,
    expected_fingerprint: str,
    actual_fingerprint: str | None,
) -> None:
    """Block optimization when baseline partition is missing/mismatched (REPRO-005)."""
    if actual_fingerprint is None:
        raise OptimizationBlockedError(
            "optimization blocked: missing baseline partition artifact (REPRO-005)"
        )
    if actual_fingerprint != expected_fingerprint:
        raise OptimizationBlockedError(
            "optimization blocked: baseline partition fingerprint mismatch "
            f"(expected={expected_fingerprint!r}, actual={actual_fingerprint!r}; "
            "REPRO-005)"
        )


def require_feature_selection_split_gate(
    *,
    artifact_question_ids: Collection[str],
    feature_selection_question_ids: Collection[str],
    optimization_question_ids: Collection[str],
    validation_question_ids: Collection[str],
    holdout_question_ids: Collection[str],
) -> None:
    """Reject feature artifacts that reference opt/val/holdout IDs (REPRO-006)."""
    fs = set(feature_selection_question_ids)
    forbidden = (
        set(optimization_question_ids)
        | set(validation_question_ids)
        | set(holdout_question_ids)
    )
    for qid in artifact_question_ids:
        if qid not in fs or qid in forbidden:
            raise HoldoutAccessError(
                "feature-selection artifact cannot reference "
                f"optimization/validation/holdout question_id={qid!r} (REPRO-006)"
            )
