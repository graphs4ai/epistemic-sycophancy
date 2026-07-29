"""Optimization phase gates (Phase I REPRO-004/005/006)."""

from __future__ import annotations


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
