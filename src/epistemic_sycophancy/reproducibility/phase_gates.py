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
