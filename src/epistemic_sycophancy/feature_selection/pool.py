"""Suppression candidate-pool eligibility and common-pool construction (Phase F)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.ranking import (
    SuppressionCandidate,
    rank_suppression_candidates,
)


@dataclass(frozen=True)
class EligibilityResult:
    """Eligible suppression candidates plus the override flag that was applied."""

    candidates: tuple[SuppressionCandidate, ...]
    pool_eligibility_override: bool


def eligible_suppression_candidates(
    *,
    signed_jacobians: Mapping[tuple[int, int], float],
    pool_eligibility_override: bool,
) -> EligibilityResult:
    """Filter suppression candidates (FEAT-025 / DEC-019).

    Default (``pool_eligibility_override=False``): keep only
    ``signed_jacobian > 0``. An override must be explicit and is logged on
    the result so the artifact can record it.
    """
    ranked = rank_suppression_candidates(signed_jacobians=signed_jacobians)
    if pool_eligibility_override:
        return EligibilityResult(
            candidates=ranked,
            pool_eligibility_override=True,
        )
    return EligibilityResult(
        candidates=tuple(c for c in ranked if c.signed_jacobian > 0.0),
        pool_eligibility_override=False,
    )
