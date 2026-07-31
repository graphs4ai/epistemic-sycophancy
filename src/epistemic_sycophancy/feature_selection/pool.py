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


@dataclass(frozen=True)
class CommonFeaturePool:
    """Per-study candidate pool from that order's lists (DEC-019 / DEC-087)."""

    feature_ids: tuple[tuple[int, int], ...]
    scales: tuple[float, ...]


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


def build_common_feature_pool(
    *,
    lists_by_order_and_component: Mapping[
        tuple[str, str], Mapping[tuple[int, int], float]
    ],
    feature_scales: Mapping[tuple[int, int], float],
    pool_quota_per_list: int,
) -> CommonFeaturePool:
    """Build the DEC-019 quota-union pool for the lists supplied by the study.

    For a single-order experiment this is typically two lists
    (resistance/recovery). Keep ``signed_jacobian > 0``, rank descending signed
    Jacobian (ties ascending ``(layer, feature_id)``), take the top
    ``pool_quota_per_list`` (or all if fewer). Union, dedupe by
    ``(layer, feature_id)``, and order the result ascending
    ``(layer, feature_id)``. Fill is a no-op; size equals ``|union|``.
    """
    if not isinstance(pool_quota_per_list, int) or isinstance(
        pool_quota_per_list, bool
    ):
        raise TypeError(
            "pool_quota_per_list must be an explicit positive int; "
            f"got {pool_quota_per_list!r}"
        )
    if pool_quota_per_list <= 0:
        raise ValueError(
            "pool_quota_per_list must be a positive int; "
            f"got {pool_quota_per_list!r}"
        )

    selected: set[tuple[int, int]] = set()
    for scores in lists_by_order_and_component.values():
        eligible = eligible_suppression_candidates(
            signed_jacobians=scores,
            pool_eligibility_override=False,
        )
        for candidate in eligible.candidates[:pool_quota_per_list]:
            selected.add((candidate.layer, candidate.feature_id))

    feature_ids = tuple(sorted(selected, key=lambda key: (key[0], key[1])))
    scales = tuple(float(feature_scales[key]) for key in feature_ids)
    return CommonFeaturePool(feature_ids=feature_ids, scales=scales)
