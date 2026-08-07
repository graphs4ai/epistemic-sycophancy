"""Suppression / bidirectional candidate-pool construction (Phase F / DEC-105)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.ranking import (
    SuppressionCandidate,
    rank_suppression_candidates,
)


def _preferred_bidirectional_sign(signed_jacobian: float) -> float:
    if signed_jacobian > 0.0:
        return -1.0
    if signed_jacobian < 0.0:
        return 1.0
    return 0.0


@dataclass(frozen=True)
class EligibilityResult:
    """Eligible candidates plus the override flag that was applied."""

    candidates: tuple[SuppressionCandidate, ...]
    pool_eligibility_override: bool


@dataclass(frozen=True)
class CommonFeaturePool:
    """Per-study candidate pool from that order's lists (DEC-019 / DEC-087 / DEC-105)."""

    feature_ids: tuple[tuple[int, int], ...]
    scales: tuple[float, ...]
    preferred_bidirectional_signs: tuple[float, ...] = ()


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


def rank_bidirectional_candidates(
    *,
    signed_jacobians: Mapping[tuple[int, int], float],
) -> tuple[SuppressionCandidate, ...]:
    """Rank by descending |J|, then ascending (layer, feature_id) (DEC-105)."""
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
        if float(jacobian) != 0.0
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.absolute_sensitivity,
            candidate.layer,
            candidate.feature_id,
        )
    )
    return tuple(candidates)


def eligible_bidirectional_candidates(
    *,
    signed_jacobians: Mapping[tuple[int, int], float],
) -> EligibilityResult:
    """Keep nonzero Jacobians ranked by descending |J| (FEAT-025b / DEC-105)."""
    return EligibilityResult(
        candidates=rank_bidirectional_candidates(signed_jacobians=signed_jacobians),
        pool_eligibility_override=False,
    )


def build_common_feature_pool(
    *,
    lists_by_order_and_component: Mapping[
        tuple[str, str], Mapping[tuple[int, int], float]
    ],
    feature_scales: Mapping[tuple[int, int], float],
    pool_quota_per_list: int,
    coefficient_mode: str = "suppression",
) -> CommonFeaturePool:
    """Build the quota-union pool for the lists supplied by the study.

    Suppression (DEC-019): keep ``signed_jacobian > 0``, rank descending signed
    Jacobian. Bidirectional (DEC-105): keep ``|J| > 0``, rank descending ``|J|``.
    Then take the top ``pool_quota_per_list`` per list, union, dedupe by
    ``(layer, feature_id)``, and order ascending ``(layer, feature_id)``.
    """
    if coefficient_mode not in ("suppression", "bidirectional"):
        raise ValueError(
            "coefficient_mode must be 'suppression' or 'bidirectional'; "
            f"got {coefficient_mode!r}"
        )
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
    # Track max-|J| signed value across nominating lists for preferred sign.
    best_abs_signed: dict[tuple[int, int], float] = {}
    for scores in lists_by_order_and_component.values():
        if coefficient_mode == "bidirectional":
            eligible = eligible_bidirectional_candidates(signed_jacobians=scores)
        else:
            eligible = eligible_suppression_candidates(
                signed_jacobians=scores,
                pool_eligibility_override=False,
            )
        for candidate in eligible.candidates[:pool_quota_per_list]:
            key = (candidate.layer, candidate.feature_id)
            selected.add(key)
            signed = float(candidate.signed_jacobian)
            prev = best_abs_signed.get(key)
            if prev is None or abs(signed) > abs(prev):
                best_abs_signed[key] = signed

    feature_ids = tuple(sorted(selected, key=lambda key: (key[0], key[1])))
    scales = tuple(float(feature_scales[key]) for key in feature_ids)
    preferred = tuple(
        _preferred_bidirectional_sign(best_abs_signed[key]) for key in feature_ids
    )
    return CommonFeaturePool(
        feature_ids=feature_ids,
        scales=scales,
        preferred_bidirectional_signs=preferred,
    )
