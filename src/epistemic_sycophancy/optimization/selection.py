"""Validation-only checkpoint selection (Phase H OPT-010 / DEC-033)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError

_HOLDOUT_MARKERS = frozenset(
    {
        "holdout",
        "holdout_l_total",
        "holdout_metrics",
        "holdout_selectivity",
        "split",
    }
)


@dataclass(frozen=True)
class ValidationMetricsCandidate:
    """Validation-only metrics for checkpoint selection (no holdout fields)."""

    checkpoint_id: str
    trial_index: int
    l_total: float
    selectivity: float | None = None


def _reject_holdout_payload(payload: Mapping[str, object]) -> None:
    for key in payload:
        key_l = str(key).lower()
        if key_l in _HOLDOUT_MARKERS or "holdout" in key_l:
            raise HoldoutAccessError(
                f"selection cannot access holdout field {key!r} (OPT-010 / DEC-033)"
            )
        if key_l == "split" and str(payload[key]).lower() == "holdout":
            raise HoldoutAccessError(
                "selection cannot reference holdout split (OPT-010 / DEC-033)"
            )


def _coerce_candidate(
    candidate: ValidationMetricsCandidate | Mapping[str, object],
) -> ValidationMetricsCandidate:
    if isinstance(candidate, ValidationMetricsCandidate):
        return candidate
    if isinstance(candidate, Mapping):
        _reject_holdout_payload(candidate)
        allowed = {"checkpoint_id", "trial_index", "l_total", "selectivity"}
        unknown = set(candidate) - allowed
        if unknown:
            # Unknown keys that are not holdout still rejected to keep API narrow
            raise ValueError(f"unsupported selection fields: {sorted(unknown)}")
        return ValidationMetricsCandidate(
            checkpoint_id=str(candidate["checkpoint_id"]),
            trial_index=int(candidate["trial_index"]),
            l_total=float(candidate["l_total"]),
            selectivity=(
                None
                if candidate.get("selectivity") is None
                else float(candidate["selectivity"])  # type: ignore[arg-type]
            ),
        )
    raise TypeError(
        "candidate must be ValidationMetricsCandidate or a validation-only mapping"
    )


def select_best_checkpoint(
    candidates: Sequence[ValidationMetricsCandidate | Mapping[str, object]],
) -> ValidationMetricsCandidate:
    """Select the best checkpoint using validation metrics only (DEC-033)."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    coerced = [_coerce_candidate(c) for c in candidates]

    def sort_key(c: ValidationMetricsCandidate) -> tuple[float, float, int]:
        # Minimize l_total; ties → higher selectivity; else lower trial_index
        selectivity = float("-inf") if c.selectivity is None else -float(c.selectivity)
        return (float(c.l_total), selectivity, int(c.trial_index))

    return min(coerced, key=sort_key)
