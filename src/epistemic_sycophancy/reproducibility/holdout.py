"""Sealed holdout access (Phase I REPRO-002 / DEC-042 / Phase M DEC-071)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError


def load_holdout_rows(
    *,
    freeze_status: str,
    frozen_config_artifact: object | None = None,
    rows_provider: Callable[[], Sequence[Any]] | None = None,
) -> Sequence[Any]:
    """Load holdout rows only after sealed freeze + holdout_started (DEC-071)."""
    if freeze_status != "sealed":
        raise HoldoutAccessError(
            "holdout loader is sealed until FrozenExperimentConfig exists "
            f"(freeze_status={freeze_status!r}; REPRO-002 / DEC-042)"
        )
    if frozen_config_artifact is None:
        raise HoldoutAccessError(
            "holdout unlock requires a sealed FrozenExperimentConfig artifact "
            "(DEC-071)"
        )
    holdout_started = bool(getattr(frozen_config_artifact, "holdout_started", False))
    if not holdout_started:
        raise HoldoutAccessError(
            "holdout unlock requires mark_holdout_started "
            "(holdout_started=False; DEC-071)"
        )
    if rows_provider is None:
        raise HoldoutAccessError(
            "holdout rows_provider is required after unlock (DEC-071)"
        )
    return list(rows_provider())
