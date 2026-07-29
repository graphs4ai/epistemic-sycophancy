"""Sealed holdout access (Phase I REPRO-002 / DEC-042)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError


def load_holdout_rows(
    *,
    freeze_status: str,
    frozen_config_artifact: object | None = None,
) -> Sequence[Any]:
    """Load holdout rows only after a sealed frozen config exists (DEC-042).

    Phase I only exercises the sealed (blocked) path — never returns holdout rows.
    """
    del frozen_config_artifact
    if freeze_status != "sealed":
        raise HoldoutAccessError(
            "holdout loader is sealed until FrozenExperimentConfig exists "
            f"(freeze_status={freeze_status!r}; REPRO-002 / DEC-042)"
        )
    # Unlock path would load rows after verified freeze; Phase I does not open holdout.
    raise HoldoutAccessError(
        "holdout unlock is not exercised in Phase I (REPRO-002 sealed-path only)"
    )
