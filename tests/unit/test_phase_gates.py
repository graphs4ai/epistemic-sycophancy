"""Phase-gate / reproducibility tests (Phase I REPRO)."""

from __future__ import annotations

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.holdout import load_holdout_rows


@pytest.mark.unit
def test_phase_gate__before_freeze__holdout_loader_raises_access_error() -> None:
    """REPRO-002: before frozen config, holdout loader raises HoldoutAccessError."""
    with pytest.raises(HoldoutAccessError):
        load_holdout_rows(freeze_status="unfrozen")
