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


@pytest.mark.unit
def test_phase_gate__failed_identity_test__blocks_optimization() -> None:
    """REPRO-004: failed SAE identity suite blocks optimization entry."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        OptimizationBlockedError,
        require_identity_gate,
    )

    with pytest.raises(OptimizationBlockedError):
        require_identity_gate(identity_passed=False)


@pytest.mark.unit
def test_phase_gate__missing_or_mismatched_baseline_partition__blocks_optimization() -> None:
    """REPRO-005: missing or mismatched baseline partition blocks optimization."""
    from epistemic_sycophancy.reproducibility.phase_gates import (
        OptimizationBlockedError,
        require_baseline_partition_gate,
    )

    with pytest.raises(OptimizationBlockedError):
        require_baseline_partition_gate(
            expected_fingerprint="abc",
            actual_fingerprint=None,
        )
    with pytest.raises(OptimizationBlockedError):
        require_baseline_partition_gate(
            expected_fingerprint="abc",
            actual_fingerprint="xyz",
        )
