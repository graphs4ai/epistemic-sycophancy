"""RUN-012: opt smoke finite objective without holdout access."""

from __future__ import annotations

import math

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.phase_gates import (
    OptimizationBlockedError,
    require_identity_gate,
)
from epistemic_sycophancy.runner.opt_smoke import run_opt_smoke


@pytest.mark.unit
def test_runner__opt_smoke__finite_objective_no_holdout_on_tiny_subset() -> None:
    """RUN-012: tiny FS/optimization subset; finite L; holdout sealed; identity gate."""
    require_identity_gate(identity_passed=True)
    result = run_opt_smoke(
        question_ids=("q1", "q2"),
        split_name="optimization",
        beta=(0.0, 0.0),
        freeze_status="unsealed",
        identity_passed=True,
    )
    assert math.isfinite(result.l_total)
    assert result.split_name == "optimization"
    assert result.holdout_accessed is False

    with pytest.raises(HoldoutAccessError):
        run_opt_smoke(
            question_ids=("q1",),
            split_name="holdout_test_behavior",
            beta=(0.0,),
            freeze_status="unsealed",
            identity_passed=True,
        )

    with pytest.raises(OptimizationBlockedError):
        run_opt_smoke(
            question_ids=("q1",),
            split_name="optimization",
            beta=(0.0,),
            freeze_status="unsealed",
            identity_passed=False,
        )
