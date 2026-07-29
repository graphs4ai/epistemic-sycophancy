"""RUN-012 / WIRE-009: opt smoke finite real objective without holdout."""

from __future__ import annotations

import math

import pytest

from epistemic_sycophancy.feature_selection.exceptions import HoldoutAccessError
from epistemic_sycophancy.reproducibility.phase_gates import (
    OptimizationBlockedError,
    require_identity_gate,
)
from epistemic_sycophancy.runner.opt_smoke import run_opt_smoke


def _tiny_objective_kwargs(*, beta: tuple[float, ...]) -> dict:
    return {
        "tau": 1.0,
        "w_r": 0.5,
        "w_u": 0.5,
        "delta_n": 0.0,
        "delta_c": 0.0,
        "lambda_n": 0.0,
        "lambda_c": 0.0,
        "lambda_beta": 0.0,
        "ib_margins_by_question": {"q1": [1.0]},
        "cb_margins_by_question": {"q2": [-0.5], "q1": [1.0]},
        "baseline_cb_margins": {"q1": [1.0], "q2": [-0.5]},
        "baseline_neutral_margins": {"q1": 1.0, "q2": -0.5},
        "current_neutral_margins": {"q1": 1.0, "q2": -0.5},
        "q_plus": ("q1",),
        "q_minus": ("q2",),
        "beta": beta,
    }


@pytest.mark.unit
def test_runner__opt_smoke__finite_objective_no_holdout_on_tiny_subset() -> None:
    """RUN-012: tiny FS/optimization subset; finite L; holdout sealed; identity gate."""
    require_identity_gate(identity_passed=True)
    result = run_opt_smoke(
        question_ids=("q1", "q2"),
        split_name="optimization",
        freeze_status="unsealed",
        identity_passed=True,
        **_tiny_objective_kwargs(beta=(0.0, 0.0)),
    )
    assert math.isfinite(result.l_total)
    assert result.split_name == "optimization"
    assert result.holdout_accessed is False

    with pytest.raises(HoldoutAccessError):
        run_opt_smoke(
            question_ids=("q1",),
            split_name="holdout_test_behavior",
            freeze_status="unsealed",
            identity_passed=True,
            **_tiny_objective_kwargs(beta=(0.0,)),
        )

    with pytest.raises(OptimizationBlockedError):
        run_opt_smoke(
            question_ids=("q1",),
            split_name="optimization",
            freeze_status="unsealed",
            identity_passed=False,
            **_tiny_objective_kwargs(beta=(0.0,)),
        )


@pytest.mark.unit
def test_runner__opt_smoke__evaluate_objective_finite_on_study_smoke_subset() -> None:
    """WIRE-009: real evaluate_objective on smoke subset; finite; no Σβ² surrogate."""
    result = run_opt_smoke(
        question_ids=("q1", "q2"),
        split_name="optimization",
        freeze_status="unsealed",
        identity_passed=True,
        **_tiny_objective_kwargs(beta=(-0.5, -0.25)),
    )
    assert math.isfinite(result.l_total)
    # Surrogate Σβ² would be 0.3125; resistance/recovery path differs.
    assert result.l_total != pytest.approx(0.5**2 + 0.25**2)




@pytest.mark.unit
def test_runner__opt_smoke__one_projected_adam_step_respects_beta_bounds() -> None:
    """WIRE-010: one ProjectedAdam step stays within [beta_lower, beta_upper]."""
    from epistemic_sycophancy.runner.opt_smoke import run_opt_smoke_adam_step

    beta_after = run_opt_smoke_adam_step(
        beta_init=(-0.1, -0.2),
        grad=(-1.0, -1.0),
        adam_lr=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_microbatch_questions=1,
        beta_lower=-2.0,
        beta_upper=0.0,
        max_steps=1,
    )
    assert len(beta_after) == 2
    assert all(-2.0 <= float(b) <= 0.0 for b in beta_after)
